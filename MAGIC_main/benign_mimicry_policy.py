"""Detector-observable benign-mimicry action baseline.

This is a query-budget-matched, edge-only adaptation of distributional
mimicry, not a verbatim reproduction of detector-specific subgraph gadgets.
The policy never reads metadata labels and never issues extra detector
queries.  It estimates, once on the clean attack graph, the incident semantic
relation distribution of nodes that the detector did *not* flag.  For every
target it then chooses the legal macro action whose simulated edge additions
make the target's incident-relation distribution closest to that reference.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch


class BenignMimicryActionPolicy:
    """Choose legal actions that imitate detector-predicted benign structure."""

    def __init__(self, action_schema, macro_action_length: int = 5):
        self.action_schema = action_schema
        self.action_dim = int(action_schema.action_dim)
        self.macro_action_length = int(macro_action_length)
        self.reference_counts = None
        self.reference_probabilities = None
        self.node_types = None
        self.triple_to_action = {}
        self.selection_counts = [0] * self.action_dim
        self.last_action_stats = {}
        for spec in action_schema.actions:
            for src_type, dst_type in spec.valid_pairs:
                self.triple_to_action[
                    (int(src_type), int(spec.edge_type), int(dst_type))
                ] = int(spec.action_id)

    def fit(self, graph, excluded_alert_nodes: Iterable[int] = ()):
        """Estimate per-node-type relation profiles from non-alert nodes."""
        node_types = torch.argmax(graph.ndata["attr"], dim=1).cpu().long()
        edge_types = torch.argmax(graph.edata["attr"], dim=1).cpu().long()
        src, dst = graph.edges(order="eid")
        src = src.cpu().long()
        dst = dst.cpu().long()
        node_type_dim = int(graph.ndata["attr"].shape[1])
        counts = torch.zeros(
            (node_type_dim, self.action_dim), dtype=torch.float64
        )
        predicted_benign = torch.ones(
            graph.number_of_nodes(), dtype=torch.bool
        )
        excluded = [
            int(node_id) for node_id in excluded_alert_nodes
            if 0 <= int(node_id) < graph.number_of_nodes()
        ]
        if excluded:
            predicted_benign[torch.tensor(excluded, dtype=torch.long)] = False

        for spec in self.action_schema.actions:
            action_id = int(spec.action_id)
            edge_mask = edge_types == int(spec.edge_type)
            if not bool(edge_mask.any()):
                continue
            for src_type, dst_type in spec.valid_pairs:
                mask = (
                    edge_mask
                    & (node_types[src] == int(src_type))
                    & (node_types[dst] == int(dst_type))
                )
                if not bool(mask.any()):
                    continue
                matched_src = src[mask]
                matched_dst = dst[mask]
                counts[int(src_type), action_id] += float(
                    predicted_benign[matched_src].sum().item()
                )
                counts[int(dst_type), action_id] += float(
                    predicted_benign[matched_dst].sum().item()
                )

        probabilities = torch.zeros_like(counts)
        for node_type in range(node_type_dim):
            legal = list(self.action_schema.legal_action_ids(node_type))
            if not legal:
                continue
            legal_counts = counts[node_type, legal]
            total = float(legal_counts.sum().item())
            if total > 0.0:
                probabilities[node_type, legal] = legal_counts / total
            else:
                probabilities[node_type, legal] = 1.0 / len(legal)

        self.node_types = node_types
        self.reference_counts = counts
        self.reference_probabilities = probabilities
        benign_type_counts = torch.bincount(
            node_types[predicted_benign], minlength=node_type_dim
        )
        return {
            "predicted_benign_nodes": int(predicted_benign.sum().item()),
            "excluded_alert_nodes": len(excluded),
            "benign_nodes_by_type": [
                int(value) for value in benign_type_counts.tolist()
            ],
            "reference_probabilities": [
                [float(value) for value in row]
                for row in probabilities.tolist()
            ],
        }

    def _target_action_counts(self, graph, target_node: int):
        if self.node_types is None:
            raise RuntimeError("benign-mimicry policy must be fitted first")
        target_node = int(target_node)
        try:
            incoming = graph.in_edges(target_node, form="eid").cpu().tolist()
            outgoing = graph.out_edges(target_node, form="eid").cpu().tolist()
        except TypeError:
            incoming = graph.in_edges(target_node, form="all")[2].cpu().tolist()
            outgoing = graph.out_edges(target_node, form="all")[2].cpu().tolist()
        edge_ids = sorted(set(int(value) for value in incoming + outgoing))
        counts = [0.0] * self.action_dim
        for edge_id in edge_ids:
            src, dst = graph.find_edges(edge_id)
            src = int(src.reshape(-1)[0].item())
            dst = int(dst.reshape(-1)[0].item())
            edge_type = int(
                torch.argmax(graph.edata["attr"][edge_id]).item()
            )
            triple = (
                int(self.node_types[src]), edge_type, int(self.node_types[dst])
            )
            action_id = self.triple_to_action.get(triple)
            if action_id is not None:
                counts[action_id] += 1.0
        return counts

    def select_action(
        self,
        graph,
        target_node: int,
        legal_action_types: Sequence[int],
        additions: Optional[int] = None,
    ) -> int:
        if self.reference_probabilities is None or self.node_types is None:
            raise RuntimeError("benign-mimicry policy must be fitted first")
        legal = sorted(set(int(value) for value in legal_action_types))
        if not legal:
            raise ValueError("benign-mimicry requires at least one legal action")
        target_type = int(self.node_types[int(target_node)])
        reference = [
            float(self.reference_probabilities[target_type, action_id])
            for action_id in range(self.action_dim)
        ]
        reference_total = sum(reference[action_id] for action_id in legal)
        if reference_total <= 0.0:
            for action_id in legal:
                reference[action_id] = 1.0 / len(legal)
        else:
            for action_id in legal:
                reference[action_id] /= reference_total

        current_counts = self._target_action_counts(graph, int(target_node))
        simulated_additions = max(
            1,
            int(
                self.macro_action_length
                if additions is None else additions
            ),
        )
        selection_scores = {}
        simulated_distances = {}
        current_total = sum(current_counts[action_id] for action_id in legal)
        for action_id in legal:
            post_total = current_total + simulated_additions
            distance = 0.0
            for candidate_action in legal:
                post_count = current_counts[candidate_action]
                if candidate_action == action_id:
                    post_count += simulated_additions
                post_probability = post_count / post_total
                distance += abs(
                    post_probability - reference[candidate_action]
                )
            simulated_distances[action_id] = distance
            selection_scores[action_id] = -distance

        selected = max(
            legal,
            key=lambda action_id: (
                selection_scores[action_id],
                reference[action_id],
                -current_counts[action_id],
                -action_id,
            ),
        )
        self.selection_counts[selected] += 1
        self.last_action_stats = {
            "selection_mode": "benign_mimicry",
            "warmup": False,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "max_action_probability": 1.0,
            "top1_top2_probability_gap": 1.0,
            "kl_to_uniform": 0.0,
            "legal_probabilities": {
                action_id: float(action_id == selected)
                for action_id in legal
            },
            "selection_scores": {
                action_id: round(selection_scores[action_id], 6)
                for action_id in legal
            },
            "reference_probabilities": {
                action_id: round(reference[action_id], 6)
                for action_id in legal
            },
            "target_action_counts": {
                action_id: int(current_counts[action_id])
                for action_id in legal
            },
            "simulated_l1_distance": {
                action_id: round(simulated_distances[action_id], 6)
                for action_id in legal
            },
            "selected_action": int(selected),
        }
        return int(selected)

    def snapshot(self):
        return {
            "selection_counts": list(self.selection_counts),
            "last_action_stats": dict(self.last_action_stats),
        }
