"""Semantically constrained candidate-node selection."""

from __future__ import annotations

import random
import heapq
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch

try:
    from .action_schema import ActionSchema, get_action_schema
except ImportError:  # direct script execution
    from action_schema import ActionSchema, get_action_schema


class CandidateSelector:
    """Select legal endpoints for a dataset-specific action schema.

    Legality is checked using the complete
    ``(source type, edge type, destination type)`` triple.  Existing and
    pending duplicate directed edges are excluded.
    """

    def __init__(
        self,
        max_hop: int = 2,
        local_ratio: float = 0.5,
        action_schema: Optional[ActionSchema] = None,
        dataset_name: str = "cadets",
        selection_mode: str = "semantic_mixed",
        seed: int = 42,
        **_legacy_kwargs,
    ):
        if selection_mode not in {"semantic_mixed", "random_legal"}:
            raise ValueError(
                f"invalid candidate selection mode: {selection_mode}"
            )
        self.max_hop = int(max_hop)
        self.local_ratio = float(local_ratio)
        self.action_schema = action_schema or get_action_schema(
            dataset_name, allow_observed_fallback=True
        )
        self.selection_mode = selection_mode
        self.rng = random.Random(int(seed) + 3001)
        self.type_index: Optional[Dict[int, List[int]]] = None
        self.node_types: Optional[List[int]] = None
        self._neighborhood_cache_key = None
        self._cached_local_nodes: Optional[Set[int]] = None
        self._cached_successors: Optional[Set[int]] = None
        self._cached_predecessors: Optional[Set[int]] = None

    def set_action_schema(self, action_schema: ActionSchema) -> None:
        self.action_schema = action_schema

    def build_type_index(self, graph) -> Dict[int, List[int]]:
        if (
            self.type_index is not None
            and self.node_types is not None
            and len(self.node_types) == graph.number_of_nodes()
        ):
            return self.type_index

        print("[CandidateSelector] 建立节点类型索引...")
        type_tensor = torch.argmax(graph.ndata["attr"], dim=1)
        self.node_types = [int(value) for value in type_tensor.cpu().tolist()]
        self.type_index = {}
        for node_id, node_type in enumerate(self.node_types):
            self.type_index.setdefault(node_type, []).append(node_id)
        print(
            "[CandidateSelector] 类型索引建立完成 - "
            f"共{len(self.type_index)}种类型"
        )
        return self.type_index

    def get_node_type(self, graph, node_id: int) -> int:
        self.build_type_index(graph)
        if self.node_types is None or not 0 <= int(node_id) < len(self.node_types):
            raise IndexError(f"invalid node ID {node_id}")
        return int(self.node_types[int(node_id)])

    def _get_khop_neighbors(
        self, graph, target_node: int, k: Optional[int] = None
    ) -> Set[int]:
        visited = {int(target_node)}
        current_level = {int(target_node)}
        for _ in range(self.max_hop if k is None else int(k)):
            next_level: Set[int] = set()
            for node in current_level:
                predecessors = graph.predecessors(node).cpu().tolist()
                successors = graph.successors(node).cpu().tolist()
                for neighbor in predecessors + successors:
                    neighbor = int(neighbor)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
            current_level = next_level
            if not current_level:
                break
        visited.discard(int(target_node))
        return visited

    def _semantic_population(
        self, target_node: int, candidate_types: Iterable[int]
    ) -> List[int]:
        if self.type_index is None:
            raise RuntimeError("build_type_index() must be called first")
        population: List[int] = []
        for node_type in sorted(set(int(value) for value in candidate_types)):
            population.extend(self.type_index.get(node_type, ()))
        return population

    @staticmethod
    def _pending_pairs(
        pending_edges: Optional[Iterable[Tuple[int, int]]]
    ) -> Set[Tuple[int, int]]:
        if not pending_edges:
            return set()
        return {(int(src), int(dst)) for src, dst in pending_edges}

    def select_candidates(
        self,
        graph,
        target_node: int,
        action_type: int,
        top_k: int = 50,
        anomaly_scores=None,
        pending_edges=None,
        max_candidate_score: Optional[float] = None,
        _local_nodes: Optional[Set[int]] = None,
        _successors: Optional[Set[int]] = None,
        _predecessors: Optional[Set[int]] = None,
    ) -> List[int]:
        if self.action_schema is None:
            raise RuntimeError(
                "candidate selector requires an action schema before use"
            )
        self.build_type_index(graph)
        target_node = int(target_node)
        target_type = self.get_node_type(graph, target_node)
        valid_types = self.action_schema.candidate_types(
            int(action_type), target_type
        )
        if not valid_types:
            return []

        if self.type_index is None or self.node_types is None:
            raise RuntimeError("node type index is unavailable")
        semantic_stream = lambda: heapq.merge(
            *(self.type_index.get(node_type, ()) for node_type in valid_types)
        )
        pending = self._pending_pairs(pending_edges)
        if _successors is None or _predecessors is None:
            try:
                successors = {
                    int(value)
                    for value in graph.successors(target_node).cpu().tolist()
                }
                predecessors = {
                    int(value)
                    for value in graph.predecessors(target_node).cpu().tolist()
                }
            except Exception:
                successors, predecessors = set(), set()
        else:
            successors = _successors
            predecessors = _predecessors

        def is_legal(candidate):
            candidate = int(candidate)
            if candidate == target_node:
                return False
            if max_candidate_score is not None:
                if anomaly_scores is None or candidate not in anomaly_scores:
                    return False
                if float(anomaly_scores[candidate]) >= float(max_candidate_score):
                    return False
            candidate_type = int(self.node_types[candidate])
            resolution = self.action_schema.resolve(
                int(action_type), target_type, candidate_type
            )
            if resolution is None:
                return False
            target_is_source, _ = resolution
            pair = (
                (target_node, candidate)
                if target_is_source
                else (candidate, target_node)
            )
            if pair in pending:
                return False
            if target_is_source and candidate in successors:
                return False
            if not target_is_source and candidate in predecessors:
                return False
            return True

        top_k = max(0, int(top_k))
        if top_k == 0:
            return []
        if self.selection_mode == "random_legal":
            # Reservoir sampling avoids materializing hundreds of thousands of
            # candidates for large TRACE graphs.
            reservoir = []
            legal_seen = 0
            for candidate in semantic_stream():
                if not is_legal(candidate):
                    continue
                legal_seen += 1
                if len(reservoir) < top_k:
                    reservoir.append(int(candidate))
                else:
                    replacement = self.rng.randrange(legal_seen)
                    if replacement < top_k:
                        reservoir[replacement] = int(candidate)
            return reservoir

        local_nodes = (
            self._get_khop_neighbors(graph, target_node)
            if _local_nodes is None
            else _local_nodes
        )
        local_limit = int(top_k * self.local_ratio)
        local = sorted(
            node
            for node in local_nodes
            if (
                0 <= int(node) < len(self.node_types)
                and int(self.node_types[int(node)]) in valid_types
                and is_legal(node)
            )
        )
        selected_local = local[:local_limit]
        selected_local_set = set(selected_local)
        selected_global = []
        global_limit = top_k - len(selected_local)
        for candidate in semantic_stream():
            candidate = int(candidate)
            if candidate in selected_local_set or not is_legal(candidate):
                continue
            selected_global.append(candidate)
            if len(selected_global) >= global_limit:
                break
        return selected_local + selected_global

    def legal_candidate_pools(
        self,
        graph,
        target_node: int,
        top_k: int = 50,
        anomaly_scores=None,
        pending_edges=None,
        max_candidate_score: Optional[float] = None,
    ) -> Dict[int, List[int]]:
        pools = {}
        if self.action_schema is None:
            raise RuntimeError(
                "candidate selector requires an action schema before use"
            )
        self.build_type_index(graph)
        target_node = int(target_node)
        target_type = self.get_node_type(graph, target_node)

        # Every action uses the same target adjacency and two-hop neighborhood.
        # Computing these once is important for high-degree alert nodes: their
        # two-hop neighborhood can cover a large fraction of the graph.
        successors: Optional[Set[int]] = None
        predecessors: Optional[Set[int]] = None
        local_nodes: Optional[Set[int]] = None
        has_legal_action = any(
            self.action_schema.candidate_types(action_id, target_type)
            for action_id in range(self.action_schema.action_dim)
        )
        cache_key = (
            target_node,
            int(graph.number_of_nodes()),
            int(graph.number_of_edges()),
            self.max_hop,
            self.selection_mode,
        )
        if has_legal_action and cache_key == self._neighborhood_cache_key:
            successors = self._cached_successors
            predecessors = self._cached_predecessors
            local_nodes = self._cached_local_nodes
        elif has_legal_action:
            try:
                successors = {
                    int(value)
                    for value in graph.successors(target_node).cpu().tolist()
                }
                predecessors = {
                    int(value)
                    for value in graph.predecessors(target_node).cpu().tolist()
                }
            except Exception:
                successors, predecessors = set(), set()
            if self.selection_mode == "semantic_mixed":
                local_nodes = self._get_khop_neighbors(graph, target_node)
            self._neighborhood_cache_key = cache_key
            self._cached_successors = successors
            self._cached_predecessors = predecessors
            self._cached_local_nodes = local_nodes

        for action_id in range(self.action_schema.action_dim):
            candidates = self.select_candidates(
                graph,
                target_node,
                action_id,
                top_k=top_k,
                anomaly_scores=anomaly_scores,
                pending_edges=pending_edges,
                max_candidate_score=max_candidate_score,
                _local_nodes=local_nodes,
                _successors=successors,
                _predecessors=predecessors,
            )
            if candidates:
                pools[action_id] = candidates
        return pools

    def resolve_operation(
        self, graph, target_node: int, action_type: int, candidate_node: int
    ) -> Dict[str, int]:
        target_type = self.get_node_type(graph, int(target_node))
        if self.action_schema is None:
            raise RuntimeError(
                "candidate selector requires an action schema before use"
            )
        candidate_type = self.get_node_type(graph, int(candidate_node))
        resolution = self.action_schema.resolve(
            int(action_type), target_type, candidate_type
        )
        if resolution is None:
            raise ValueError(
                "illegal action triple: "
                f"action={action_type}, target_type={target_type}, "
                f"candidate_type={candidate_type}"
            )
        target_is_source, pair = resolution
        src = int(target_node) if target_is_source else int(candidate_node)
        dst = int(candidate_node) if target_is_source else int(target_node)
        spec = self.action_schema.get(int(action_type))
        return {
            "src": src,
            "dst": dst,
            "src_type": int(pair[0]),
            "dst_type": int(pair[1]),
            "edge_type": int(spec.edge_type),
        }

    def clear_cache(self) -> None:
        # The original node type index remains valid because this action
        # vocabulary currently adds edges only.
        self._neighborhood_cache_key = None
        self._cached_local_nodes = None
        self._cached_successors = None
        self._cached_predecessors = None
