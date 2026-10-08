"""Encode observable attack context for the PPO policy."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch


class StateEncoder:
    """Encode a 50-dimensional cross-dataset attack state.

    Layout:
      0..10  current target node type one-hot (11 slots)
      11     current anomaly score / run score scale
      12     current threshold-margin rank
      13..14 current in/out degree (log scaled)
      15     cumulative added edges for this target
      16     fraction of target modifications that are incoming edges
      17     remaining per-target action budget
      18..28 previous feedback target type one-hot (11 slots)
      29..37 previous feedback action one-hot
      38     previous target score drop, clipped to [-1, 1]
      39     previous target crossed the detector threshold
      40     previous batch reward, tanh scaled
      41..49 recent action histogram (probability distribution)

    "Previous" means the latest detector-feedback batch, which can belong to
    the current target or to the previously attacked target.  No labels or
    metadata are included.
    """

    # TRACE uses 11 node features, CADETS 6, and THEIA 5.  A shared width
    # prevents checkpoint architecture changes when datasets are switched.
    NODE_TYPE_DIM = 11
    ACTION_DIM = 9
    BASE_DIM = 18
    STATE_DIM = 50
    MAX_ANOMALY_SCORE = None

    @classmethod
    def feature_names(cls):
        names = [f"target_type_{idx}" for idx in range(cls.NODE_TYPE_DIM)]
        names.extend(
            [
                "current_score",
                "current_margin_rank",
                "current_in_degree",
                "current_out_degree",
                "added_edges",
                "incoming_edge_fraction",
                "remaining_budget",
            ]
        )
        names.extend(
            f"previous_target_type_{idx}" for idx in range(cls.NODE_TYPE_DIM)
        )
        names.extend(
            f"previous_action_{idx}" for idx in range(cls.ACTION_DIM)
        )
        names.extend(
            [
                "previous_score_drop",
                "previous_threshold_cross",
                "previous_reward",
            ]
        )
        names.extend(
            f"recent_action_fraction_{idx}" for idx in range(cls.ACTION_DIM)
        )
        return names

    @staticmethod
    def set_max_anomaly_score(max_score):
        StateEncoder.MAX_ANOMALY_SCORE = float(max_score)
        print(
            "[StateEncoder] 设置MAX_ANOMALY_SCORE = "
            f"{StateEncoder.MAX_ANOMALY_SCORE:.4f}"
        )

    @staticmethod
    def _node_type(graph, node_id: int) -> int:
        node_feat = graph.ndata["attr"][int(node_id)]
        return int(torch.argmax(node_feat).item())

    @staticmethod
    def _score_scale() -> float:
        value = StateEncoder.MAX_ANOMALY_SCORE
        return float(value) if value is not None and value > 0 else 200.0

    @classmethod
    def encode_state(
        cls,
        graph,
        target_node,
        current_score,
        current_rank,
        action_history,
        budget_info,
        attack_state=None,
        top_k=30,
        previous_context: Optional[Dict] = None,
    ):
        del action_history  # retained for API compatibility
        state = torch.zeros(cls.STATE_DIM, dtype=torch.float32)

        try:
            target_type = cls._node_type(graph, int(target_node))
            if 0 <= target_type < cls.NODE_TYPE_DIM:
                state[target_type] = 1.0
        except Exception:
            state[0] = 1.0

        score_scale = cls._score_scale()
        state[11] = max(
            0.0, min(float(current_score) / score_scale, 1.0)
        )
        top_k = max(int(top_k), 1)
        state[12] = max(
            0.0, min(1.0 - (float(current_rank) - 1.0) / top_k, 1.0)
        )

        try:
            in_degree = float(graph.in_degrees(int(target_node)))
            out_degree = float(graph.out_degrees(int(target_node)))
            max_deg = max(
                1, min(int(graph.number_of_nodes()) // 10, 10000)
            )
            denominator = math.log(max_deg + 1.0)
            state[13] = min(math.log(in_degree + 1.0) / denominator, 1.0)
            state[14] = min(math.log(out_degree + 1.0) / denominator, 1.0)
        except Exception:
            pass

        max_steps = 10.0
        if attack_state is not None:
            added_edges = float(
                getattr(attack_state, "num_added_edges", 0)
            )
            state[15] = min(
                added_edges / max_steps,
                1.0,
            )
            if added_edges > 0:
                state[16] = min(
                    float(
                        getattr(attack_state, "num_incoming_edges", 0)
                    ) / added_edges,
                    1.0,
                )
        if budget_info:
            state[17] = max(
                0.0, min(float(budget_info[0]) / max_steps, 1.0)
            )

        context = previous_context or {}
        previous_target_type = int(context.get("target_type", -1))
        if 0 <= previous_target_type < cls.NODE_TYPE_DIM:
            state[18 + previous_target_type] = 1.0

        previous_action = int(context.get("action_type", -1))
        if 0 <= previous_action < cls.ACTION_DIM:
            state[29 + previous_action] = 1.0

        score_drop = float(context.get("score_drop", 0.0))
        state[38] = max(-1.0, min(score_drop / score_scale, 1.0))
        state[39] = (
            1.0 if context.get("crossed_threshold", False) else 0.0
        )
        state[40] = math.tanh(float(context.get("reward", 0.0)))

        histogram = list(context.get("recent_action_histogram", ()))
        if len(histogram) < cls.ACTION_DIM:
            histogram.extend([0.0] * (cls.ACTION_DIM - len(histogram)))
        histogram = histogram[: cls.ACTION_DIM]
        total = sum(max(0.0, float(value)) for value in histogram)
        if total > 0:
            for action_id, count in enumerate(histogram):
                state[41 + action_id] = max(0.0, float(count)) / total

        return state

    @staticmethod
    def compute_max_anomaly_score(target_nodes, query_result):
        all_scores = query_result["all_scores"]
        target_scores = [
            float(all_scores.get(node_id, 0.0))
            for node_id in target_nodes
            if float(all_scores.get(node_id, 0.0)) > 0
        ]
        return max(target_scores) * 1.5 if target_scores else 200.0
