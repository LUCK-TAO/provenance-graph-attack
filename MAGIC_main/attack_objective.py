"""Minimal detector-agnostic, label-free reward for graph attacks."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Hashable


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    direct_score: float
    threshold_cross: float
    indirect_score_drop: float
    indirect_cross: float
    indirect_cross_count: int
    indirect_positive_count: int
    indirect_candidate_count: int
    indirect_reward: float
    perturbation_cost: float


def compute_blackbox_reward(
    *,
    target_node: Hashable,
    previous_scores: Mapping[Hashable, float],
    current_scores: Mapping[Hashable, float],
    threshold: float,
    added_edges: int,
    batch_actions: int,
    direct_weight: float = 1.0,
    crossing_weight: float = 0.5,
    indirect_node_ids: Iterable[Hashable] = (),
    indirect_weight: float = 0.2,
    indirect_top_k: int = 5,
    score_epsilon: float = 1e-6,
    perturbation_weight: float = 0.05,
) -> RewardBreakdown:
    """Compute a detector-only reward for one queried macro action.

    ``indirect_node_ids`` must be observable nodes that were not directly
    attacked.  No ground-truth labels are accepted.  Indirect score reduction
    uses the mean of the largest ``indirect_top_k`` positive normalized drops,
    while indirect threshold crossings use a bounded count.  A node that has
    already crossed is not rewarded again because it is no longer above the
    threshold in ``previous_scores``.
    """
    previous_target = float(previous_scores.get(target_node, 0.0))
    current_target = float(current_scores.get(target_node, previous_target))
    scale = max(abs(previous_target), abs(threshold), score_epsilon)
    direct_score = max(-1.0, min(1.0, (previous_target - current_target) / scale))
    threshold_cross = float(previous_target >= threshold > current_target)

    indirect_candidates = tuple(dict.fromkeys(indirect_node_ids))
    positive_drops = []
    indirect_cross_count = 0
    for node_id in indirect_candidates:
        if node_id not in previous_scores or node_id not in current_scores:
            continue
        previous_score = float(previous_scores[node_id])
        current_score = float(current_scores[node_id])
        if previous_score < threshold:
            continue
        node_scale = max(abs(previous_score), abs(threshold), score_epsilon)
        normalized_drop = max(
            0.0,
            min(1.0, (previous_score - current_score) / node_scale),
        )
        if normalized_drop > score_epsilon:
            positive_drops.append(normalized_drop)
        if current_score < threshold:
            indirect_cross_count += 1

    indirect_top_k = max(int(indirect_top_k), 1)
    strongest_drops = sorted(positive_drops, reverse=True)[:indirect_top_k]
    indirect_score_drop = (
        sum(strongest_drops) / len(strongest_drops)
        if strongest_drops else 0.0
    )
    indirect_cross = math.tanh(indirect_cross_count / 2.0)
    indirect_reward = 0.5 * indirect_score_drop + 0.5 * indirect_cross

    perturbation_cost = max(added_edges, 0) / max(batch_actions, 1)

    total = (
        direct_weight * direct_score
        + crossing_weight * threshold_cross
        + indirect_weight * indirect_reward
        - perturbation_weight * perturbation_cost
    )
    total = max(-5.0, min(5.0, total))
    return RewardBreakdown(
        total=total,
        direct_score=direct_score,
        threshold_cross=threshold_cross,
        indirect_score_drop=indirect_score_drop,
        indirect_cross=indirect_cross,
        indirect_cross_count=indirect_cross_count,
        indirect_positive_count=len(positive_drops),
        indirect_candidate_count=len(indirect_candidates),
        indirect_reward=indirect_reward,
        perturbation_cost=perturbation_cost,
    )
