"""Detector-agnostic target ranking helpers for threshold-based attacks."""

from collections.abc import Mapping
import math
import random
from typing import Hashable, Optional


def rank_alerts_by_threshold_margin(
    all_scores: Mapping[Hashable, float], threshold: float
) -> list[Hashable]:
    """Return current alerts from closest to farthest above ``threshold``.

    Ground-truth labels are deliberately not accepted by this API.  Node IDs
    are used only as a deterministic tie-break when two scores are equal.
    """
    alerts = (
        (node_id, float(score))
        for node_id, score in all_scores.items()
        if float(score) >= threshold
    )
    return [
        node_id
        for node_id, _ in sorted(
            alerts, key=lambda item: (item[1] - threshold, item[0])
        )
    ]


def rank_alerts(
    all_scores: Mapping[Hashable, float],
    threshold: float,
    strategy: str = "threshold_margin",
    rng: Optional[random.Random] = None,
    limit: Optional[int] = None,
    margin_fraction: float = 0.5,
) -> list[Hashable]:
    """Rank current alerts using a reproducible detector-only strategy.

    ``threshold_margin`` is the proposed method, while ``highest_score`` and
    ``random_alert`` provide target-selection baselines. Ground-truth labels
    are deliberately absent from this API.
    """
    if strategy == "threshold_margin":
        return rank_alerts_by_threshold_margin(all_scores, threshold)

    if strategy == "hybrid_margin_random":
        if limit is None or limit <= 0:
            raise ValueError("hybrid_margin_random requires a positive limit")
        if not 0.0 <= margin_fraction <= 1.0:
            raise ValueError("margin_fraction must be within [0, 1]")

        margin_ranked = rank_alerts_by_threshold_margin(all_scores, threshold)
        margin_count = min(
            len(margin_ranked), int(math.ceil(limit * margin_fraction))
        )
        selected_margin = margin_ranked[:margin_count]
        selected_set = set(selected_margin)
        random_pool = [
            node_id for node_id in margin_ranked if node_id not in selected_set
        ]
        (rng or random).shuffle(random_pool)
        random_count = max(0, min(limit, len(margin_ranked)) - margin_count)
        return selected_margin + random_pool[:random_count]

    alerts = [
        node_id for node_id, score in all_scores.items()
        if float(score) >= threshold
    ]
    if strategy == "highest_score":
        return sorted(alerts, key=lambda node_id: (-float(all_scores[node_id]), node_id))
    if strategy == "random_alert":
        ranked = sorted(alerts)
        (rng or random).shuffle(ranked)
        return ranked
    raise ValueError(
        "strategy must be one of: threshold_margin, highest_score, "
        "random_alert, hybrid_margin_random"
    )


def get_threshold_margin_rank(
    node_id: Hashable,
    all_scores: Mapping[Hashable, float],
    threshold: float,
) -> int:
    """Return a 1-based rank where rank 1 is closest above the threshold.

    A node that is absent or has already crossed below the threshold is ranked
    after every current alert.  The implementation is linear because attacks
    query this value repeatedly on large graphs.
    """
    if node_id not in all_scores:
        return sum(float(score) >= threshold for score in all_scores.values()) + 1

    target_score = float(all_scores[node_id])
    if target_score < threshold:
        return sum(float(score) >= threshold for score in all_scores.values()) + 1

    closer = 0
    tied_before = 0
    for other_id, score in all_scores.items():
        score = float(score)
        if threshold <= score < target_score:
            closer += 1
        elif score == target_score and other_id < node_id:
            tied_before += 1
    return closer + tied_before + 1
