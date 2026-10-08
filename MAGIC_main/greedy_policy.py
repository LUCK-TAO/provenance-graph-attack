"""Query-budget-fair greedy baseline for legal graph actions."""

from __future__ import annotations

import math


class GreedyActionPolicy:
    """Explore every legal action once, then exploit empirical batch reward.

    The policy uses only the same detector feedback batches available to PPO;
    it never performs extra look-ahead detector queries.  Candidate choice is
    deterministic: lowest observable anomaly score, then lowest graph degree,
    then node id.
    """

    def __init__(self, action_dim: int):
        self.action_dim = int(action_dim)
        self.counts = [0] * self.action_dim
        self.reward_sums = [0.0] * self.action_dim

    def mean_reward(self, action_type: int) -> float:
        count = self.counts[int(action_type)]
        return self.reward_sums[int(action_type)] / count if count else 0.0

    def select_action_type(self, legal_action_types):
        legal = sorted(set(int(value) for value in legal_action_types))
        if not legal:
            raise ValueError("greedy policy requires at least one legal action")
        unexplored = [value for value in legal if self.counts[value] == 0]
        if unexplored:
            return unexplored[0]
        return max(
            legal,
            key=lambda value: (
                self.mean_reward(value),
                -value,
            ),
        )

    @staticmethod
    def select_candidate_index(candidates, anomaly_scores, graph):
        if not candidates:
            raise ValueError("greedy policy requires at least one candidate")
        scores = anomaly_scores or {}

        def candidate_key(item):
            index, node_id = item
            score = float(scores.get(int(node_id), math.inf))
            try:
                degree = int(
                    graph.in_degrees(int(node_id)).item()
                    + graph.out_degrees(int(node_id)).item()
                )
            except Exception:
                degree = math.inf
            return score, degree, int(node_id), index

        return min(enumerate(candidates), key=candidate_key)[0]

    def observe_batch(self, action_types, total_reward: float):
        actions = [int(value) for value in action_types]
        if not actions:
            return self.snapshot()
        credit = float(total_reward) / len(actions)
        for action_type in actions:
            self.counts[action_type] += 1
            self.reward_sums[action_type] += credit
        return self.snapshot()

    def snapshot(self):
        return {
            "counts": list(self.counts),
            "mean_rewards": [self.mean_reward(i) for i in range(self.action_dim)],
        }
