"""Fast regression checks for the label-free black-box reward."""

from __future__ import annotations

import math

from attack_objective import compute_blackbox_reward


def main() -> None:
    reward = compute_blackbox_reward(
        target_node=0,
        previous_scores={0: 10.0, 1: 9.0, 2: 10.0, 3: 7.0},
        current_scores={0: 9.0, 1: 7.0, 2: 9.0, 3: 6.0},
        threshold=8.0,
        added_edges=5,
        batch_actions=5,
        indirect_node_ids={1, 2, 3},
    )
    expected_indirect = (
        0.5 * ((2.0 / 9.0 + 0.1) / 2.0)
        + 0.5 * math.tanh(0.5)
    )
    expected_total = 0.1 + 0.2 * expected_indirect - 0.05
    assert abs(reward.total - expected_total) < 1e-12
    assert reward.indirect_cross_count == 1
    assert reward.indirect_positive_count == 2
    assert reward.indirect_candidate_count == 3

    direct_only = compute_blackbox_reward(
        target_node=0,
        previous_scores={0: 10.0},
        current_scores={0: 9.0},
        threshold=8.0,
        added_edges=5,
        batch_actions=5,
        indirect_node_ids=(),
    )
    assert abs(direct_only.total - 0.05) < 1e-12
    assert direct_only.indirect_reward == 0.0

    no_indirect_gain = compute_blackbox_reward(
        target_node=0,
        previous_scores={0: 10.0, 1: 9.0},
        current_scores={0: 11.0, 1: 10.0},
        threshold=8.0,
        added_edges=0,
        batch_actions=5,
        indirect_node_ids={1},
    )
    assert no_indirect_gain.indirect_reward == 0.0
    assert no_indirect_gain.total < 0.0

    print(
        "attack_objective_validation_ok "
        f"total={reward.total:.12f} "
        f"indirect={reward.indirect_reward:.12f} "
        f"crossed={reward.indirect_cross_count}"
    )


if __name__ == "__main__":
    main()
