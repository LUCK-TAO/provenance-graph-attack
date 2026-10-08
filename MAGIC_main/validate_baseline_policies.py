"""Fast, detector-free validation for adaptive and mimicry baselines."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import dgl

from baseline_policies import DoubleDQNActionPolicy, LinUCBActionPolicy
from benign_mimicry_policy import BenignMimicryActionPolicy
from action_schema import ActionSchema, ActionSpec
from attack_directed_topk import DirectedTopKAttackAgent
from ppo_agent import PPOAgent


def main():
    state = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    legal_mask = torch.tensor([False, False, True])

    linucb = LinUCBActionPolicy(6, 3, alpha=1.0)
    linucb_action = linucb.select_action(state, [0, 2])
    if linucb_action not in {0, 2}:
        raise AssertionError("LinUCB selected an illegal action")
    linucb_stats = linucb.observe(state, linucb_action, 0.75)
    if linucb_stats["total_updates"] != 1:
        raise AssertionError("LinUCB did not consume its feedback")
    contextless = LinUCBActionPolicy(
        6, 3, alpha=1.0, context_mode="constant"
    )
    constant_a = contextless._state_array(state)
    constant_b = contextless._state_array(state * 9.0)
    if not (constant_a == constant_b).all():
        raise AssertionError("LinUCB constant-context ablation leaked state")
    if constant_a.tolist() != [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
        raise AssertionError("LinUCB constant-context intercept is invalid")

    double_dqn = DoubleDQNActionPolicy(
        state_dim=6,
        action_dim=3,
        hidden_dim=8,
        device="cpu",
        seed=42,
        batch_size=1,
        target_update_interval=1,
    )
    dqn_action = double_dqn.select_action(state, legal_mask)
    if dqn_action != 2:
        raise AssertionError("Double DQN selected an illegal action")
    dqn_stats = double_dqn.observe_episode([{
        "state": state,
        "action_type": dqn_action,
        "reward": 1.0,
        "action_mask": legal_mask,
    }])
    loss = dqn_stats["last_update"].get("loss")
    if dqn_stats["total_updates"] != 1 or not math.isfinite(loss):
        raise AssertionError("Double DQN optimizer validation failed")
    if dqn_stats["target_syncs"] != 1:
        raise AssertionError("Double DQN target network did not synchronize")

    mimicry_schema = ActionSchema(
        dataset="unit",
        version="unit-mimicry-v1",
        node_types={0: "PROCESS", 1: "OBJECT"},
        edge_types={0: "BENIGN_RELATION", 1: "OTHER_RELATION"},
        actions=(
            ActionSpec(0, "MIMIC", 0, "BENIGN_RELATION", ((0, 1),)),
            ActionSpec(1, "OTHER", 1, "OTHER_RELATION", ((0, 1),)),
        ),
    )
    mimicry_graph = dgl.graph(
        (torch.tensor([1, 1, 0]), torch.tensor([2, 3, 2])),
        num_nodes=4,
    )
    mimicry_graph.ndata["attr"] = torch.tensor([
        [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0],
    ])
    mimicry_graph.edata["attr"] = torch.tensor([
        [1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
    ])
    mimicry = BenignMimicryActionPolicy(
        mimicry_schema, macro_action_length=1
    )
    mimicry_fit = mimicry.fit(mimicry_graph, excluded_alert_nodes={0})
    mimicry_action = mimicry.select_action(
        mimicry_graph, target_node=0, legal_action_types=[0, 1], additions=1
    )
    if mimicry_action != 0:
        raise AssertionError(
            "Benign-Mimicry did not choose the action matching its benign profile"
        )
    if mimicry_fit["excluded_alert_nodes"] != 1:
        raise AssertionError("Benign-Mimicry alert exclusion failed")

    ppo_config = SimpleNamespace(
        STATE_DIM=6,
        ACTION_DIM=3,
        HIDDEN_DIM=8,
        DEVICE=torch.device("cpu"),
        LEARNING_RATE=3e-4,
        WARMUP_UPDATES=0,
    )
    ppo = PPOAgent(ppo_config)
    with torch.no_grad():
        for parameter in ppo.network.parameters():
            parameter.zero_()
        ppo.network.actor_action_type[-1].bias.copy_(
            torch.tensor([0.0, 1.0, 2.0])
        )
    ppo_mask = torch.tensor([True, False, True])
    argmax_actions = [
        ppo.select_action(
            state, action_mask=ppo_mask, selection_mode="argmax"
        )[0]
        for _ in range(20)
    ]
    if set(argmax_actions) != {2}:
        raise AssertionError(
            "Frozen argmax selection changed across repeated evaluations"
        )
    ppo_stats = dict(ppo.last_action_stats)
    if ppo_stats["selection_mode"] != "argmax":
        raise AssertionError("PPO selection mode was not recorded")
    if not 0.0 <= ppo_stats["normalized_entropy"] <= 1.0 + 1e-6:
        raise AssertionError("PPO normalized entropy is outside [0, 1]")
    if ppo_stats["max_action_probability"] <= 0.5:
        raise AssertionError("PPO maximum legal action probability is invalid")
    sampled_action = ppo.select_action(
        state, action_mask=ppo_mask, selection_mode="sample"
    )[0]
    if sampled_action not in {0, 2}:
        raise AssertionError("Sampled PPO action violated the legal mask")

    health_agent = object.__new__(DirectedTopKAttackAgent)
    health_agent.config = SimpleNamespace(PPO_ROLLOUT_SIZE=8)
    health_agent.ppo_selection_mode = "sample"
    health_agent.query_records = [
        {
            "kind": "attack_feedback",
            "macro_action_id": 0,
            "total_reward": 1.0,
            "crossed_threshold": True,
            "indirect_reward": 0.25,
            "policy_selection": {
                "selection_mode": "sample",
                "normalized_entropy": 0.9,
                "max_action_probability": 0.4,
                "top1_top2_probability_gap": 0.1,
                "kl_to_uniform": 0.05,
            },
        },
        {
            "kind": "attack_feedback",
            "macro_action_id": 0,
            "total_reward": -0.2,
            "crossed_threshold": False,
            "indirect_reward": 0.0,
            "policy_selection": {
                "selection_mode": "sample",
                "normalized_entropy": 0.7,
                "max_action_probability": 0.6,
                "top1_top2_probability_gap": 0.3,
                "kl_to_uniform": 0.2,
            },
        },
    ]
    health = health_agent._summarize_policy_health()
    if abs(health["mean_normalized_entropy"] - 0.8) > 1e-12:
        raise AssertionError("Policy-health entropy aggregation failed")
    if abs(health["indirect_reward_nonzero_rate"] - 0.5) > 1e-12:
        raise AssertionError("Indirect reward hit-rate aggregation failed")
    if abs(health["per_action_reward"][0]["mean_reward"] - 0.4) > 1e-12:
        raise AssertionError("Per-action reward aggregation failed")

    print(
        "baseline_policy_validation_ok "
        f"linucb_action={linucb_action} "
        f"ddqn_action={dqn_action} ddqn_loss={loss:.6f} "
        f"mimicry_action={mimicry_action} "
        f"ppo_argmax={argmax_actions[0]} "
        f"ppo_norm_entropy={ppo_stats['normalized_entropy']:.6f} "
        f"health_mean_entropy={health['mean_normalized_entropy']:.6f}"
    )


if __name__ == "__main__":
    main()
