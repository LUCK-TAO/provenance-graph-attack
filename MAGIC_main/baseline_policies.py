"""Query-fair learning baselines for the semantic macro-action attack."""

from __future__ import annotations

import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinUCBActionPolicy:
    """Disjoint LinUCB over legal semantic actions.

    Each action owns an independent linear reward model.  Selection and
    updates use only the state and reward already exposed to PPO, so LinUCB
    performs no additional detector query.
    """

    CONTEXT_MODES = {"full", "constant"}

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float = 1.0,
        context_mode: str = "full",
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.alpha = float(alpha)
        if context_mode not in self.CONTEXT_MODES:
            raise ValueError(f"invalid LinUCB context mode: {context_mode}")
        self.context_mode = context_mode
        self.a_matrices = np.repeat(
            np.eye(self.state_dim, dtype=np.float64)[None, :, :],
            self.action_dim,
            axis=0,
        )
        self.b_vectors = np.zeros(
            (self.action_dim, self.state_dim), dtype=np.float64
        )
        self.counts = np.zeros(self.action_dim, dtype=np.int64)
        self.reward_sums = np.zeros(self.action_dim, dtype=np.float64)
        self.total_updates = 0
        self.last_action_stats = {}

    def _state_array(self, state):
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        context = np.asarray(state, dtype=np.float64).reshape(-1)
        if self.context_mode == "constant":
            # A constant intercept turns disjoint LinUCB into a context-free
            # UCB-style bandit while preserving the action mask and update
            # schedule.  This is the state-encoding ablation.
            context = np.zeros_like(context)
            if context.size:
                context[0] = 1.0
        return context

    def select_action(self, state, legal_action_types):
        legal = sorted(set(int(value) for value in legal_action_types))
        if not legal:
            raise ValueError("LinUCB requires at least one legal action")
        context = self._state_array(state)
        if context.size != self.state_dim:
            raise ValueError(
                f"LinUCB state size {context.size} != {self.state_dim}"
            )
        scores = {}
        predicted_rewards = {}
        confidence_widths = {}
        for action_id in legal:
            a_inv = np.linalg.inv(self.a_matrices[action_id])
            theta = a_inv @ self.b_vectors[action_id]
            predicted = float(theta @ context)
            confidence = float(
                self.alpha
                * math.sqrt(max(context @ a_inv @ context, 0.0))
            )
            predicted_rewards[action_id] = predicted
            confidence_widths[action_id] = confidence
            scores[action_id] = predicted + confidence
        selected = max(legal, key=lambda action_id: (scores[action_id], -action_id))
        self.last_action_stats = {
            "selected_action": int(selected),
            "alpha": self.alpha,
            "context_mode": self.context_mode,
            "selection_scores": {
                int(key): round(value, 6) for key, value in scores.items()
            },
            "predicted_rewards": {
                int(key): round(value, 6)
                for key, value in predicted_rewards.items()
            },
            "confidence_widths": {
                int(key): round(value, 6)
                for key, value in confidence_widths.items()
            },
        }
        return int(selected)

    def observe(self, state, action_type: int, reward: float):
        context = self._state_array(state)
        action_id = int(action_type)
        self.a_matrices[action_id] += np.outer(context, context)
        self.b_vectors[action_id] += float(reward) * context
        self.counts[action_id] += 1
        self.reward_sums[action_id] += float(reward)
        self.total_updates += 1
        return self.snapshot()

    def snapshot(self):
        mean_rewards = np.divide(
            self.reward_sums,
            self.counts,
            out=np.zeros_like(self.reward_sums),
            where=self.counts > 0,
        )
        return {
            "alpha": self.alpha,
            "context_mode": self.context_mode,
            "total_updates": int(self.total_updates),
            "counts": self.counts.tolist(),
            "mean_rewards": mean_rewards.tolist(),
        }

    def save_checkpoint(self, path):
        torch.save({
            "policy": "linucb",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "alpha": self.alpha,
            "context_mode": self.context_mode,
            "a_matrices": torch.from_numpy(self.a_matrices),
            "b_vectors": torch.from_numpy(self.b_vectors),
            "counts": torch.from_numpy(self.counts),
            "reward_sums": torch.from_numpy(self.reward_sums),
            "total_updates": self.total_updates,
        }, path)


class NodeTypeLinUCBActionPolicy:
    """Independent LinUCB learners for each observable target node type.

    Orthrus full-mode feedback differs sharply between subjects and files.
    Keeping a separate learner per node type prevents rewards from one type
    from contaminating action estimates for another while retaining the same
    state representation and legal-action mask.
    """

    def __init__(self, state_dim, action_dim, alpha=1.0, context_mode="full"):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self._alpha = float(alpha)
        self._context_mode = str(context_mode)
        self.policies = {}
        self.last_action_stats = {}

    def _policy(self, target_type: int):
        key = int(target_type)
        if key not in self.policies:
            self.policies[key] = LinUCBActionPolicy(
                self.state_dim,
                self.action_dim,
                alpha=self._alpha,
                context_mode=self._context_mode,
            )
        return self.policies[key]

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = float(value)
        for policy in self.policies.values():
            policy.alpha = self._alpha

    @property
    def context_mode(self):
        return self._context_mode

    @context_mode.setter
    def context_mode(self, value):
        if value not in LinUCBActionPolicy.CONTEXT_MODES:
            raise ValueError(f"invalid LinUCB context mode: {value}")
        self._context_mode = str(value)
        for policy in self.policies.values():
            policy.context_mode = self._context_mode

    @property
    def total_updates(self):
        return sum(policy.total_updates for policy in self.policies.values())

    def select_action(self, state, legal_action_types, *, target_type=0):
        policy = self._policy(target_type)
        action = policy.select_action(state, legal_action_types)
        self.last_action_stats = {
            **policy.last_action_stats,
            "target_type": int(target_type),
            "type_separated": True,
        }
        return action

    def observe(self, state, action_type, reward, *, target_type=0):
        self._policy(target_type).observe(state, action_type, reward)
        return self.snapshot()

    def snapshot(self):
        counts = np.zeros(self.action_dim, dtype=np.int64)
        reward_sums = np.zeros(self.action_dim, dtype=np.float64)
        by_target_type = {}
        for target_type, policy in sorted(self.policies.items()):
            counts += policy.counts
            reward_sums += policy.reward_sums
            by_target_type[int(target_type)] = policy.snapshot()
        means = np.divide(
            reward_sums,
            counts,
            out=np.zeros_like(reward_sums),
            where=counts > 0,
        )
        return {
            "alpha": self._alpha,
            "context_mode": self._context_mode,
            "type_separated": True,
            "total_updates": int(self.total_updates),
            "counts": counts.tolist(),
            "mean_rewards": means.tolist(),
            "by_target_type": by_target_type,
        }

    def save_checkpoint(self, path):
        torch.save({
            "policy": "node_type_linucb",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "alpha": self._alpha,
            "context_mode": self._context_mode,
            "policies": {
                int(target_type): {
                    "a_matrices": torch.from_numpy(policy.a_matrices),
                    "b_vectors": torch.from_numpy(policy.b_vectors),
                    "counts": torch.from_numpy(policy.counts),
                    "reward_sums": torch.from_numpy(policy.reward_sums),
                    "total_updates": policy.total_updates,
                }
                for target_type, policy in self.policies.items()
            },
        }, path)


class _QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.layers(state)


class DoubleDQNActionPolicy:
    """Masked Double DQN using the same macro transitions as PPO."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        device,
        seed: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 16,
        replay_capacity: int = 4096,
        target_update_interval: int = 25,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 50.0,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_interval = int(target_update_interval)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = random.Random(int(seed) + 3001)
        self.online_network = _QNetwork(
            self.state_dim, int(hidden_dim), self.action_dim
        ).to(self.device)
        self.target_network = _QNetwork(
            self.state_dim, int(hidden_dim), self.action_dim
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=float(learning_rate)
        )
        self.replay = deque(maxlen=int(replay_capacity))
        self.selection_steps = 0
        self.training_stats = {
            "total_updates": 0,
            "total_loss": 0.0,
            "target_syncs": 0,
        }
        self.last_action_stats = {}
        self.last_update_stats = {}

    @property
    def epsilon(self):
        return self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-self.selection_steps / self.epsilon_decay)

    def select_action(self, state, action_mask):
        mask = action_mask.detach().to(dtype=torch.bool).reshape(-1)
        legal = torch.nonzero(mask, as_tuple=False).reshape(-1).cpu().tolist()
        if not legal:
            raise ValueError("Double DQN requires at least one legal action")
        epsilon = self.epsilon
        with torch.no_grad():
            q_values = self.online_network(
                state.to(self.device).reshape(1, -1)
            ).reshape(-1)
        explore = self.rng.random() < epsilon
        if explore:
            selected = self.rng.choice(legal)
        else:
            selected = max(
                legal, key=lambda action_id: (float(q_values[action_id]), -action_id)
            )
        self.selection_steps += 1
        self.last_action_stats = {
            "selected_action": int(selected),
            "epsilon": float(epsilon),
            "explore": bool(explore),
            "selection_scores": {
                int(action_id): round(float(q_values[action_id]), 6)
                for action_id in legal
            },
        }
        return int(selected)

    @staticmethod
    def _cpu_state(state):
        return state.detach().to(device="cpu", dtype=torch.float32).reshape(-1)

    @staticmethod
    def _cpu_mask(mask):
        return mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)

    def observe_episode(self, experiences):
        experiences = list(experiences)
        for index, item in enumerate(experiences):
            done = index == len(experiences) - 1
            if done:
                next_state = torch.zeros_like(self._cpu_state(item["state"]))
                next_mask = torch.zeros_like(
                    self._cpu_mask(item["action_mask"])
                )
            else:
                next_item = experiences[index + 1]
                next_state = self._cpu_state(next_item["state"])
                next_mask = self._cpu_mask(next_item["action_mask"])
            self.replay.append((
                self._cpu_state(item["state"]),
                int(item["action_type"]),
                float(item["reward"]),
                next_state,
                bool(done),
                next_mask,
            ))
            self._optimize_once()
        return self.snapshot()

    def _optimize_once(self):
        if len(self.replay) < self.batch_size:
            self.last_update_stats = {
                "updated": False,
                "replay_size": len(self.replay),
                "batch_size": self.batch_size,
            }
            return None
        batch = self.rng.sample(list(self.replay), self.batch_size)
        states = torch.stack([item[0] for item in batch]).to(self.device)
        actions = torch.tensor(
            [item[1] for item in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [item[2] for item in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.stack([item[3] for item in batch]).to(self.device)
        dones = torch.tensor(
            [item[4] for item in batch], dtype=torch.bool, device=self.device
        )
        next_masks = torch.stack([item[5] for item in batch]).to(self.device)

        predicted_q = self.online_network(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        with torch.no_grad():
            online_next_q = self.online_network(next_states)
            online_next_q = online_next_q.masked_fill(~next_masks, -1e9)
            next_actions = online_next_q.argmax(dim=1)
            target_next_q = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_next_q = torch.where(
                dones, torch.zeros_like(target_next_q), target_next_q
            )
            targets = rewards + self.gamma * target_next_q

        loss = F.smooth_l1_loss(predicted_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(), max_norm=1.0
        )
        self.optimizer.step()

        self.training_stats["total_updates"] += 1
        self.training_stats["total_loss"] += float(loss.item())
        synced = False
        if (
            self.training_stats["total_updates"]
            % self.target_update_interval
            == 0
        ):
            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )
            self.training_stats["target_syncs"] += 1
            synced = True
        self.last_update_stats = {
            "updated": True,
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "q_mean": float(predicted_q.mean().item()),
            "target_mean": float(targets.mean().item()),
            "replay_size": len(self.replay),
            "batch_size": self.batch_size,
            "target_synced": synced,
            "total_updates": int(self.training_stats["total_updates"]),
        }
        return float(loss.item())

    def snapshot(self):
        return {
            **self.training_stats,
            "selection_steps": int(self.selection_steps),
            "epsilon": float(self.epsilon),
            "replay_size": len(self.replay),
            "last_update": dict(self.last_update_stats),
        }

    def save_checkpoint(self, path):
        torch.save({
            "policy": "double_dqn",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_stats": dict(self.training_stats),
            "selection_steps": self.selection_steps,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "target_update_interval": self.target_update_interval,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        }, path)
