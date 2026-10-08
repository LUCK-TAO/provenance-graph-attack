"""
PPO智能体模块 - 对抗攻击的核心强化学习组件
这个模块在两种攻击模式下完全共享
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import time


# ============================================================================
# Actor-Critic网络架构
# ============================================================================

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic网络 - 宏动作架构
    - Actor: 输出语义边类型
    - Critic: 状态价值估计
    """
    
    def __init__(self, state_dim=8, hidden_dim=128, action_type_dim=9):
        super().__init__()
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 【Actor分支】动作类型选择 (9种动作)
        self.actor_action_type = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_type_dim)
        )
        
        # 【Critic分支】状态价值估计
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        """前向传播"""
        features = self.shared_layers(state)
        
        # Actor输出
        action_type_logits = self.actor_action_type(features)
        # Critic输出
        state_value = self.critic(features)
        
        return action_type_logits, state_value


# ============================================================================
# PPO Agent
# ============================================================================

class PPOAgent:
    """
    PPO智能体 - 负责策略优化和动作选择
    
    核心功能：
    1. select_action(): 基于当前策略选择动作
    2. update(): 使用PPO算法更新网络参数
    3. compute_gae(): 计算广义优势估计
    """
    
    def __init__(self, config):
        """
        Args:
            config: 配置对象，包含超参数
        """
        self.config = config
        self.state_dim = config.STATE_DIM
        self.action_type_dim = config.ACTION_DIM
        self.policy_schema_version = "macro-edge-v1"
        
        # 初始化网络
        self.network = ActorCriticNetwork(
            state_dim=self.state_dim,
            hidden_dim=config.HIDDEN_DIM,
            action_type_dim=self.action_type_dim,
        ).to(config.DEVICE)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=config.LEARNING_RATE
        )
        
        # 训练统计
        self.training_stats = {
            'total_updates': 0,
            'total_loss': 0.0
        }
        self.last_update_stats = {}
        self.last_action_stats = {}
        self.loaded_action_schema_version = None
    
    @staticmethod
    def _apply_action_mask(logits, action_mask):
        if action_mask is None:
            return logits
        action_mask = action_mask.to(device=logits.device, dtype=torch.bool)
        if action_mask.ndim == 1 and not bool(action_mask.any()):
            raise ValueError("action mask contains no legal action")
        if action_mask.ndim > 1 and bool((~action_mask.any(dim=-1)).any()):
            raise ValueError("at least one action-mask row has no legal action")
        return logits.masked_fill(~action_mask, -1e9)

    def select_action(self, state, action_mask=None, selection_mode="sample"):
        """
        选择一个语义宏动作。
        
        Args:
            state: 状态向量
            selection_mode: ``sample``用于在线探索，``argmax``用于冻结评估
            
        Returns:
            action_type: 语义边类型
            log_prob: 动作的对数概率
            value: 状态价值估计
        """
        if selection_mode not in {"sample", "argmax"}:
            raise ValueError(
                "selection_mode must be either 'sample' or 'argmax'"
            )

        # 行为选择不需要保留计算图；训练阶段会用缓存状态重新前向计算。
        with torch.no_grad():
            action_type_logits, state_value = self.network(state)
            warmup = self.training_stats['total_updates'] < getattr(
                self.config, 'WARMUP_UPDATES', 0)
            if warmup:
                action_type_logits = torch.zeros_like(action_type_logits)
            action_type_logits = self._apply_action_mask(
                action_type_logits, action_mask
            )
            
            action_type_dist = Categorical(logits=action_type_logits)
            if selection_mode == "argmax":
                action_type = torch.argmax(action_type_logits, dim=-1)
            else:
                action_type = action_type_dist.sample()
            action_type_log_prob = action_type_dist.log_prob(action_type)
            probabilities = action_type_dist.probs.reshape(-1)
            if action_mask is None:
                legal_ids = list(range(self.action_type_dim))
            else:
                legal_ids = (
                    torch.nonzero(
                        action_mask.reshape(-1).to(dtype=torch.bool),
                        as_tuple=False,
                    )
                    .reshape(-1)
                    .cpu()
                    .tolist()
                )
            entropy = float(action_type_dist.entropy().item())
            legal_probability_values = [
                float(probabilities[action_id].item())
                for action_id in legal_ids
            ]
            ordered_probabilities = sorted(
                legal_probability_values, reverse=True
            )
            max_probability = (
                ordered_probabilities[0]
                if ordered_probabilities else 0.0
            )
            probability_gap = (
                ordered_probabilities[0] - ordered_probabilities[1]
                if len(ordered_probabilities) > 1 else 1.0
            )
            legal_action_count = len(legal_ids)
            if legal_action_count > 1:
                uniform_entropy = math.log(legal_action_count)
                normalized_entropy = entropy / uniform_entropy
                kl_to_uniform = max(uniform_entropy - entropy, 0.0)
            else:
                normalized_entropy = 0.0
                kl_to_uniform = 0.0
            self.last_action_stats = {
                'warmup': bool(warmup),
                'selection_mode': selection_mode,
                'entropy': entropy,
                'normalized_entropy': float(normalized_entropy),
                'max_action_probability': float(max_probability),
                'top1_top2_probability_gap': float(probability_gap),
                'kl_to_uniform': float(kl_to_uniform),
                'legal_action_count': legal_action_count,
                'selected_action': int(action_type.item()),
                'legal_probabilities': {
                    int(action_id): round(
                        float(probabilities[action_id].item()), 6
                    )
                    for action_id in legal_ids
                },
            }
        
        return action_type.item(), action_type_log_prob, state_value.squeeze()
    
    def compute_gae(
        self, rewards, values, next_value=0.0, dones=None,
        gamma=0.99, lambda_=0.95,
    ):
        """
        计算广义优势估计 (Generalized Advantage Estimation)
        
        Args:
            rewards: 奖励序列
            values: 状态价值序列
            next_value: 下一个状态的价值
            gamma: 折扣因子
            lambda_: GAE参数
            
        Returns:
            advantages: 优势函数估计
            returns: 回报序列
        """
        values = list(values) + [next_value]
        dones = list(dones) if dones is not None else [False] * len(rewards)
        if len(dones) != len(rewards):
            raise ValueError("dones and rewards must have the same length")
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            nonterminal = 0.0 if dones[t] else 1.0
            delta = (
                rewards[t]
                + gamma * values[t + 1] * nonterminal
                - values[t]
            )
            gae = delta + gamma * lambda_ * nonterminal * gae
            advantages.insert(0, gae)
        
        # 计算回报
        returns = []
        for t in range(len(rewards)):
            returns.append(advantages[t] + values[t])
        
        return advantages, returns
    
    def update(
        self, states, actions, old_log_probs, returns, advantages,
        action_masks=None,
    ):
        """
        使用PPO算法更新网络参数
        
        Args:
            states: 状态批次
            actions: 宏动作类型列表
            old_log_probs: 旧策略的对数概率
            returns: 回报批次
            advantages: 优势批次
            
        Returns:
            loss: 总损失
        """
        # 转换为tensor
        states = torch.stack(states)
        old_log_probs = torch.stack(old_log_probs)
        returns = torch.tensor(returns, device=self.config.DEVICE)
        advantages = torch.tensor(advantages, device=self.config.DEVICE)
        
        # 标准化优势
        if advantages.numel() > 1:
            advantage_std = advantages.std(unbiased=False)
            if float(advantage_std) > 1e-8:
                advantages = (
                    advantages - advantages.mean()
                ) / advantage_std
        
        action_types = torch.tensor(actions, device=self.config.DEVICE)
        if action_masks is not None:
            action_masks = torch.stack(action_masks).to(self.config.DEVICE)
        
        # 多次epoch更新
        total_loss = 0.0
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_entropies = []
        epoch_ratio_means = []
        epoch_clip_fractions = []
        epoch_grad_norms = []
        epoch_approx_kls = []
        for _ in range(4):  # PPO通常进行4个epoch
            # 前向传播
            action_type_logits, values = self.network(states)
            action_type_logits = self._apply_action_mask(
                action_type_logits, action_masks
            )
            
            # 计算新策略的对数概率
            action_type_dist = Categorical(logits=action_type_logits)
            new_log_probs = action_type_dist.log_prob(action_types)
            
            # PPO裁剪
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.CLIP_EPSILON, 1 + self.config.CLIP_EPSILON) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic损失
            # Keep the batch dimension for single-experience updates.  A plain
            # squeeze() turns a [1, 1] tensor into a scalar and makes MSE rely
            # on broadcasting against returns shaped [1].
            critic_loss = F.mse_loss(values.reshape(-1), returns.reshape(-1))
            
            # 熵正则化（鼓励探索）
            action_type_entropy = action_type_dist.entropy().mean()
            entropy_loss = -self.config.ENTROPY_COEF * action_type_entropy
            
            # 总损失
            loss = actor_loss + 0.5 * critic_loss + entropy_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), max_norm=0.5
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            epoch_actor_losses.append(actor_loss.item())
            epoch_critic_losses.append(critic_loss.item())
            epoch_entropies.append(action_type_entropy.item())
            epoch_ratio_means.append(ratios.mean().item())
            epoch_clip_fractions.append(
                (torch.abs(ratios - 1.0) > self.config.CLIP_EPSILON)
                .float().mean().item()
            )
            epoch_grad_norms.append(float(grad_norm))
            epoch_approx_kls.append(
                (old_log_probs - new_log_probs).mean().item()
            )
        
        # 更新统计
        self.training_stats['total_updates'] += 1
        self.training_stats['total_loss'] += total_loss
        epochs = max(len(epoch_actor_losses), 1)
        self.last_update_stats = {
            'total_loss': total_loss,
            'avg_loss': total_loss / epochs,
            'actor_loss': sum(epoch_actor_losses) / epochs,
            'critic_loss': sum(epoch_critic_losses) / epochs,
            'entropy': sum(epoch_entropies) / epochs,
            'ratio_mean': sum(epoch_ratio_means) / epochs,
            'clip_fraction': sum(epoch_clip_fractions) / epochs,
            'grad_norm': sum(epoch_grad_norms) / epochs,
            'approx_kl': sum(epoch_approx_kls) / epochs,
            'advantage_mean': float(advantages.mean().item()),
            'advantage_std': float(
                advantages.std(unbiased=False).item()
            ),
            'return_mean': float(returns.mean().item()),
            'epochs': epochs,
            'batch_size': len(states),
        }
        
        return total_loss
    
    def save_checkpoint(self, path):
        """保存模型检查点"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'state_dim': self.state_dim,
            'action_dim': self.action_type_dim,
            'policy_schema_version': self.policy_schema_version,
            'action_schema_version': getattr(
                self.config, 'ACTION_SCHEMA_VERSION', None
            ),
        }, path)
        print(f"[PPO Agent] 模型已保存到: {path}")
    
    def load_checkpoint(self, path):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.config.DEVICE)
        checkpoint_policy_schema = checkpoint.get('policy_schema_version')
        if checkpoint_policy_schema != self.policy_schema_version:
            raise ValueError(
                "checkpoint policy structure is incompatible: "
                f"checkpoint={checkpoint_policy_schema!r}, "
                f"current={self.policy_schema_version!r}. "
                "Train a new macro-action checkpoint."
            )
        self.loaded_action_schema_version = checkpoint.get(
            'action_schema_version'
        )
        network_state = checkpoint.get('network_state_dict') or {}
        inferred_state_dim = (
            int(network_state['shared_layers.0.weight'].shape[1])
            if 'shared_layers.0.weight' in network_state
            else self.state_dim
        )
        inferred_action_dim = (
            int(network_state['actor_action_type.2.weight'].shape[0])
            if 'actor_action_type.2.weight' in network_state
            else self.action_type_dim
        )
        checkpoint_state_dim = int(
            checkpoint.get('state_dim', inferred_state_dim)
        )
        checkpoint_action_dim = int(
            checkpoint.get('action_dim', inferred_action_dim)
        )
        if (
            checkpoint_state_dim != self.state_dim
            or checkpoint_action_dim != self.action_type_dim
        ):
            raise ValueError(
                "checkpoint architecture is incompatible with the current "
                f"semantic action/state design: checkpoint="
                f"({checkpoint_state_dim}, {checkpoint_action_dim}), "
                f"current=({self.state_dim}, {self.action_type_dim}). "
                "Train a new checkpoint."
            )
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loaded_stats = checkpoint.get('training_stats') or {}
        self.training_stats = {
            'total_updates': int(loaded_stats.get('total_updates', 0)),
            'total_loss': float(loaded_stats.get('total_loss', 0.0)),
        }
        self.last_update_stats = {}
        self.last_action_stats = {}
        print(f"[PPO Agent] 模型已从 {path} 加载")
