"""
基于有向图语义的阈值邻近 Top-K 攻击策略
- 动作空间区分入边/出边操作(符合APT攻击语义)
- 只攻击检测器当前阈值以上的告警
- 按 score-threshold 从小到大，优先攻击最接近逃逸边界的节点
- 每5条同语义边构成一个宏动作，每个宏动作只查询和反馈一次
- 攻击选点和奖励只使用检测器可观察量，不读取 metadata 真值标签
"""

import sys
import time
import random
import math
import logging
import copy
from collections import deque
from pathlib import Path
import torch
import dgl
import numpy as np

# Support both ``python ATTACK.py`` package imports and direct execution via
# ``python MAGIC_main/attack_directed_topk.py`` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger('attack.directed_topk')

# 配置日志：INFO输出轮次/ASR/预算/最终指标，DEBUG输出embedding/KNN/逐节点细节（默认关闭）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# 导入共享模块
from ppo_agent import PPOAgent
from state_encoder import StateEncoder
from attack_state import AttackState
from candidate_selector import CandidateSelector
from utils.loaddata import load_entity_level_dataset
from DL_detector import DLDetector
from attack_targeting import rank_alerts, get_threshold_margin_rank
from attack_objective import compute_blackbox_reward
from paper_metrics import compute_primary_metrics
from greedy_policy import GreedyActionPolicy
from benign_mimicry_policy import BenignMimicryActionPolicy
from baseline_policies import (
    LinUCBActionPolicy,
    NodeTypeLinUCBActionPolicy,
    DoubleDQNActionPolicy,
)
from action_schema import (
    CADETS_ACTION_SCHEMA,
    build_observed_action_schema,
    get_action_schema,
)


# ============================================================================
# 配置类
# ===========================================
# =================================

class DirectedAttackConfig:
    """有向图攻击配置参数"""
    DATASET_NAME = 'cadets'  # 数据集名称: 'cadets', 'theia', 'trace'

    # ==================== 攻击流程配置 ====================
    MAX_ROUNDS = 5              # 总轮数
    TOP_K = 40                 # Top-K恶意节点(改为100)
    MAX_STEPS_PER_NODE = 10     # 每个节点最大步数(改为10)
    MACRO_ACTION_LENGTH = 5      # 每个宏动作连续添加5条同类型边
    
    # ==================== 停止条件 ====================
    # MAX_ROUNDS is the only default global stopping budget. Other budgets are
    # opt-in so that ``--max-rounds 5`` is not silently cut short.
    MAX_QUERIES = None
    MAX_TOTAL_ADDED_EDGES = None
    MAX_RUNTIME_SECONDS = None
    NO_IMPROVEMENT_PATIENCE = None
    MIN_SCORE_IMPROVEMENT = 1e-4
    
    # ==================== PPO训练配置 ====================
    LEARNING_RATE = 3e-4
    GAMMA = 0.99                # 折扣因子
    LAMBDA = 0.95               # GAE参数
    CLIP_EPSILON = 0.2          # PPO裁剪系数
    ENTROPY_COEF = 0.01         # 熵正则化系数
    WARMUP_UPDATES = 0           # 立即使用并更新 PPO 策略
    PPO_ROLLOUT_SIZE = 8         # 至少积累8个宏动作后更新，降低反馈稀疏性

    # ==================== 对比策略配置 ====================
    LINUCB_ALPHA = 1.0
    DDQN_BATCH_SIZE = 16
    DDQN_REPLAY_CAPACITY = 4096
    DDQN_TARGET_UPDATE_INTERVAL = 25
    DDQN_EPSILON_DECAY = 50.0
    
    # ==================== 奖励配置 ====================
    # 所有奖励项只依赖检测器分数、固定阈值和图修改量。
    SCORE_DROP_WEIGHT = 1.0
    THRESHOLD_CROSS_WEIGHT = 0.5
    INDIRECT_WEIGHT = 0.2
    PERTURBATION_WEIGHT = 0.05

    # 图修改率只报告，不作为 TP 减少成功与否的门槛。
    
    # ==================== 状态空间配置 ====================
    STATE_DIM = StateEncoder.STATE_DIM
    ACTION_DIM = CADETS_ACTION_SCHEMA.action_dim
    ACTION_SCHEMA_VERSION = CADETS_ACTION_SCHEMA.version
    
    # ==================== 网络架构 ====================
    HIDDEN_DIM = 128            # 隐藏层维度
    NUM_LAYERS = 2              # 网络层数
    
    # ==================== 设备 ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 动作空间定义(基于APT攻击语义的有向操作)
# ============================================================================

class DirectedActionTypes:
    """Compatibility facade over the active dataset action schema."""

    SCHEMA = CADETS_ACTION_SCHEMA
    ACTION_TO_EDGE_TYPE = {
        spec.action_id: spec.edge_type for spec in SCHEMA.actions
    }
    ACTION_NAMES = {
        spec.action_id: spec.name for spec in SCHEMA.actions
    }
    
    @classmethod
    def get_edge_type(cls, action_type):
        if int(action_type) not in cls.ACTION_TO_EDGE_TYPE:
            raise ValueError(f"unknown action ID {action_type}")
        return cls.ACTION_TO_EDGE_TYPE[int(action_type)]

    @classmethod
    def get_action_name(cls, action_type):
        return cls.ACTION_NAMES.get(action_type, f'UNKNOWN_ACTION_{action_type}')


# ============================================================================
# 辅助类
# ============================================================================

class ModificationLog:
    """记录语义合法的有向加边操作，避免每步都 clone 图。"""
    
    def __init__(self, original_graph):
        self.original_graph = original_graph
        self.modifications = []
    
    def add_edge(self, src, dst, action_type, edge_type=None):
        """
        添加有向边
        
        Args:
            src: 源节点
            dst: 目标节点
            action_type: 动作类型
        """
        if edge_type is None:
            edge_type = DirectedActionTypes.get_edge_type(action_type)
        self.modifications.append({
            'type': 'add_edge',
            'src': src,
            'dst': dst,
            'edge_type': edge_type
        })

    def pending_edge_pairs(self):
        return {(m['src'], m['dst']) for m in self.modifications
                if m['type'] == 'add_edge'}
    
    def apply_modifications(self):
        """应用所有修改生成新图"""
        modified_graph = self.original_graph.clone()
        
        edge_src_list = []
        edge_dst_list = []
        edge_type_list = []
        
        for mod in self.modifications:
            edge_src_list.append(mod['src'])
            edge_dst_list.append(mod['dst'])
            edge_type_list.append(mod['edge_type'])
        
        # 添加边
        if len(edge_src_list) > 0:
            num_edge_types = modified_graph.edata['attr'].shape[1]
            edge_feats_list = []
            
            for edge_type in edge_type_list:
                if not 0 <= int(edge_type) < num_edge_types:
                    raise ValueError(
                        f"edge type {edge_type} is outside graph feature "
                        f"dimension {num_edge_types}"
                    )
                edge_feat = torch.zeros(num_edge_types)
                edge_feat[int(edge_type)] = 1.0
                edge_feats_list.append(edge_feat)
            
            edge_feats_tensor = torch.stack(edge_feats_list)
            src_tensor = torch.tensor(edge_src_list)
            dst_tensor = torch.tensor(edge_dst_list)
            
            modified_graph = dgl.add_edges(modified_graph, src_tensor, dst_tensor,
                                          data={'attr': edge_feats_tensor})
        
        return modified_graph
    

class ScoreTracker:
    """维护Top-K节点分数表格，跟踪每轮攻击中的分数变化"""
    
    def __init__(self, top_k=None):
        if top_k is None:
            from utils import get_config
            top_k = get_config().TOP_K
        self.top_k = top_k
        self.node_ids = []
        self.initial_scores = {}
        self.latest_scores = {}
    
    def initialize(self, query_result):
        """使用检测器返回的Top-K节点初始化分数表格"""
        self.node_ids = query_result['top_k_nodes'][:self.top_k]
        all_scores = query_result['all_scores']
        self.initial_scores = {node: all_scores.get(node, 0.0) for node in self.node_ids}
        self.latest_scores = dict(self.initial_scores)
    
    def get_initial_score(self, node_id):
        return self.initial_scores.get(node_id, 0.0)
    
    def update_latest_score(self, node_id, new_score):
        if node_id in self.latest_scores:
            self.latest_scores[node_id] = new_score


# ============================================================================
# 主攻击智能体
# ============================================================================

class DirectedTopKAttackAgent:
    """
    基于有向图语义的Top-K攻击智能体
    
    核心特点：
    1. 动作空间区分入边/出边(符合APT语义)
    2. 只根据检测器告警、分数和固定阈值选择目标
    3. 使用查询、时间和图修改预算等可观察停止条件
    4. PPO只选择语义边类型；一个宏动作连续执行5条同类型边
    5. 每个宏动作只产生一次检测器查询、奖励和PPO transition
    6. 在线奖励不读取 metadata 真值；metadata 仅用于最终 TP 减少评估
    """
    
    def __init__(self, dataset_name='cadets', top_k=None, detector=None,
                 action_policy='ppo_online', target_policy='threshold_margin',
                 target_margin_fraction=0.5, seed=42,
                 candidate_policy='semantic_mixed', experiment_variant='unspecified',
                 ppo_selection_mode=None, orthrus_full_aware=False):
        """
        Args:
            dataset_name: 数据集名称
            top_k: Top-K恶意节点数量（如果为None，则从配置读取）
        """
        self.dataset_name = dataset_name
        if action_policy not in {
            'ppo_online', 'ppo_frozen', 'random', 'greedy',
            'linucb', 'double_dqn', 'benign_mimicry', 'no_attack'
        }:
            raise ValueError("invalid action_policy")
        if target_policy not in {
            'threshold_margin', 'highest_score', 'random_alert',
            'hybrid_margin_random', 'initial_tp_cycle', 'initial_tp_margin',
            'initial_tp_random'
        }:
            raise ValueError("invalid target_policy")
        if not 0.0 <= float(target_margin_fraction) <= 1.0:
            raise ValueError("target_margin_fraction must be within [0, 1]")
        self.action_policy = action_policy
        if ppo_selection_mode in {None, 'auto'}:
            if action_policy == 'ppo_online':
                ppo_selection_mode = 'sample'
            elif action_policy == 'ppo_frozen':
                ppo_selection_mode = 'argmax'
            else:
                ppo_selection_mode = 'not_applicable'
        if action_policy == 'ppo_online' and ppo_selection_mode != 'sample':
            raise ValueError(
                "ppo_online must use sample selection for on-policy exploration"
            )
        if (
            action_policy == 'ppo_frozen'
            and ppo_selection_mode not in {'sample', 'argmax'}
        ):
            raise ValueError(
                "ppo_frozen selection mode must be sample or argmax"
            )
        if (
            action_policy not in {'ppo_online', 'ppo_frozen'}
            and ppo_selection_mode != 'not_applicable'
        ):
            raise ValueError(
                "PPO selection mode is only valid for PPO policies"
            )
        self.ppo_selection_mode = ppo_selection_mode
        self.target_policy = target_policy
        self.target_margin_fraction = float(target_margin_fraction)
        self.candidate_policy = candidate_policy
        self.experiment_variant = experiment_variant
        self.orthrus_full_aware = bool(orthrus_full_aware)
        self.orthrus_candidate_probe_size = 20
        self.orthrus_full_aware_stats = {
            'candidate_probe_queries': 0,
            'rolled_back_macros': 0,
            'score_increase_rollbacks': 0,
            'protected_target_rollbacks': 0,
            'accepted_macros': 0,
        }
        self._protected_initial_tp_nodes = set()
        self.reward_mode = 'full'
        self.seed = int(seed)
        self.action_rng = random.Random(self.seed + 1001)
        self.target_rng = random.Random(self.seed + 2001)
        
        # 【关键修复】从配置读取top_k，如果没有传入参数
        if top_k is None:
            top_k = DirectedAttackConfig().TOP_K
        
        self.top_k = top_k
        self.config = DirectedAttackConfig()
        if (
            self.config.MAX_STEPS_PER_NODE
            % self.config.MACRO_ACTION_LENGTH
            != 0
        ):
            raise ValueError(
                "MAX_STEPS_PER_NODE must be divisible by "
                "MACRO_ACTION_LENGTH so every PPO macro action contains "
                "exactly five edge insertions"
            )
        self.action_schema = get_action_schema(
            dataset_name, allow_observed_fallback=True
        )
        self.config.ACTION_DIM = 9
        self.config.ACTION_SCHEMA_VERSION = (
            self.action_schema.version
            if self.action_schema is not None
            else f"{dataset_name}-observed-deferred"
        )
        if self.action_schema is not None:
            DirectedActionTypes.SCHEMA = self.action_schema
            DirectedActionTypes.ACTION_TO_EDGE_TYPE = {
                spec.action_id: spec.edge_type
                for spec in self.action_schema.actions
            }
            DirectedActionTypes.ACTION_NAMES = {
                spec.action_id: spec.name
                for spec in self.action_schema.actions
            }
        self.reward_config = {
            'mode': 'full',
            'score_weight': self.config.SCORE_DROP_WEIGHT,
            'crossing_weight': self.config.THRESHOLD_CROSS_WEIGHT,
            'indirect_weight': self.config.INDIRECT_WEIGHT,
            'perturbation_weight': self.config.PERTURBATION_WEIGHT,
        }
        # Flash carries a semantic sidecar in addition to the DGL graph.  Its
        # detector adapter advertises the resulting PPO state size.
        # Final detector-specific size is resolved after detector construction.
        
        # The attack algorithm only depends on the public detector protocol.  Keeping
        # construction outside this class makes the original MAGIC implementation
        # reusable with any detector adapter.
        logger.info("[Attack Agent] 正在初始化检测器...")
        self.detector = detector or DLDetector(dataset_name=dataset_name, top_k=top_k)
        self.detector_label = (
            self.detector.__class__.__name__.removesuffix("Detector") or "Detector"
        )
        if getattr(self.detector, 'force_observed_action_schema', False):
            # Some detectors use a different native node/edge vocabulary from
            # MAGIC even on the same DARPA dataset. Defer schema construction
            # until their actual attack graph is available.
            self.action_schema = None
            self.config.ACTION_SCHEMA_VERSION = (
                f"{dataset_name}-{self.detector_label.lower()}-observed-deferred"
            )
        detector_extra_dim = getattr(
            self.detector, 'attack_state_extra_dim', None
        )
        if detector_extra_dim is not None:
            self.config.STATE_DIM = (
                StateEncoder.STATE_DIM + int(detector_extra_dim)
            )
        else:
            self.config.STATE_DIM = getattr(
                self.detector, 'attack_state_dim', StateEncoder.STATE_DIM
            )
        
        # 初始化候选选择器
        self.candidate_selector = CandidateSelector(
            max_hop=2, 
            local_ratio=0.5,
            action_schema=self.action_schema,
            dataset_name=dataset_name,
            selection_mode=self.candidate_policy,
            seed=self.seed,
        )
        
        # 初始化PPO智能体
        self.ppo_agent = PPOAgent(self.config)
        self.greedy_policy = GreedyActionPolicy(self.config.ACTION_DIM)
        linucb_class = (
            NodeTypeLinUCBActionPolicy
            if self.orthrus_full_aware else LinUCBActionPolicy
        )
        self.linucb_policy = (
            linucb_class(
                self.config.STATE_DIM,
                self.config.ACTION_DIM,
                alpha=self.config.LINUCB_ALPHA,
            )
            if self.action_policy == 'linucb' else None
        )
        self.double_dqn_policy = (
            DoubleDQNActionPolicy(
                state_dim=self.config.STATE_DIM,
                action_dim=self.config.ACTION_DIM,
                hidden_dim=self.config.HIDDEN_DIM,
                device=self.config.DEVICE,
                seed=self.seed,
                learning_rate=self.config.LEARNING_RATE,
                gamma=self.config.GAMMA,
                batch_size=self.config.DDQN_BATCH_SIZE,
                replay_capacity=self.config.DDQN_REPLAY_CAPACITY,
                target_update_interval=self.config.DDQN_TARGET_UPDATE_INTERVAL,
                epsilon_decay=self.config.DDQN_EPSILON_DECAY,
            )
            if self.action_policy == 'double_dqn' else None
        )
        # THEIA/TRACE resolve their observed action schema only after the graph
        # is loaded, so the mimicry policy is constructed in schema validation.
        self.benign_mimicry_policy = None
        if self.action_policy == 'ppo_frozen':
            self.ppo_agent.network.eval()
        
        # TP基准追踪
        self.initial_tp_nodes = None
        self.initial_tp_count = 0
        self.query_count = 0
        self.attacked_nodes = set()
        # A target is attempted once it enters its attack episode, even if no
        # legal edit can ultimately be applied.  This avoids inflating ASR by
        # silently dropping hard-to-edit targets from its denominator.
        self.attempted_target_nodes = set()
        self._initial_tp_cycle_order = []
        self._initial_tp_cycle_cursor = 0
        self.observable_crossed_nodes = set()
        self.total_actions = 0
        self.stop_reason = None
        self._attack_started_at = None
        self._original_node_count = 0
        self._original_edge_count = 0
        self._initial_alert_nodes = set()
        self._consecutive_no_improvement = 0
        self.clean_repeat_checks = 0
        self.threshold_sensitivity_epsilons = (1e-6, 1e-3, 1e-1)
        self.modification_records = []
        self.query_records = []
        self.round_records = []
        self.reused_query_count = 0
        self.final_all_scores = {}
        self.final_graph = None
        self._ppo_rollout_buffer = []
        self._recent_action_ids = deque(maxlen=20)
        self._previous_feedback_context = {
            'target_type': -1,
            'action_type': -1,
            'score_drop': 0.0,
            'crossed_threshold': False,
            'reward': 0.0,
            'recent_action_histogram': [0] * self.config.ACTION_DIM,
        }
        self.checkpoint_path = (
            f'ppo_checkpoint_magic_{dataset_name}_directed_topk.pth'
        )
        
        logger.info(
            f"[Attack Agent] 初始化完成 - Directed Top-K模式 (top_k={top_k}, "
            f"action_policy={self.action_policy}, target_policy={self.target_policy}, "
            f"ppo_selection_mode={self.ppo_selection_mode}, "
            f"candidate_policy={self.candidate_policy}, "
            f"variant={self.experiment_variant}, seed={self.seed}, "
            f"orthrus_full_aware={self.orthrus_full_aware})"
        )

    def _query_detector(self, graph, node_indices=None):
        """Count attack-time black-box queries in one detector-agnostic place."""
        self.query_count += 1
        return self.detector.query(graph, node_indices=node_indices)

    def _probe_orthrus_candidates(self, candidates, inspect_nodes):
        """Count and execute one native provisional-edge screening query."""
        self.query_count += 1
        self.orthrus_full_aware_stats['candidate_probe_queries'] += 1
        return self.detector.probe_edge_candidates(
            candidates,
            inspect_nodes=inspect_nodes,
        )

    @staticmethod
    def _orthrus_max_loss_edge(query_result, target_node):
        edges = query_result.get('metadata', {}).get('max_loss_edges', {})
        return edges.get(int(target_node), edges.get(str(int(target_node))))

    def _orthrus_inspect_nodes(self, target_node):
        return sorted({
            int(target_node),
            *[int(node) for node in self._protected_initial_tp_nodes],
        })

    def _rank_orthrus_candidates_by_native_loss(
        self,
        graph,
        target_node,
        action_type,
        candidates,
        anchor_edge,
    ):
        """Probe a bounded candidate subset and rank it by native edge loss."""
        if not self.orthrus_full_aware or not candidates or not anchor_edge:
            return list(candidates), {}
        anchor_time = int(anchor_edge.get('time', -1))
        anchor_window = int(anchor_edge.get('window', -1))
        if anchor_time < 0 or anchor_window < 0:
            logger.warning(
                "[OrthrusFullAware] target=%s maximum loss lies outside the "
                "replay interval; falling back to semantic candidate order",
                target_node,
            )
            return list(candidates), {}

        anomaly_scores = getattr(self, 'cached_anomaly_scores', {})

        def fallback_key(node_id):
            score = float(anomaly_scores.get(int(node_id), math.inf))
            try:
                degree = int(
                    graph.in_degrees(int(node_id)).item()
                    + graph.out_degrees(int(node_id)).item()
                )
            except Exception:
                degree = math.inf
            return score, degree, int(node_id)

        ordered = sorted({int(node) for node in candidates}, key=fallback_key)
        probe_nodes = ordered[: max(1, int(self.orthrus_candidate_probe_size))]
        provisional = []
        probe_to_candidate = {}
        for probe_index, candidate in enumerate(probe_nodes):
            resolved = self.candidate_selector.resolve_operation(
                graph, target_node, action_type, candidate
            )
            probe_id = 1_000_000 + probe_index
            probe_to_candidate[probe_id] = int(candidate)
            provisional.append({
                'src': int(resolved['src']),
                'dst': int(resolved['dst']),
                'edge_type': int(resolved['edge_type']),
                'window': anchor_window,
                'before_time': anchor_time,
                'probe_id': probe_id,
            })
        probe_result = self._probe_orthrus_candidates(
            provisional,
            self._orthrus_inspect_nodes(target_node),
        )
        losses_by_probe = probe_result.get('candidate_losses', {})
        losses_by_candidate = {
            probe_to_candidate[int(probe_id)]: float(loss)
            for probe_id, loss in losses_by_probe.items()
            if int(probe_id) in probe_to_candidate
        }
        ranked_probe = sorted(
            probe_nodes,
            key=lambda node: (
                losses_by_candidate.get(int(node), math.inf),
                *fallback_key(node),
            ),
        )
        ranked = ranked_probe + [node for node in ordered if node not in probe_nodes]
        logger.info(
            "[OrthrusCandidateProbe] target=%s action=%s anchor=(w=%s,t=%s," 
            "loss=%.6f) probed=%d best=%s",
            target_node,
            DirectedActionTypes.get_action_name(action_type),
            anchor_window,
            anchor_time,
            float(anchor_edge.get('loss', float('nan'))),
            len(probe_nodes),
            [
                (node, round(losses_by_candidate.get(node, math.inf), 6))
                for node in ranked_probe[: self.config.MACRO_ACTION_LENGTH]
            ],
        )
        return ranked, losses_by_candidate

    def _observable_stop_reason(self, graph, scores=None):
        """Return a label-free global stop reason, or ``None``."""
        if self.config.MAX_RUNTIME_SECONDS is not None and self._attack_started_at is not None:
            if time.time() - self._attack_started_at >= self.config.MAX_RUNTIME_SECONDS:
                return 'max_runtime'
        if self.config.MAX_QUERIES is not None and self.query_count >= self.config.MAX_QUERIES:
            return 'max_queries'
        added_edges = graph.number_of_edges() - self._original_edge_count
        if (self.config.MAX_TOTAL_ADDED_EDGES is not None
                and added_edges >= self.config.MAX_TOTAL_ADDED_EDGES):
            return 'max_added_edges'
        if (self.config.NO_IMPROVEMENT_PATIENCE is not None
                and self._consecutive_no_improvement >= self.config.NO_IMPROVEMENT_PATIENCE):
            return 'no_score_improvement'
        return None

    def _initial_tp_cycle_targets(self, scores):
        """Return the next active initial direct-TP batch in cyclic order.

        This is deliberately a label-assisted evaluation protocol.  It never
        changes the detector's native scores, threshold, or two-hop audit; it
        only replaces the black-box target-ranking step.
        """
        order = self._initial_tp_cycle_order
        if not order:
            return []
        selected = []
        scanned = 0
        while scanned < len(order) and len(selected) < self.top_k:
            node_id = order[self._initial_tp_cycle_cursor]
            self._initial_tp_cycle_cursor = (
                self._initial_tp_cycle_cursor + 1
            ) % len(order)
            scanned += 1
            if scores.get(node_id, float('-inf')) >= self.detector.best_threshold:
                selected.append(node_id)
        return selected

    def _initial_tp_margin_targets(self, scores):
        """Rank active initial direct TPs by distance above the threshold."""
        threshold = self.detector.best_threshold
        return [
            node_id
            for node_id in sorted(
                (
                    node_id for node_id in self._initial_tp_cycle_order
                    if scores.get(node_id, float('-inf')) >= threshold
                ),
                key=lambda node_id: (
                    float(scores[node_id]) - threshold,
                    node_id,
                ),
            )[:self.top_k]
        ]

    def _initial_tp_random_targets(self, scores):
        """Return the next unattempted active TPs from a seeded random order.

        The fixed initial direct-TP pool is shuffled once at attack startup.
        Advancing through that order without replacement gives every seed a
        reproducible random sample while keeping the target denominator free
        from accidental duplicates across rounds.
        """
        threshold = self.detector.best_threshold
        selected = []
        order = self._initial_tp_cycle_order
        while (
            self._initial_tp_cycle_cursor < len(order)
            and len(selected) < self.top_k
        ):
            node_id = order[self._initial_tp_cycle_cursor]
            self._initial_tp_cycle_cursor += 1
            if node_id in self.attempted_target_nodes:
                continue
            if scores.get(node_id, float('-inf')) >= threshold:
                selected.append(node_id)
        return selected

    def _all_initial_tp_escaped(self, scores):
        return bool(self._initial_tp_cycle_order) and all(
            scores.get(node_id, float('-inf')) < self.detector.best_threshold
            for node_id in self._initial_tp_cycle_order
        )
    
    def _get_node_type(self, graph, node_idx):
        """获取节点类型"""
        node_feat = graph.ndata['attr'][node_idx]
        return torch.argmax(node_feat).item()

    def _state_feedback_context(self):
        context = dict(self._previous_feedback_context)
        histogram = [0] * self.config.ACTION_DIM
        for action_id in self._recent_action_ids:
            if 0 <= int(action_id) < self.config.ACTION_DIM:
                histogram[int(action_id)] += 1
        context['recent_action_histogram'] = histogram
        return context

    def _record_recent_action(self, action_type):
        self._recent_action_ids.append(int(action_type))

    def _policy_update_count(self):
        if self.action_policy == 'linucb':
            return int(self.linucb_policy.total_updates)
        if self.action_policy == 'double_dqn':
            return int(
                self.double_dqn_policy.training_stats['total_updates']
            )
        if self.action_policy == 'greedy':
            return int(sum(self.greedy_policy.counts))
        if self.action_policy in {'ppo_online', 'ppo_frozen'}:
            return int(
                self.ppo_agent.training_stats.get('total_updates', 0)
            )
        return 0

    def _summarize_policy_health(self):
        """Summarize observable policy sharpness and reward separation."""
        feedback_records = [
            record for record in self.query_records
            if record.get('kind') == 'attack_feedback'
        ]
        ppo_records = [
            record for record in feedback_records
            if record.get('policy_selection', {}).get('selection_mode')
            in {'sample', 'argmax'}
        ]

        def _finite_values(field):
            values = []
            for record in ppo_records:
                value = record.get('policy_selection', {}).get(field)
                if value is not None and math.isfinite(float(value)):
                    values.append(float(value))
            return values

        def _mean(values):
            return sum(values) / len(values) if values else None

        def _window_mean(values, first):
            if not values:
                return None
            width = min(self.config.PPO_ROLLOUT_SIZE, len(values))
            window = values[:width] if first else values[-width:]
            return _mean(window)

        entropy = _finite_values('normalized_entropy')
        max_probability = _finite_values('max_action_probability')
        probability_gap = _finite_values('top1_top2_probability_gap')
        kl_to_uniform = _finite_values('kl_to_uniform')
        indirect_nonzero = sum(
            abs(float(record.get('indirect_reward', 0.0))) > 1e-12
            for record in feedback_records
        )

        per_action = []
        action_histogram = {}
        for action_id in sorted({
            int(record['macro_action_id'])
            for record in feedback_records
            if 'macro_action_id' in record
        }):
            action_records = [
                record for record in feedback_records
                if int(record.get('macro_action_id', -1)) == action_id
            ]
            rewards = [
                float(record.get('total_reward', 0.0))
                for record in action_records
            ]
            indirect_rewards = [
                float(record.get('indirect_reward', 0.0))
                for record in action_records
            ]
            crossed = sum(
                bool(record.get('crossed_threshold', False))
                for record in action_records
            )
            action_name = DirectedActionTypes.get_action_name(action_id)
            action_histogram[action_name] = len(action_records)
            per_action.append({
                'action_id': action_id,
                'action_name': action_name,
                'count': len(action_records),
                'mean_reward': _mean(rewards),
                'direct_cross_rate': (
                    crossed / len(action_records)
                    if action_records else 0.0
                ),
                'mean_indirect_reward': _mean(indirect_rewards),
                'indirect_nonzero_rate': (
                    sum(abs(value) > 1e-12 for value in indirect_rewards)
                    / len(indirect_rewards)
                    if indirect_rewards else 0.0
                ),
            })

        return {
            'selection_mode': self.ppo_selection_mode,
            'selection_count': len(ppo_records),
            'mean_normalized_entropy': _mean(entropy),
            'initial_normalized_entropy': _window_mean(entropy, True),
            'final_normalized_entropy': _window_mean(entropy, False),
            'mean_max_action_probability': _mean(max_probability),
            'initial_max_action_probability': _window_mean(
                max_probability, True
            ),
            'final_max_action_probability': _window_mean(
                max_probability, False
            ),
            'mean_top1_top2_probability_gap': _mean(probability_gap),
            'initial_top1_top2_probability_gap': _window_mean(
                probability_gap, True
            ),
            'final_top1_top2_probability_gap': _window_mean(
                probability_gap, False
            ),
            'mean_kl_to_uniform': _mean(kl_to_uniform),
            'initial_kl_to_uniform': _window_mean(kl_to_uniform, True),
            'final_kl_to_uniform': _window_mean(kl_to_uniform, False),
            'indirect_reward_nonzero_count': indirect_nonzero,
            'indirect_reward_nonzero_rate': (
                indirect_nonzero / len(feedback_records)
                if feedback_records else 0.0
            ),
            'action_histogram': action_histogram,
            'per_action_reward': per_action,
        }

    def _update_ppo_rollout(self, force=False):
        """Update PPO from complete macro-action episodes."""
        if self.action_policy != 'ppo_online':
            return None
        if not self._ppo_rollout_buffer:
            return None
        if (
            not force
            and len(self._ppo_rollout_buffer)
            < self.config.PPO_ROLLOUT_SIZE
        ):
            return None

        rollout = self._ppo_rollout_buffer
        self._ppo_rollout_buffer = []
        rewards = [item['reward'] for item in rollout]
        values = [item['value'].item() for item in rollout]
        dones = [item['done'] for item in rollout]
        advantages, returns = self.ppo_agent.compute_gae(
            rewards,
            values,
            next_value=0.0,
            dones=dones,
            gamma=self.config.GAMMA,
            lambda_=self.config.LAMBDA,
        )
        loss = self.ppo_agent.update(
            [item['state'] for item in rollout],
            [item['action_type'] for item in rollout],
            [item['log_prob'] for item in rollout],
            returns,
            advantages,
            action_masks=[item['action_mask'] for item in rollout],
        )
        stats = getattr(self.ppo_agent, 'last_update_stats', {})
        action_histogram = {}
        for item in rollout:
            name = DirectedActionTypes.get_action_name(item['action_type'])
            action_histogram[name] = action_histogram.get(name, 0) + 1
        logger.info(
            f"[PPOMacroUpdate] update="
            f"{self.ppo_agent.training_stats.get('total_updates', 0)}, "
            f"macro_batch={len(rollout)}, episodes={sum(dones)}, "
            f"total_loss={loss:.6f}, "
            f"actor_loss={stats.get('actor_loss', float('nan')):.6f}, "
            f"critic_loss={stats.get('critic_loss', float('nan')):.6f}, "
            f"entropy={stats.get('entropy', float('nan')):.6f}, "
            f"ratio_mean={stats.get('ratio_mean', float('nan')):.6f}, "
            f"clip_fraction={stats.get('clip_fraction', float('nan')):.2%}, "
            f"approx_kl={stats.get('approx_kl', float('nan')):.6f}, "
            f"grad_norm={stats.get('grad_norm', float('nan')):.6f}, "
            f"adv_mean={stats.get('advantage_mean', float('nan')):+.6f}, "
            f"adv_std={stats.get('advantage_std', float('nan')):.6f}, "
            f"return_mean={stats.get('return_mean', float('nan')):+.6f}, "
            f"reward_mean={float(np.mean(rewards)):+.6f}, "
            f"reward_min={min(rewards):+.6f}, "
            f"reward_max={max(rewards):+.6f}, "
            f"actions={action_histogram}"
        )
        return loss

    def _update_feedback_context(
        self,
        graph,
        target_node,
        action_type,
        score_before,
        score_after,
        reward,
    ):
        self._previous_feedback_context = {
            'target_type': int(self._get_node_type(graph, target_node)),
            'action_type': int(action_type),
            'score_drop': float(score_before) - float(score_after),
            'crossed_threshold': bool(
                float(score_before) >= self.detector.best_threshold
                > float(score_after)
            ),
            'reward': float(reward),
            'recent_action_histogram': [0] * self.config.ACTION_DIM,
        }

    def _validate_action_schema(self, graph):
        if self.action_schema is None:
            detector_schema_builder = getattr(
                self.detector, "build_action_schema", None
            )
            if callable(detector_schema_builder):
                self.action_schema = detector_schema_builder(
                    graph, action_dim=self.config.ACTION_DIM
                )
            else:
                self.action_schema = build_observed_action_schema(
                    graph,
                    dataset=self.dataset_name,
                    action_dim=self.config.ACTION_DIM,
                )
            self.config.ACTION_SCHEMA_VERSION = self.action_schema.version
            self.candidate_selector.set_action_schema(self.action_schema)
            DirectedActionTypes.SCHEMA = self.action_schema
            DirectedActionTypes.ACTION_TO_EDGE_TYPE = {
                spec.action_id: spec.edge_type
                for spec in self.action_schema.actions
            }
            DirectedActionTypes.ACTION_NAMES = {
                spec.action_id: spec.name
                for spec in self.action_schema.actions
            }
        if (
            self.action_policy == 'benign_mimicry'
            and self.benign_mimicry_policy is None
        ):
            self.benign_mimicry_policy = BenignMimicryActionPolicy(
                self.action_schema,
                macro_action_length=self.config.MACRO_ACTION_LENGTH,
            )
        loaded_schema = getattr(
            self.ppo_agent, 'loaded_action_schema_version', None
        )
        if (
            loaded_schema is not None
            and loaded_schema != self.action_schema.version
        ):
            raise ValueError(
                "checkpoint action semantics do not match this dataset: "
                f"checkpoint={loaded_schema}, "
                f"current={self.action_schema.version}"
            )
        node_dim = int(graph.ndata['attr'].shape[1])
        edge_dim = int(graph.edata['attr'].shape[1])
        if node_dim > StateEncoder.NODE_TYPE_DIM:
            raise ValueError(
                f"graph has {node_dim} node types, but state encoder reserves "
                f"{StateEncoder.NODE_TYPE_DIM}; increase NODE_TYPE_DIM"
            )
        if node_dim != len(self.action_schema.node_types):
            raise ValueError(
                f"{self.action_schema.version} expects "
                f"{len(self.action_schema.node_types)} node features, "
                f"graph has {node_dim}"
            )
        if edge_dim != len(self.action_schema.edge_types):
            raise ValueError(
                f"{self.action_schema.version} expects "
                f"{len(self.action_schema.edge_types)} edge features, "
                f"graph has {edge_dim}"
            )

        node_types = torch.argmax(graph.ndata['attr'], dim=1).cpu().tolist()
        edge_types = torch.argmax(graph.edata['attr'], dim=1).cpu().tolist()
        src_nodes, dst_nodes = graph.edges()
        observed = {
            (
                int(node_types[int(src)]),
                int(edge_type),
                int(node_types[int(dst)]),
            )
            for src, dst, edge_type in zip(
                src_nodes.cpu().tolist(),
                dst_nodes.cpu().tolist(),
                edge_types,
            )
        }
        missing = []
        for spec in self.action_schema.actions:
            for src_type, dst_type in spec.valid_pairs:
                triple = (src_type, spec.edge_type, dst_type)
                if triple not in observed:
                    missing.append((spec.name, triple))
        if missing:
            raise ValueError(
                "action schema contains triples absent from this graph: "
                f"{missing}"
            )
        allowed_triples = {
            (src_type, spec.edge_type, dst_type)
            for spec in self.action_schema.actions
            for src_type, dst_type in spec.valid_pairs
        }
        covered_edges = sum(
            (
                int(node_types[int(src)]),
                int(edge_type),
                int(node_types[int(dst)]),
            )
            in allowed_triples
            for src, dst, edge_type in zip(
                src_nodes.cpu().tolist(),
                dst_nodes.cpu().tolist(),
                edge_types,
            )
        )
        coverage = covered_edges / max(graph.number_of_edges(), 1)
        logger.info(
            f"[ActionSchema] dataset={self.action_schema.dataset}, "
            f"version={self.action_schema.version}, "
            f"actions={self.action_schema.action_dim}, "
            f"state_dim={self.config.STATE_DIM}, "
            f"node_types={node_dim}, edge_types={edge_dim}, "
            f"edge_coverage={covered_edges}/{graph.number_of_edges()}"
            f"({coverage:.2%})"
        )
        for spec in self.action_schema.actions:
            logger.info(
                f"[ActionSchema] id={spec.action_id}, name={spec.name}, "
                f"edge={spec.edge_type}({spec.edge_name}), "
                f"valid_pairs={list(spec.valid_pairs)}"
            )

    def _run_clean_repeat_audit(self, graph, reference_scores):
        """Recompute the clean graph and report score/alert reproducibility."""
        reports = []
        reference_alerts = {
            node_id for node_id, score in reference_scores.items()
            if score >= self.detector.best_threshold
        }
        for repeat_index in range(self.clean_repeat_checks):
            cache_cleared = False
            embedding_cache = getattr(self.detector, '_embedding_cache', None)
            if isinstance(embedding_cache, dict):
                embedding_cache.clear()
                cache_cleared = True
            repeated = self.detector.query(graph)['all_scores']
            common = set(reference_scores) & set(repeated)
            max_abs_delta = max(
                (abs(reference_scores[node] - repeated[node]) for node in common),
                default=0.0,
            )
            repeated_alerts = {
                node_id for node_id, score in repeated.items()
                if score >= self.detector.best_threshold
            }
            report = {
                'kind': 'clean_repeat',
                'repeat': repeat_index + 1,
                'cache_cleared': cache_cleared,
                'reference_score_count': len(reference_scores),
                'repeat_score_count': len(repeated),
                'missing_from_repeat': len(set(reference_scores) - set(repeated)),
                'new_in_repeat': len(set(repeated) - set(reference_scores)),
                'max_abs_score_delta': float(max_abs_delta),
                'alert_symmetric_difference': len(reference_alerts ^ repeated_alerts),
            }
            reports.append(report)
            self.query_records.append(report)
            logger.info(
                f"[CleanRepeat] repeat={repeat_index + 1}/{self.clean_repeat_checks}, "
                f"cache_cleared={cache_cleared}, max_abs_score_delta={max_abs_delta:.12g}, "
                f"alert_symmetric_difference={report['alert_symmetric_difference']}, "
                f"missing_scores={report['missing_from_repeat']}, "
                f"new_scores={report['new_in_repeat']}"
            )
        return reports
    
    def attack(self, test_graph):
        """
        执行有向图Top-K攻击
        
        Args:
            test_graph: 测试图
            
        Returns:
            dict: 攻击结果
        """
        logger.info("开始Directed Top-K攻击")
        logger.info("目标模式: directed_top_k")
        logger.info(
            f"实验策略: action_policy={self.action_policy}, "
            f"target_policy={self.target_policy}, seed={self.seed}"
        )
        checkpoint_updates_start = self.ppo_agent.training_stats.get('total_updates', 0)
        policy_updates_start = self._policy_update_count()
        online_adaptation = self.action_policy in {
            'ppo_online', 'greedy', 'linucb', 'double_dqn'
        }
        logger.info(
            f"[Policy] mode={self.action_policy}, "
            f"selection_mode={self.ppo_selection_mode}, "
            f"checkpoint_updates={checkpoint_updates_start}, "
            f"policy_updates={policy_updates_start}, "
            f"warmup_updates={self.config.WARMUP_UPDATES}, "
            f"online_adaptation={online_adaptation}"
        )
        logger.info(
            f"[MacroPolicy] semantic_actions={self.config.ACTION_DIM}, "
            f"edges_per_macro={self.config.MACRO_ACTION_LENGTH}, "
            f"macros_per_target="
            f"{self.config.MAX_STEPS_PER_NODE // self.config.MACRO_ACTION_LENGTH}, "
            f"ppo_rollout_size={self.config.PPO_ROLLOUT_SIZE}, "
            "endpoint_policy=shared_deterministic_legal"
        )
        logger.info(
            f"[AblationConfig] variant={self.experiment_variant}, "
            f"candidate_policy={self.candidate_policy}, "
            f"reward_mode={self.reward_mode}, "
            f"weights=(score={self.config.SCORE_DROP_WEIGHT}, "
            f"crossing={self.config.THRESHOLD_CROSS_WEIGHT}, "
            f"indirect={self.config.INDIRECT_WEIGHT}, "
            f"perturbation={self.config.PERTURBATION_WEIGHT})"
        )
        
        # 【新增】在攻击开始前计算初始性能指标
        logger.info("【攻击前】检测器初始性能评估")
        # 记录原始图
        original_graph = test_graph.clone()
        self._attack_started_at = time.time()
        self._original_node_count = original_graph.number_of_nodes()
        self._original_edge_count = original_graph.number_of_edges()
        self.query_count = 0
        self.attacked_nodes = set()
        self.attempted_target_nodes = set()
        self._initial_tp_cycle_order = []
        self._initial_tp_cycle_cursor = 0
        self.observable_crossed_nodes = set()
        self.total_actions = 0
        self.stop_reason = None
        self._consecutive_no_improvement = 0
        self.modification_records = []
        self.query_records = []
        self.round_records = []
        self.reused_query_count = 0
        self.final_all_scores = {}
        self.final_graph = None
        self._ppo_rollout_buffer = []
        self._recent_action_ids.clear()
        self._previous_feedback_context = {
            'target_type': -1,
            'action_type': -1,
            'score_drop': 0.0,
            'crossed_threshold': False,
            'reward': 0.0,
            'recent_action_histogram': [0] * self.config.ACTION_DIM,
        }
        self._protected_initial_tp_nodes = set()
        for key in self.orthrus_full_aware_stats:
            self.orthrus_full_aware_stats[key] = 0
        if hasattr(self, '_graph_modified'):
            delattr(self, '_graph_modified')

        if self.orthrus_full_aware:
            enable_full_aware = getattr(
                self.detector, 'enable_full_aware', None
            )
            if not callable(enable_full_aware):
                raise ValueError(
                    "--orthrus-full-aware requires the Orthrus detector adapter"
                )
            state = getattr(self.detector, 'state', None)
            if state is None or getattr(state, 'alert_mode', None) != 'full':
                raise ValueError(
                    "--orthrus-full-aware requires an Orthrus full-mode attack state"
                )
            enable_full_aware(True)
            logger.info(
                "[OrthrusFullAware] enabled: max-loss anchoring, native edge "
                "loss probes, rollback, protected targets, node-type LinUCB"
            )
        
        # 建立类型索引
        self.candidate_selector.build_type_index(test_graph)
        self._validate_action_schema(test_graph)
        
        # 【关键】初始化TP基准(只在攻击开始前做一次)
        # initialize_tp_baseline 内部已调用 query()，返回的 all_scores 可直接复用
        tp_baseline = self.detector.initialize_tp_baseline(test_graph)
        # A cached TRACE clean response avoids physical recomputation but still
        # consumes one logical black-box query in every strategy's budget.
        self.query_count = 1
        if getattr(self.detector, 'last_baseline_cache_hit', False):
            logger.info(
                "[QueryCacheHit] purpose=initial_baseline, "
                "logical_queries=1, physical_queries=0"
            )
        # 阈值和基准指标由目标检测器适配器按其原生评估逻辑提供。
        self.initial_metrics = self.detector.initial_metrics
        self.initial_tp_nodes = tp_baseline['initial_tp_nodes']
        initial_all_scores = tp_baseline['all_scores']  # 复用，避免重复 query
        self.initial_all_scores = dict(initial_all_scores)
        threshold = tp_baseline['threshold']
        self._initial_alert_nodes = set(self.detector.initial_alert_nodes)
        self.native_tp_count = self.detector.true_tp_count
        self.initial_tp_count = self.native_tp_count
        if self.target_policy in {
            'initial_tp_cycle', 'initial_tp_margin', 'initial_tp_random'
        }:
            self._initial_tp_cycle_order = sorted(
                int(node_id) for node_id in self.detector.initial_true_tp_nodes
            )
            if self.target_policy == 'initial_tp_random':
                self.target_rng.shuffle(self._initial_tp_cycle_order)
            logger.info(
                "[InitialTPPool] label-assisted target pool fixed before attack: "
                f"direct_initial_tp={len(self._initial_tp_cycle_order)}, "
                f"ranking={self.target_policy}, "
                f"batch_size={self.top_k}, rounds={self.config.MAX_ROUNDS}"
            )
        
        logger.info(
            f"{self.detector_label}原生基准已初始化: "
            f"初始告警={len(self._initial_alert_nodes)}, "
            f"metadata TP={self.native_tp_count}, FP={self.detector.false_positive_count}, "
            f"阈值={threshold:.4f} (来源={getattr(self.detector, 'threshold_source', 'detector_native')})"
        )

        if self.action_policy == 'benign_mimicry':
            if self.benign_mimicry_policy is None:
                raise RuntimeError(
                    "Benign-Mimicry requires a resolved action schema"
                )
            mimicry_fit = self.benign_mimicry_policy.fit(
                test_graph,
                excluded_alert_nodes=self._initial_alert_nodes,
            )
            logger.info(
                "[BenignMimicryFit] reference=detector_predicted_benign, "
                f"reference_nodes={mimicry_fit['predicted_benign_nodes']}, "
                f"excluded_alerts={mimicry_fit['excluded_alert_nodes']}, "
                f"nodes_by_type={mimicry_fit['benign_nodes_by_type']}"
            )

        clean_repeat_reports = self._run_clean_repeat_audit(
            test_graph, self.initial_all_scores
        )
        
        initial_query = {
            'all_scores': initial_all_scores,
            'top_k_nodes': sorted(
                initial_all_scores.keys(), key=lambda k: initial_all_scores[k], reverse=True
            ),
            'native_alerts': sorted(self._initial_alert_nodes),
            'threshold': threshold,
        }
        
        # 多轮攻击
        current_graph = test_graph.clone()
        latest_query_result = initial_query
        
        if self.action_policy == 'no_attack':
            logger.info("[NoAttackBaseline] 跳过所有图修改和PPO更新，仅重复最终评估")
            self.stop_reason = 'no_attack_baseline'
        round_indices = (
            range(0) if self.action_policy == 'no_attack'
            else range(self.config.MAX_ROUNDS)
        )
        for round_idx in round_indices:
            logger.debug(f"\n{'=' * 40} 第{round_idx + 1}轮攻击 {'=' * 40}")

            self.stop_reason = self._observable_stop_reason(
                current_graph, latest_query_result['all_scores']
            )
            if self.stop_reason:
                logger.info(f"达到可观察停止条件: {self.stop_reason}")
                break

            if (
                self.target_policy in {
                    'initial_tp_cycle', 'initial_tp_margin', 'initial_tp_random'
                }
                and self._all_initial_tp_escaped(latest_query_result['all_scores'])
            ):
                self.stop_reason = 'all_initial_tp_escaped'
                logger.info("[InitialTPCycle] 所有初始直接TP均已逃逸，提前终止")
                break
            
            # 上一轮结束后图没有发生未经查询的修改，因此
            # latest_query_result始终与current_graph严格对应。轮首直接复用
            # 最近观测，避免对同一图重复执行昂贵的检测器查询。
            query_result = latest_query_result
            self.reused_query_count += 1
            self.query_records.append({
                'kind': 'query_reuse',
                'purpose': 'round_start',
                'round': round_idx + 1,
                'saved_detector_queries': 1,
            })
            logger.info(
                f"[QueryReuse] purpose=round_start, round={round_idx + 1}, "
                f"graph_nodes={current_graph.number_of_nodes()}, "
                f"graph_edges={current_graph.number_of_edges()}, "
                f"cumulative_saved={self.reused_query_count}"
            )
            latest_query_result = query_result
            
            # 严格黑盒目标选择：只使用检测器分数和固定阈值，不读取真值标签。
            # 距离阈值最近的告警通常需要更小的分数下降即可逃逸。
            if self.target_policy == 'initial_tp_cycle':
                ranked_alerts = self._initial_tp_cycle_targets(
                    query_result['all_scores']
                )
            elif self.target_policy == 'initial_tp_margin':
                ranked_alerts = self._initial_tp_margin_targets(
                    query_result['all_scores']
                )
            elif self.target_policy == 'initial_tp_random':
                ranked_alerts = self._initial_tp_random_targets(
                    query_result['all_scores']
                )
            else:
                ranked_alerts = rank_alerts(
                    query_result['all_scores'],
                    self.detector.best_threshold,
                    strategy=self.target_policy,
                    rng=self.target_rng,
                    limit=self.top_k,
                    margin_fraction=self.target_margin_fraction,
                )
            
            # 如果未逃逸节点不足top_k个，扩大搜索范围（而不是回填已逃逸节点）
            if not ranked_alerts:
                logger.info("No detector alerts remain; stopping by observable black-box condition")
                break
            round_target_nodes = ranked_alerts[:self.top_k]

            target_preview = [
                {
                    'node': node_id,
                    'score': query_result['all_scores'][node_id],
                    'margin': query_result['all_scores'][node_id] - self.detector.best_threshold,
                }
                for node_id in round_target_nodes[:3]
            ]
            logger.info(
                f"[Attack Agent] 第{round_idx + 1}轮 - {self.target_policy}目标: "
                f"{len(round_target_nodes)}个, 前3个={target_preview}"
            )
            self.round_records.append({
                'round': round_idx + 1,
                'target_policy': self.target_policy,
                'target_margin_fraction': self.target_margin_fraction,
                'targets': [
                    {
                        'node': int(node_id),
                        'score': float(query_result['all_scores'][node_id]),
                        'margin': float(
                            query_result['all_scores'][node_id]
                            - self.detector.best_threshold
                        ),
                    }
                    for node_id in round_target_nodes
                ],
            })
            
            # 【问题6修复】只在第一轮初始化ScoreTracker，后续轮次只更新分数
            if round_idx == 0:
                score_tracker = ScoreTracker(top_k=self.top_k)
                tracking_query = dict(query_result)
                tracking_query['top_k_nodes'] = round_target_nodes
                score_tracker.initialize(tracking_query)
                logger.debug(f"[ScoreTracker] 已初始化 {len(score_tracker.node_ids)} 个节点")
            else:
                # 【问题3修复】后续轮次：更新所有节点的分数到最新值，并为新节点设置initial_score
                for node_id in round_target_nodes:
                    if node_id in query_result['all_scores']:
                        current_score = query_result['all_scores'][node_id]
                        # 如果节点不在score_tracker中，添加它并设置initial_score
                        if node_id not in score_tracker.node_ids:
                            score_tracker.node_ids.append(node_id)
                            score_tracker.initial_scores[node_id] = current_score
                            score_tracker.latest_scores[node_id] = current_score
                        else:
                            # 已存在的节点，只更新latest_score
                            score_tracker.latest_scores[node_id] = current_score
                logger.debug(f"[ScoreTracker] 已更新 {len(score_tracker.node_ids)} 个节点的分数")
            
            # 清空缓存
            self.candidate_selector.clear_cache()
            if hasattr(self, 'cached_anomaly_scores'):
                delattr(self, 'cached_anomaly_scores')
            
            # 对每个目标节点进行攻击
            for target_idx, target_node in enumerate(round_target_nodes):
                self.stop_reason = self._observable_stop_reason(
                    current_graph, latest_query_result['all_scores']
                )
                if self.stop_reason:
                    logger.info(f"达到可观察停止条件: {self.stop_reason}")
                    break

                # 仅用于诊断：所选目标中距离阈值相对较远的后20%。
                low_priority_start = max(1, int(np.ceil(len(round_target_nodes) * 0.8)))
                is_low_priority = target_idx >= low_priority_start
                if is_low_priority:
                    logger.debug(f"[低优先级节点] 当前是第{target_idx + 1}个阈值邻近目标")
                
                # 排名语义与目标策略保持一致：rank=1 表示最接近阈值。
                active_scores = latest_query_result['all_scores']
                target_node_score = active_scores.get(target_node, 0.0)
                current_rank_check = get_threshold_margin_rank(
                    target_node, active_scores, self.detector.best_threshold
                )
                
                logger.debug(
                    f"[攻击节点 {target_node} ({target_idx + 1}/{len(round_target_nodes)})] "
                    f"分数={target_node_score:.4f}, 阈值距离排名={current_rank_check}"
                )
                
                # 不能再用“全局最高分 Top-2K”过滤，否则阈值附近目标会被误跳过。
                if (target_node not in active_scores
                        or target_node_score < self.detector.best_threshold):
                    if (
                        self.orthrus_full_aware
                        and target_node in active_scores
                        and target_node_score < self.detector.best_threshold
                    ):
                        self._protected_initial_tp_nodes.add(int(target_node))
                        logger.info(
                            "[OrthrusProtection] target=%s already escaped; "
                            "future macros must keep it below threshold",
                            target_node,
                        )
                    logger.debug(
                        f"节点已不属于当前告警（分数={target_node_score:.4f}, "
                        f"阈值={self.detector.best_threshold:.4f}），跳过"
                    )
                    continue

                if self.orthrus_full_aware:
                    # The cached node score is sufficient for ordinary
                    # black-box attacks, but full-aware mode additionally
                    # needs the native edge that realizes the maximum loss.
                    latest_query_result = self._query_detector(
                        current_graph,
                        self._orthrus_inspect_nodes(target_node),
                    )
                    active_scores = latest_query_result['all_scores']
                    target_node_score = active_scores.get(
                        target_node, float('-inf')
                    )
                    if target_node_score < self.detector.best_threshold:
                        self._protected_initial_tp_nodes.add(int(target_node))
                        logger.info(
                            "[OrthrusFullAware] target=%s escaped before its "
                            "turn; added to protected set",
                            target_node,
                        )
                        continue
                    current_rank_check = get_threshold_margin_rank(
                        target_node,
                        active_scores,
                        self.detector.best_threshold,
                    )
                    anchor_edge = self._orthrus_max_loss_edge(
                        latest_query_result, target_node
                    )
                    logger.info(
                        "[OrthrusMaxLoss] target=%s node_score=%.6f edge=%s",
                        target_node,
                        target_node_score,
                        anchor_edge,
                    )
                else:
                    anchor_edge = None

                # Count the target as an attempted attack before candidate
                # generation.  Targets with no legal edit remain failures in
                # the targeted-ASR denominator.
                self.attempted_target_nodes.add(int(target_node))
                
                # 【关键修复】直接使用query_result的分数作为本轮初始分数
                # 【原因】后续轮次的新Top-K节点可能不在ScoreTracker的initial_scores中
                round_initial_score = active_scores.get(target_node, 0.0)
                
                # 【修复】本轮初始排名使用全图真实排名（而非攻击顺序）
                # 目标列表经过逃逸过滤+扩大范围补充后，攻击顺序≠全图排名
                round_initial_rank = current_rank_check
                
                logger.debug(f"[攻击节点 {target_node}] 初始分数={round_initial_score:.4f}, 初始排名={round_initial_rank}")
                
                # 初始化ModificationLog
                mod_log = ModificationLog(current_graph)
                
                predicted_score = round_initial_score  # 【关键修复】从本轮初始分数开始
                
                # 创建AttackState
                try:
                    init_in_deg = current_graph.in_degrees(target_node)
                    init_out_deg = current_graph.out_degrees(target_node)
                    init_in_deg = int(init_in_deg) if isinstance(init_in_deg, (int, float)) else init_in_deg.item()
                    init_out_deg = int(init_out_deg) if isinstance(init_out_deg, (int, float)) else init_out_deg.item()
                except Exception:
                    init_in_deg, init_out_deg = 0, 0
                attack_state = AttackState(
                    target_node,
                    init_in_deg,
                    init_out_deg,
                    action_dim=self.config.ACTION_DIM,
                )
                target_node_type = self._get_node_type(
                    current_graph, target_node
                )
                
                if target_idx == 0:
                    # 当前轮查询已经是最新观测，不再为候选缓存额外消耗一次查询。
                    self.cached_anomaly_scores = dict(active_scores)
                    max_score = StateEncoder.compute_max_anomaly_score(
                        round_target_nodes, latest_query_result
                    )
                    StateEncoder.set_max_anomaly_score(max_score)
                
                # 每5条同类型边构成一个宏动作；一次检测器反馈对应一个transition。
                macro_experiences = []
                macro_action_type = None
                macro_experience = None
                macro_legal_action_ids = []
                macro_ranked_candidates = []
                macro_candidate_losses = {}
                macro_anchor_edge = anchor_edge
                macro_graph_before = None
                macro_detector_mod_count = 0
                macro_modification_record_start = 0
                macro_total_actions_start = 0
                macro_target_actions_start = 0
                macro_attack_state_before = None
                macro_recent_actions_before = []
                macro_target_was_attacked = False
                
                # 【修复】用本轮query_result中的真实排名初始化last_known_rank
                last_known_rank = current_rank_check
                previous_observation_scores = dict(active_scores)
                target_actions_applied = 0
                for step in range(self.config.MAX_STEPS_PER_NODE):
                    step_start_time = time.time()
                    macro_start = (
                        step % self.config.MACRO_ACTION_LENGTH == 0
                    )

                    # 每5个动作必须查询一次；没有查询预算就不再盲目修改图。
                    if (
                        step % self.config.MACRO_ACTION_LENGTH == 0
                        and self.config.MAX_QUERIES is not None
                    ):
                        if self.query_count >= self.config.MAX_QUERIES:
                            self.stop_reason = 'max_queries'
                            break
                    
                    budget_info = [self.config.MAX_STEPS_PER_NODE - step]
                    
                    # 【修复】使用最近一次真实全图排名（而非恒为1的伪排名）
                    current_score = predicted_score
                    current_rank = last_known_rank
                    
                    # 编码统一50维跨数据集状态（检测器可追加私有特征）
                    state = StateEncoder.encode_state(
                        current_graph, target_node, current_score, current_rank,
                        [], budget_info, attack_state=attack_state,
                        top_k=self.top_k,
                        previous_context=self._state_feedback_context(),
                    )
                    state = self.detector.encode_attack_state(
                        state, graph=current_graph, target_node=target_node,
                        action_state=attack_state
                    ).to(self.config.DEVICE)
                    
                    # All policies share the same target-type-aware legal mask.
                    # CandidateSelector caches only the expensive target
                    # adjacency/two-hop neighborhood. Pools are still rebuilt
                    # against the latest pending edges, preserving the exact
                    # endpoint ordering and selection semantics.
                    candidate_pools = self.candidate_selector.legal_candidate_pools(
                        current_graph,
                        target_node,
                        top_k=50,
                        anomaly_scores=getattr(
                            self, 'cached_anomaly_scores', None
                        ),
                        pending_edges=mod_log.pending_edge_pairs(),
                        max_candidate_score=(
                            self.detector.best_threshold
                            if self.action_policy == 'benign_mimicry'
                            else None
                        ),
                    )
                    if not candidate_pools:
                        logger.info(
                            f"[ActionMask] target={target_node}, "
                            f"step={step + 1}: 没有语义合法动作"
                        )
                        break
                    if macro_start:
                        required_candidates = min(
                            self.config.MACRO_ACTION_LENGTH,
                            self.config.MAX_STEPS_PER_NODE - step,
                        )
                        macro_legal_action_ids = sorted(
                            action_id
                            for action_id, pool in candidate_pools.items()
                            if len(pool) >= required_candidates
                        )
                        if not macro_legal_action_ids:
                            logger.info(
                                f"[MacroActionMask] target={target_node}, "
                                f"macro={step // self.config.MACRO_ACTION_LENGTH + 1}: "
                                f"没有至少{required_candidates}个合法端点的动作"
                            )
                            break
                        action_mask = torch.zeros(
                            self.config.ACTION_DIM,
                            dtype=torch.bool,
                            device=self.config.DEVICE,
                        )
                        action_mask[macro_legal_action_ids] = True
                        if self.action_policy == 'greedy':
                            macro_action_type = (
                                self.greedy_policy.select_action_type(
                                    macro_legal_action_ids
                                )
                            )
                            log_prob = torch.tensor(
                                0.0, dtype=torch.float32,
                                device=self.config.DEVICE,
                            )
                            value = torch.tensor(
                                0.0, dtype=torch.float32,
                                device=self.config.DEVICE,
                            )
                            policy_stats = {
                                'warmup': False,
                                'entropy': 0.0,
                                'legal_probabilities': {
                                    action_id: float(
                                        action_id == macro_action_type
                                    )
                                    for action_id in macro_legal_action_ids
                                },
                            }
                        elif self.action_policy == 'random':
                            macro_action_type = self.action_rng.choice(
                                macro_legal_action_ids
                            )
                            log_prob = torch.tensor(
                                -math.log(len(macro_legal_action_ids)),
                                dtype=torch.float32,
                                device=self.config.DEVICE,
                            )
                            value = torch.tensor(
                                0.0, dtype=torch.float32,
                                device=self.config.DEVICE,
                            )
                            uniform_probability = (
                                1.0 / len(macro_legal_action_ids)
                            )
                            policy_stats = {
                                'warmup': False,
                                'entropy': math.log(
                                    len(macro_legal_action_ids)
                                ),
                                'legal_probabilities': {
                                    action_id: round(
                                        uniform_probability, 6
                                    )
                                    for action_id in macro_legal_action_ids
                                },
                            }
                        elif self.action_policy == 'linucb':
                            if self.orthrus_full_aware:
                                macro_action_type = self.linucb_policy.select_action(
                                    state,
                                    macro_legal_action_ids,
                                    target_type=target_node_type,
                                )
                            else:
                                macro_action_type = self.linucb_policy.select_action(
                                    state, macro_legal_action_ids
                                )
                            log_prob = torch.tensor(
                                0.0, dtype=torch.float32,
                                device=self.config.DEVICE,
                            )
                            value = torch.tensor(
                                0.0, dtype=torch.float32,
                                device=self.config.DEVICE,
                            )
                            policy_stats = dict(
                                self.linucb_policy.last_action_stats
                            )
                        elif self.action_policy == 'double_dqn':
                            macro_action_type = (
                                self.double_dqn_policy.select_action(
                                    state, action_mask
                                )
                            )
                            log_prob = torch.tensor(
                                0.0, dtype=torch.float32,
                                device=self.config.DEVICE,
                            )
                            value = torch.tensor(
                                0.0, dtype=torch.float32,
                                device=self.config.DEVICE,
                            )
                            policy_stats = dict(
                                self.double_dqn_policy.last_action_stats
                            )
                        elif self.action_policy == 'benign_mimicry':
                            macro_action_type = (
                                self.benign_mimicry_policy.select_action(
                                    current_graph,
                                    target_node,
                                    macro_legal_action_ids,
                                    additions=required_candidates,
                                )
                            )
                            log_prob = torch.tensor(
                                0.0, dtype=torch.float32,
                                device=self.config.DEVICE,
                            )
                            value = torch.tensor(
                                0.0, dtype=torch.float32,
                                device=self.config.DEVICE,
                            )
                            policy_stats = dict(
                                self.benign_mimicry_policy.last_action_stats
                            )
                        else:
                            (
                                macro_action_type,
                                log_prob,
                                value,
                            ) = self.ppo_agent.select_action(
                                state,
                                action_mask=action_mask,
                                selection_mode=self.ppo_selection_mode,
                            )
                            policy_stats = getattr(
                                self.ppo_agent, 'last_action_stats', {}
                            )
                        macro_experience = {
                            'state': state.detach().clone(),
                            'action_type': int(macro_action_type),
                            'log_prob': log_prob.detach().clone(),
                            'value': value.detach().clone(),
                            'action_mask': action_mask.detach().clone(),
                            'reward': 0.0,
                            'done': False,
                            'macro_index': (
                                step // self.config.MACRO_ACTION_LENGTH + 1
                            ),
                            'micro_operations': [],
                        }
                        if self.orthrus_full_aware:
                            macro_anchor_edge = self._orthrus_max_loss_edge(
                                latest_query_result, target_node
                            )
                            if not macro_anchor_edge:
                                raise RuntimeError(
                                    "Orthrus full-aware query did not return the "
                                    "target's maximum-loss edge"
                                )
                            (
                                macro_ranked_candidates,
                                macro_candidate_losses,
                            ) = self._rank_orthrus_candidates_by_native_loss(
                                current_graph,
                                target_node,
                                int(macro_action_type),
                                candidate_pools[int(macro_action_type)],
                                macro_anchor_edge,
                            )
                            macro_graph_before = current_graph
                            macro_detector_mod_count = len(
                                self.detector.modifications
                            )
                            macro_modification_record_start = len(
                                self.modification_records
                            )
                            macro_total_actions_start = self.total_actions
                            macro_target_actions_start = target_actions_applied
                            macro_attack_state_before = copy.deepcopy(attack_state)
                            macro_recent_actions_before = list(
                                self._recent_action_ids
                            )
                            macro_target_was_attacked = (
                                int(target_node) in self.attacked_nodes
                            )
                        logger.info(
                            f"[MacroActionStart] round={round_idx + 1}, "
                            f"target={target_node}, "
                            f"macro={macro_experience['macro_index']}, "
                            f"action_id={macro_action_type}"
                            f"({DirectedActionTypes.get_action_name(macro_action_type)}), "
                            f"repeat={required_candidates}, "
                            f"legal_actions={macro_legal_action_ids}, "
                            f"log_prob={float(log_prob.detach().cpu().item()):.6f}, "
                            f"state_value={float(value.detach().cpu().item()):.6f}, "
                            f"warmup={policy_stats.get('warmup', False)}, "
                            f"selection_mode="
                            f"{policy_stats.get('selection_mode', 'not_applicable')}, "
                            f"policy_entropy="
                            f"{policy_stats.get('entropy', float('nan')):.6f}, "
                            f"normalized_entropy="
                            f"{policy_stats.get('normalized_entropy', float('nan')):.6f}, "
                            f"max_probability="
                            f"{policy_stats.get('max_action_probability', float('nan')):.6f}, "
                            f"top1_top2_gap="
                            f"{policy_stats.get('top1_top2_probability_gap', float('nan')):.6f}, "
                            f"kl_to_uniform="
                            f"{policy_stats.get('kl_to_uniform', float('nan')):.6f}, "
                            f"epsilon={policy_stats.get('epsilon', float('nan')):.6f}, "
                            f"explore={policy_stats.get('explore', False)}, "
                            f"legal_probabilities="
                            f"{policy_stats.get('legal_probabilities', {})}, "
                            f"selection_scores="
                            f"{policy_stats.get('selection_scores', {})}"
                        )

                    action_type = int(macro_action_type)
                    candidates = candidate_pools.get(action_type, [])
                    if not candidates:
                        raise RuntimeError(
                            "selected macro action exhausted its legal "
                            "candidate pool before five edge insertions"
                        )
                    if self.orthrus_full_aware:
                        selected_candidate = next(
                            (
                                int(candidate)
                                for candidate in macro_ranked_candidates
                                if int(candidate) in candidates
                            ),
                            None,
                        )
                        if selected_candidate is None:
                            raise RuntimeError(
                                "Orthrus native-loss-ranked macro exhausted its "
                                "legal candidate pool"
                            )
                        macro_ranked_candidates.remove(selected_candidate)
                        candidate_idx = candidates.index(selected_candidate)
                    elif self.candidate_policy == 'random_legal':
                        # Candidate-selector ablation: retain only semantic
                        # legality, then choose the endpoint uniformly instead
                        # of applying the proposed locality/low-score ranking.
                        candidate_idx = self.action_rng.randrange(
                            len(candidates)
                        )
                    else:
                        candidate_idx = (
                            GreedyActionPolicy.select_candidate_index(
                                candidates,
                                getattr(self, 'cached_anomaly_scores', None),
                                current_graph,
                            )
                        )
                    candidate_pool_size = len(candidates)
                    selected_candidate = candidates[candidate_idx]
                    
                    # 【关键修改】根据动作类型添加有向边
                    resolved = self.candidate_selector.resolve_operation(
                        current_graph,
                        target_node,
                        action_type,
                        selected_candidate,
                    )
                    src = int(resolved['src'])
                    dst = int(resolved['dst'])
                    mod_log.add_edge(
                        src,
                        dst,
                        action_type,
                        edge_type=resolved['edge_type'],
                    )
                    self.detector.record_attack_modification(
                        kind='edge',
                        src=src,
                        dst=dst,
                        action_type=action_type,
                        edge_type=int(resolved['edge_type']),
                        before_time=(
                            macro_anchor_edge.get('time')
                            if self.orthrus_full_aware else None
                        ),
                        window=(
                            macro_anchor_edge.get('window')
                            if self.orthrus_full_aware else None
                        ),
                        graph=current_graph,
                    )
                    attack_state.record_add_edge(
                        action_type,
                        is_incoming=(dst == int(target_node)),
                    )
                    modification_detail = {
                        'kind': 'add_edge',
                        'src': src,
                        'dst': dst,
                        'src_type': int(resolved['src_type']),
                        'dst_type': int(resolved['dst_type']),
                        'native_probe_loss': (
                            macro_candidate_losses.get(int(selected_candidate))
                            if self.orthrus_full_aware else None
                        ),
                        'insert_before_time': (
                            int(macro_anchor_edge['time'])
                            if self.orthrus_full_aware else None
                        ),
                        'insert_window': (
                            int(macro_anchor_edge['window'])
                            if self.orthrus_full_aware else None
                        ),
                    }

                    target_actions_applied += 1
                    self.total_actions += 1
                    self.attacked_nodes.add(target_node)
                    self._record_recent_action(action_type)
                    action_name = DirectedActionTypes.get_action_name(action_type)
                    action_spec = self.action_schema.get(action_type)
                    operation = (
                        f"{src}(type={resolved['src_type']})"
                        f"->{dst}(type={resolved['dst_type']}), "
                        f"edge={action_spec.edge_type}"
                        f"({action_spec.edge_name})"
                    )
                    self.modification_records.append({
                        'sequence': len(self.modification_records) + 1,
                        'round': round_idx + 1,
                        'target_position': target_idx + 1,
                        'target': int(target_node),
                        'step': step + 1,
                        'macro_index': int(macro_experience['macro_index']),
                        'micro_index': (
                            step % self.config.MACRO_ACTION_LENGTH + 1
                        ),
                        'action_policy': self.action_policy,
                        'action_id': int(action_type),
                        'action_name': action_name,
                        'edge_type': int(DirectedActionTypes.get_edge_type(action_type)),
                        'selected_candidate': int(selected_candidate),
                        'candidate_pool_size': candidate_pool_size,
                        'target_score_before_batch': float(current_score),
                        'operation': operation,
                        **modification_detail,
                    })
                    logger.info(
                        f"[Action] round={round_idx + 1}, "
                        f"target={target_node}({target_idx + 1}/{len(round_target_nodes)}), "
                        f"step={step + 1}/{self.config.MAX_STEPS_PER_NODE}, "
                        f"macro={macro_experience['macro_index']}, "
                        f"micro={step % self.config.MACRO_ACTION_LENGTH + 1}/"
                        f"{self.config.MACRO_ACTION_LENGTH}, "
                        f"action_id={action_type}({action_name}), operation={operation}, "
                        f"macro_legal_actions={macro_legal_action_ids}, "
                        f"candidate_pool={candidate_pool_size}, selected_candidate={selected_candidate}, "
                        f"target_score(last_query)={current_score:.6f}, "
                        f"pending_actions={target_actions_applied}"
                    )

                    macro_experience['micro_operations'].append({
                        'step': step + 1,
                        'candidate': int(selected_candidate),
                        'operation': operation,
                    })

                    # 一个宏动作的5条同类型边全部应用后只查询一次。
                    if (
                        (step + 1) % self.config.MACRO_ACTION_LENGTH == 0
                    ):
                        batch_label = (
                            f"{step - self.config.MACRO_ACTION_LENGTH + 2}-"
                            f"{step + 1}"
                        )
                        logger.info(
                            f"[Query] target={target_node}, batch_steps="
                            f"{batch_label}, "
                            f"macro_action={action_name} 开始检测器查询"
                        )
                        modified_graph = mod_log.apply_modifications()
                        if (
                            modified_graph.number_of_nodes()
                            != current_graph.number_of_nodes()
                        ):
                            raise RuntimeError(
                                "edge-only attack unexpectedly changed the node count"
                            )
                        batch_added_edges = (
                            modified_graph.number_of_edges() - current_graph.number_of_edges()
                        )
                        post_query = self._query_detector(
                            modified_graph,
                            (
                                self._orthrus_inspect_nodes(target_node)
                                if self.orthrus_full_aware else None
                            ),
                        )
                        trial_scores = post_query['all_scores']
                        trial_score = trial_scores.get(
                            target_node, predicted_score
                        )
                        trial_rank = get_threshold_margin_rank(
                            target_node,
                            trial_scores,
                            self.detector.best_threshold,
                        )

                        reward_info = compute_blackbox_reward(
                            target_node=target_node,
                            previous_scores=previous_observation_scores,
                            current_scores=trial_scores,
                            threshold=self.detector.best_threshold,
                            added_edges=batch_added_edges,
                            batch_actions=self.config.MACRO_ACTION_LENGTH,
                            direct_weight=self.config.SCORE_DROP_WEIGHT,
                            crossing_weight=self.config.THRESHOLD_CROSS_WEIGHT,
                            indirect_node_ids=(
                                self._initial_alert_nodes - self.attacked_nodes
                            ),
                            indirect_weight=self.config.INDIRECT_WEIGHT,
                            perturbation_weight=self.config.PERTURBATION_WEIGHT,
                        )
                        combined_reward_base = reward_info.total
                        previous_target_score = previous_observation_scores.get(
                            target_node, trial_score
                        )
                        direct_contribution = (
                            self.config.SCORE_DROP_WEIGHT * reward_info.direct_score
                        )
                        crossing_contribution = (
                            self.config.THRESHOLD_CROSS_WEIGHT * reward_info.threshold_cross
                        )
                        indirect_contribution = (
                            self.config.INDIRECT_WEIGHT * reward_info.indirect_reward
                        )
                        perturbation_penalty = (
                            self.config.PERTURBATION_WEIGHT * reward_info.perturbation_cost
                        )
                        raw_reward = float(combined_reward_base)
                        score_increased = bool(
                            self.orthrus_full_aware
                            and trial_score > previous_target_score + 1e-12
                        )
                        protected_realerts = sorted(
                            int(node_id)
                            for node_id in self._protected_initial_tp_nodes
                            if trial_scores.get(node_id, float('-inf'))
                            >= self.detector.best_threshold
                        )
                        rolled_back = bool(
                            self.orthrus_full_aware
                            and (score_increased or protected_realerts)
                        )
                        rollback_reasons = []
                        if score_increased:
                            rollback_reasons.append('target_score_increased')
                        if protected_realerts:
                            rollback_reasons.append('protected_target_realerted')

                        if rolled_back:
                            # The detector query is retained in AQT/query
                            # accounting, but none of the five attempted edits
                            # survive in the graph, adapter, or edit budget.
                            self.detector.rollback_attack_modifications(
                                macro_detector_mod_count
                            )
                            current_graph = macro_graph_before
                            mod_log = ModificationLog(current_graph)
                            del self.modification_records[
                                macro_modification_record_start:
                            ]
                            self.total_actions = macro_total_actions_start
                            target_actions_applied = macro_target_actions_start
                            attack_state = macro_attack_state_before
                            self._recent_action_ids.clear()
                            self._recent_action_ids.extend(
                                macro_recent_actions_before
                            )
                            if not macro_target_was_attacked:
                                self.attacked_nodes.discard(int(target_node))
                            current_scores = dict(previous_observation_scores)
                            current_score = current_scores.get(
                                target_node, predicted_score
                            )
                            current_rank = get_threshold_margin_rank(
                                target_node,
                                current_scores,
                                self.detector.best_threshold,
                            )
                            combined_reward_base = -max(
                                1.0, abs(raw_reward)
                            )
                            self.orthrus_full_aware_stats[
                                'rolled_back_macros'
                            ] += 1
                            if score_increased:
                                self.orthrus_full_aware_stats[
                                    'score_increase_rollbacks'
                                ] += 1
                            if protected_realerts:
                                self.orthrus_full_aware_stats[
                                    'protected_target_rollbacks'
                                ] += 1
                            logger.info(
                                "[OrthrusRollback] target=%s macro=%s "
                                "trial_score=%.6f restored_score=%.6f "
                                "reasons=%s protected=%s",
                                target_node,
                                macro_experience['macro_index'],
                                trial_score,
                                current_score,
                                rollback_reasons,
                                protected_realerts,
                            )
                        else:
                            current_scores = trial_scores
                            current_score = trial_score
                            current_rank = trial_rank
                            last_known_rank = current_rank
                            for node_id in score_tracker.node_ids:
                                if node_id in current_scores:
                                    score_tracker.update_latest_score(
                                        node_id, current_scores[node_id]
                                    )
                            if hasattr(self, 'cached_anomaly_scores'):
                                self.cached_anomaly_scores.update(current_scores)
                            current_graph = modified_graph
                            # Future macros start from the committed graph and
                            # retain only their new delta.
                            mod_log = ModificationLog(current_graph)
                            latest_query_result = post_query
                            self._graph_modified = True
                            predicted_score = current_score
                            previous_observation_scores = dict(current_scores)
                            if self.orthrus_full_aware:
                                self.orthrus_full_aware_stats[
                                    'accepted_macros'
                                ] += 1
                                for node_id in self._initial_tp_cycle_order:
                                    if current_scores.get(
                                        int(node_id), float('inf')
                                    ) < self.detector.best_threshold:
                                        self._protected_initial_tp_nodes.add(
                                            int(node_id)
                                        )

                        query_record = {
                            'kind': 'attack_feedback',
                            'query_index': self.query_count,
                            'round': round_idx + 1,
                            'target': int(target_node),
                            'macro_index': int(
                                macro_experience['macro_index']
                            ),
                            'macro_action_id': int(action_type),
                            'macro_action_name': action_name,
                            'batch_steps': batch_label,
                            'score_before': float(previous_target_score),
                            'score_after': float(current_score),
                            'trial_score_after': float(trial_score),
                            'accepted': not rolled_back,
                            'rolled_back': rolled_back,
                            'rollback_reasons': rollback_reasons,
                            'protected_realerts': protected_realerts,
                            'threshold': float(self.detector.best_threshold),
                            'crossed_threshold': bool(
                                previous_target_score >= self.detector.best_threshold
                                > current_score
                            ),
                            'trial_crossed_threshold': bool(
                                previous_target_score >= self.detector.best_threshold
                                > trial_score
                            ),
                            'raw_reward_before_rollback': raw_reward,
                            'total_reward': float(combined_reward_base),
                            'direct_contribution': float(direct_contribution),
                            'crossing_contribution': float(crossing_contribution),
                            'indirect_contribution': float(indirect_contribution),
                            'indirect_reward': float(
                                reward_info.indirect_reward
                            ),
                            'indirect_score_drop': float(
                                reward_info.indirect_score_drop
                            ),
                            'indirect_cross': float(reward_info.indirect_cross),
                            'indirect_cross_count': int(
                                reward_info.indirect_cross_count
                            ),
                            'indirect_positive_count': int(
                                reward_info.indirect_positive_count
                            ),
                            'indirect_candidate_count': int(
                                reward_info.indirect_candidate_count
                            ),
                            'perturbation_penalty': float(perturbation_penalty),
                            'added_edges': int(batch_added_edges),
                            'committed_edges': (
                                0 if rolled_back else int(batch_added_edges)
                            ),
                            'policy_selection': dict(policy_stats),
                        }
                        self.query_records.append(query_record)
                        logger.info(
                            f"[RewardBatch] target={target_node}, "
                            f"macro={macro_experience['macro_index']}, "
                            f"action={action_name}, steps={batch_label}, "
                            f"score={previous_target_score:.6f}->{trial_score:.6f}, "
                            f"effective={current_score:.6f}, accepted={not rolled_back}, "
                            f"threshold={self.detector.best_threshold:.6f}, "
                            f"margin_rank={current_rank}, total_reward={combined_reward_base:.6f}"
                        )
                        logger.info(
                            f"[RewardBreakdown] direct={direct_contribution:+.6f} "
                            f"(raw={reward_info.direct_score:+.6f}, "
                            f"weight={self.config.SCORE_DROP_WEIGHT}), "
                            f"cross={crossing_contribution:+.6f} "
                            f"(raw={reward_info.threshold_cross:.0f}, "
                            f"weight={self.config.THRESHOLD_CROSS_WEIGHT}), "
                            f"indirect={indirect_contribution:+.6f} "
                            f"(raw={reward_info.indirect_reward:.6f}, "
                            f"top5_drop={reward_info.indirect_score_drop:.6f}, "
                            f"bounded_cross={reward_info.indirect_cross:.6f}, "
                            f"crossed={reward_info.indirect_cross_count}, "
                            f"positive={reward_info.indirect_positive_count}, "
                            f"candidates={reward_info.indirect_candidate_count}, "
                            f"weight={self.config.INDIRECT_WEIGHT}), "
                            f"perturb_penalty=-{perturbation_penalty:.6f} "
                            f"(raw={reward_info.perturbation_cost:.6f}, "
                            f"ΔE={batch_added_edges})"
                        )
                        self._update_feedback_context(
                            current_graph,
                            target_node,
                            action_type,
                            previous_target_score,
                            trial_score,
                            combined_reward_base,
                        )
                        
                        macro_experience['reward'] = float(
                            combined_reward_base
                        )
                        macro_experiences.append(macro_experience)
                        logger.info(
                            f"[MacroTransition] target={target_node}, "
                            f"macro={macro_experience['macro_index']}, "
                            f"action={action_name}, "
                            f"micro_steps={len(macro_experience['micro_operations'])}, "
                            f"reward={combined_reward_base:+.6f}, "
                            f"rollout_pending="
                            f"{len(self._ppo_rollout_buffer) + len(macro_experiences)}"
                        )
                        if self.action_policy == 'greedy':
                            greedy_snapshot = self.greedy_policy.observe_batch(
                                [action_type],
                                combined_reward_base,
                            )
                            logger.info(
                                f"[GreedyUpdate] target={target_node}, "
                                f"macro={macro_experience['macro_index']}, "
                                f"action={action_name}, "
                                f"reward={combined_reward_base:+.6f}, "
                                f"counts={greedy_snapshot['counts']}, "
                                f"mean_rewards="
                                f"{[round(v, 6) for v in greedy_snapshot['mean_rewards']]}"
                            )
                            query_record['policy_update'] = {
                                'kind': 'greedy',
                                **greedy_snapshot,
                            }
                        elif self.action_policy == 'linucb':
                            if self.orthrus_full_aware:
                                linucb_snapshot = self.linucb_policy.observe(
                                    macro_experience['state'],
                                    action_type,
                                    combined_reward_base,
                                    target_type=target_node_type,
                                )
                            else:
                                linucb_snapshot = self.linucb_policy.observe(
                                    macro_experience['state'],
                                    action_type,
                                    combined_reward_base,
                                )
                            logger.info(
                                f"[LinUCBUpdate] target={target_node}, "
                                f"macro={macro_experience['macro_index']}, "
                                f"action={action_name}, "
                                f"reward={combined_reward_base:+.6f}, "
                                f"updates={linucb_snapshot['total_updates']}, "
                                f"counts={linucb_snapshot['counts']}, "
                                f"mean_rewards="
                                f"{[round(v, 6) for v in linucb_snapshot['mean_rewards']]}"
                            )
                            query_record['policy_update'] = {
                                'kind': 'linucb',
                                **linucb_snapshot,
                            }
                        
                        # 只依据可观察阈值判断当前目标，不读取 metadata 真值。
                        if current_score < self.detector.best_threshold:
                            self.observable_crossed_nodes.add(target_node)
                            logger.info(
                                f"可观察目标告警已跨阈值(节点={target_node}, "
                                f"分数{current_score:.4f} < "
                                f"阈值{self.detector.best_threshold:.4f}); "
                                "终止该目标后续宏动作"
                            )
                            break

                        self.stop_reason = self._observable_stop_reason(
                            current_graph, current_scores
                        )
                        if self.stop_reason:
                            logger.info(f"达到可观察停止条件: {self.stop_reason}")
                            break

                    # 记录进度
                    step_elapsed = time.time() - step_start_time
                    logger.info(
                        f"[StepDone] target={target_node}, step={step + 1}, "
                        f"action={DirectedActionTypes.get_action_name(action_type)}, "
                        f"elapsed={step_elapsed:.4f}s"
                    )
                
                if macro_experiences:
                    # A target boundary is an episode boundary for GAE.
                    macro_experiences[-1]['done'] = True
                    rewards = [
                        item['reward'] for item in macro_experiences
                    ]
                    action_histogram = {}
                    for item in macro_experiences:
                        label = DirectedActionTypes.get_action_name(
                            item['action_type']
                        )
                        action_histogram[label] = (
                            action_histogram.get(label, 0) + 1
                        )
                    if self.action_policy == 'ppo_online':
                        self._ppo_rollout_buffer.extend(macro_experiences)
                        logger.info(
                            f"[PPORollout] target={target_node}, "
                            f"added_macros={len(macro_experiences)}, "
                            f"buffer={len(self._ppo_rollout_buffer)}/"
                            f"{self.config.PPO_ROLLOUT_SIZE}, "
                            f"actions={action_histogram}, "
                            f"rewards={[round(value, 6) for value in rewards]}"
                        )
                        self._update_ppo_rollout(force=False)
                    elif self.action_policy == 'double_dqn':
                        dqn_snapshot = (
                            self.double_dqn_policy.observe_episode(
                                macro_experiences
                            )
                        )
                        update_stats = dqn_snapshot['last_update']
                        logger.info(
                            f"[DoubleDQNUpdate] target={target_node}, "
                            f"added_transitions={len(macro_experiences)}, "
                            f"replay={dqn_snapshot['replay_size']}/"
                            f"{self.config.DDQN_REPLAY_CAPACITY}, "
                            f"updates={dqn_snapshot['total_updates']}, "
                            f"epsilon={dqn_snapshot['epsilon']:.6f}, "
                            f"updated={update_stats.get('updated', False)}, "
                            f"loss={update_stats.get('loss', float('nan')):.6f}, "
                            f"q_mean={update_stats.get('q_mean', float('nan')):.6f}, "
                            f"target_mean="
                            f"{update_stats.get('target_mean', float('nan')):.6f}, "
                            f"target_synced="
                            f"{update_stats.get('target_synced', False)}, "
                            f"actions={action_histogram}, "
                            f"rewards={[round(value, 6) for value in rewards]}"
                        )
                        self.query_records.append({
                            'kind': 'policy_update',
                            'policy': 'double_dqn',
                            'round': round_idx + 1,
                            'target': int(target_node),
                            **update_stats,
                        })
                    elif self.action_policy in {'greedy', 'linucb'}:
                        logger.info(
                            f"[AdaptivePolicyBatch] mode={self.action_policy}, "
                            f"target={target_node}, "
                            f"macro_count={len(macro_experiences)}, "
                            f"policy_updates={self._policy_update_count()}, "
                            f"reward_mean={float(np.mean(rewards)):+.6f}, "
                            f"actions={action_histogram}"
                        )
                    else:
                        logger.info(
                            f"[PolicyNoUpdate] mode={self.action_policy}, "
                            f"target={target_node}, "
                            f"macro_count={len(macro_experiences)}, "
                            f"reward_mean={float(np.mean(rewards)):+.6f}, "
                            f"reward_min={min(rewards):+.6f}, "
                            f"reward_max={max(rewards):+.6f}, "
                            f"actions={action_histogram}"
                        )
                    if is_low_priority and float(np.mean(rewards)) < 0:
                        logger.info(
                            "[PolicyWarning] 低优先级节点宏动作平均奖励为负: "
                            f"rewards={[round(value, 6) for value in rewards]}"
                        )

                observable_drop = round_initial_score - predicted_score
                if observable_drop > self.config.MIN_SCORE_IMPROVEMENT:
                    self._consecutive_no_improvement = 0
                else:
                    self._consecutive_no_improvement += 1
                if self.stop_reason:
                    break

            # 每轮结束只使用最近一次黑盒观测，不额外查询。
            round_end_query = latest_query_result
            
            logger.debug(f"\n第{round_idx + 1}轮攻击节点分数汇总:")
            for node_id in round_target_nodes:
                init_score = score_tracker.get_initial_score(node_id)
                final_score = round_end_query['all_scores'].get(node_id, 0.0)
                change = final_score - init_score
                rank = get_threshold_margin_rank(
                    node_id, round_end_query['all_scores'], self.detector.best_threshold
                )
                logger.debug(f"  Node {node_id}: {init_score:.4f} -> {final_score:.4f} ({change:+.4f}) rank={rank}")
            
            # 【新增】记录本轮攻击的节点，用于区分直接/间接影响
            self.detector._last_attacked_nodes = set(self.attacked_nodes)
            
            # 【关键修复】过滤掉一开始就低于阈值的节点
            initial_threshold = self.detector.best_threshold
            logger.debug(f"本轮分数变化详情 (仅显示初始分数>={initial_threshold:.4f}的节点)")
            
            nodes_to_display = []
            for node_id in score_tracker.node_ids:
                init_score = score_tracker.get_initial_score(node_id)
                if init_score >= initial_threshold:
                    nodes_to_display.append(node_id)
            
            if len(nodes_to_display) > 0:
                logger.debug(f"有 {len(nodes_to_display)} 个节点初始分数>=阈值")
            else:
                logger.debug("无符合条件的节点")
            
            current_alerts = {
                node_id for node_id, score in round_end_query['all_scores'].items()
                if node_id < self._original_node_count
                and score >= self.detector.best_threshold
            }
            new_alert_count = len(current_alerts - self._initial_alert_nodes)
            logger.info(
                f"第{round_idx + 1}轮结束（仅可观察统计）: "
                f"当前告警={len(current_alerts)}, 新增告警={new_alert_count}, "
                f"累计跨阈值目标={len(self.observable_crossed_nodes)}, "
                f"查询={self.query_count}, 已攻击唯一节点={len(self.attacked_nodes)}"
            )
            if self.stop_reason:
                break
            
        
        # 不丢弃实验末尾不足一个完整rollout的宏动作。它们仍是完整的
        # transition，只是批量数量小于PPO_ROLLOUT_SIZE。
        self._update_ppo_rollout(force=True)

        # 最终评估
        logger.info("攻击完成！")
        
        # 检测器适配器决定原生评估范围。MAGIC/TRACE在固定的原始跨图
        # 评估集合上计算指标，但只有当前被攻击分图的分数会发生替换。
        logger.info("【攻击后】检测器原生评估范围上的最终性能评估")
        
        best_thres = self.detector.best_threshold
        logger.info(f"使用初始阈值: {best_thres:.4f}")
        
        malicious = set(getattr(
            self.detector, 'true_malicious_set',
            getattr(self.detector, 'malicious_nodes', ()),
        ))

        # 最后一次宏动作反馈（或无攻击时的初始化查询）已经对应current_graph。
        # 最终指标只依赖这些分数与离线标签，无需再次查询同一张图。
        attack_query_count = self.query_count
        final_query = latest_query_result
        self.reused_query_count += 1
        self.query_records.append({
            'kind': 'query_reuse',
            'purpose': 'final_evaluation',
            'saved_detector_queries': 1,
        })
        logger.info(
            f"[QueryReuse] purpose=final_evaluation, "
            f"graph_nodes={current_graph.number_of_nodes()}, "
            f"graph_edges={current_graph.number_of_edges()}, "
            f"cumulative_saved={self.reused_query_count}"
        )
        self.final_graph = current_graph
        self.final_all_scores = dict(final_query['all_scores'])
        original_node_count = original_graph.number_of_nodes()
        valid_malicious = [idx for idx in malicious if idx < original_node_count]
        test_labels = np.zeros(original_node_count)
        if len(valid_malicious) > 0:
            test_labels[valid_malicious] = 1.0
        scores_array = np.asarray([
            final_query['all_scores'].get(i, float('-inf'))
            for i in range(original_node_count)
        ])
        
        final_all_scores = {i: float(scores_array[i]) for i in range(original_node_count)}
        score_integrity = {
            'original_node_count': int(original_node_count),
            'initial_score_count': len(self.initial_all_scores),
            'final_score_count': len(self.final_all_scores),
            'initial_missing_original': sum(
                node_id not in self.initial_all_scores
                for node_id in range(original_node_count)
            ),
            'final_missing_original': sum(
                node_id not in self.final_all_scores
                for node_id in range(original_node_count)
            ),
            'initial_nonfinite': sum(
                not math.isfinite(float(score))
                for score in self.initial_all_scores.values()
            ),
            'final_nonfinite': sum(
                not math.isfinite(float(score))
                for score in self.final_all_scores.values()
            ),
        }
        logger.info(
            f"[ScoreIntegrity] original_nodes={original_node_count}, "
            f"initial_scores={score_integrity['initial_score_count']}, "
            f"final_scores={score_integrity['final_score_count']}, "
            f"initial_missing={score_integrity['initial_missing_original']}, "
            f"final_missing={score_integrity['final_missing_original']}, "
            f"initial_nonfinite={score_integrity['initial_nonfinite']}, "
            f"final_nonfinite={score_integrity['final_nonfinite']}"
        )
        
        # MAGIC/TRACE的原始评估集合跨越多个序列分图。检测器适配器负责
        # 将当前被攻击分图的最终分数替换回固定的全局评估向量；其他检测器
        # 若未提供该接口，则保持原来的当前图metadata评估。
        native_evaluator = getattr(
            self.detector, 'evaluate_native_scores', None
        )
        if callable(native_evaluator):
            final_metrics = native_evaluator(
                final_all_scores, threshold=best_thres
            )
        else:
            from sklearn.metrics import roc_auc_score
            auc = (
                roc_auc_score(test_labels, scores_array)
                if len(np.unique(test_labels)) == 2 else float('nan')
            )
            predicted = scores_array >= best_thres
            positive = test_labels == 1.0
            tp = int(np.sum(predicted & positive))
            fp = int(np.sum(predicted & ~positive))
            fn = int(np.sum(~predicted & positive))
            tn = int(np.sum(~predicted & ~positive))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
            final_metrics = {
                'auc': auc, 'f1': f1, 'precision': precision,
                'recall': recall, 'tn': tn, 'fn': fn,
                'tp': tp, 'fp': fp, 'threshold': best_thres,
            }
        auc = final_metrics['auc']
        tn = int(final_metrics['tn'])
        fn = int(final_metrics['fn'])
        tp = int(final_metrics['tp'])
        fp = int(final_metrics['fp'])
        
        # 下列检测性能指标使用目标检测器的固定原生评估口径。
        logger.info(f"[{self.detector_label}原生metadata评估]")
        logger.info('AUC: {}'.format(auc))
        logger.info('F1: {}'.format(final_metrics['f1']))
        logger.info('PRECISION: {}'.format(final_metrics['precision']))
        logger.info('RECALL: {}'.format(final_metrics['recall']))
        logger.info('TN: {}'.format(tn))
        logger.info('FN: {}'.format(fn))
        logger.info('TP: {}'.format(tp))
        logger.info('FP: {}'.format(fp))
        
        # 图修改率只作为攻击成本报告，不参与成功判定。
        initial_nodes = original_graph.number_of_nodes()
        initial_edges = original_graph.number_of_edges()
        final_nodes = current_graph.number_of_nodes()
        final_edges = current_graph.number_of_edges()
        
        delta_nodes = abs(final_nodes - initial_nodes)
        delta_edges = abs(final_edges - initial_edges)
        graph_modification_rate = (delta_nodes + delta_edges) / (initial_nodes + initial_edges)
        
        logger.info(f"图结构变化: 节点{initial_nodes}->{final_nodes}, 边{initial_edges}->{final_edges}, 修改率={graph_modification_rate:.4f}({graph_modification_rate*100:.2f}%)")
        
        # 黑盒预算效率：仅使用初始告警、最终分数和实际攻击集合。
        delta_e = delta_edges
        delta_v = delta_nodes
        eligible_attacked_nodes = self.attacked_nodes & self._initial_alert_nodes
        all_observable_escaped_nodes = {
            node_id for node_id in self._initial_alert_nodes
            if final_all_scores.get(node_id, float('-inf')) < best_thres
        }
        direct_observable_escaped_nodes = (
            all_observable_escaped_nodes & eligible_attacked_nodes
        )
        indirect_observable_escaped_nodes = (
            all_observable_escaped_nodes - eligible_attacked_nodes
        )
        direct_observable_count = len(direct_observable_escaped_nodes)
        total_observable_count = len(all_observable_escaped_nodes)
        target_evasion_rate = (
            direct_observable_count / len(eligible_attacked_nodes)
            if eligible_attacked_nodes else 0.0
        )
        observable_evasion_amplification = (
            total_observable_count / len(eligible_attacked_nodes)
            if eligible_attacked_nodes else 0.0
        )
        indirect_evasion_factor = (
            len(indirect_observable_escaped_nodes) / len(eligible_attacked_nodes)
            if eligible_attacked_nodes else 0.0
        )
        observable_aab = (
            (delta_e + delta_v) / direct_observable_count
            if direct_observable_count else float('inf')
        )
        avg_steps_per_direct = (
            self.total_actions / direct_observable_count
            if direct_observable_count else float('inf')
        )
        avg_steps_per_total = (
            self.total_actions / total_observable_count
            if total_observable_count else float('inf')
        )
        logger.info(
            f"黑盒预算指标: 实际攻击唯一初始告警={len(eligible_attacked_nodes)}, "
            f"直接最终逃逸={direct_observable_count}, "
            f"间接最终逃逸={len(indirect_observable_escaped_nodes)}, "
            f"总最终逃逸={total_observable_count}"
        )
        logger.info(
            f"TargetEvasionRate={target_evasion_rate:.2%}, "
            f"ObservableEvasionAmplification={observable_evasion_amplification:.4f}, "
            f"IndirectEvasionFactor={indirect_evasion_factor:.4f}"
        )
        logger.info(
            f"ObservableAAB={observable_aab:.4f}, 总动作={self.total_actions}, "
            f"平均动作/直接逃逸={avg_steps_per_direct:.4f}, "
            f"平均动作/总逃逸={avg_steps_per_total:.4f}"
        )
        
        logger.info("攻击前后性能对比报告:")
        
        metrics_list = [
            ('AUC', 'auc'),
            ('F1', 'f1'),
            ('Precision', 'precision'),
            ('Recall', 'recall')
        ]
        
        for name, key in metrics_list:
            before = self.initial_metrics[key]
            after = final_metrics[key]
            change = after - before
            change_rate = (change / (before + 1e-8)) * 100
            
            # 用颜色标记变化（红色=下降，绿色=上升）
            if change < 0:
                change_str = f"{change:+.4f} ↓"
                rate_str = f"{change_rate:+.2f}% ↓"
            else:
                change_str = f"{change:+.4f} ↑"
                rate_str = f"{change_rate:+.2f}% ↑"
            
            logger.info(f"{name}: {before:.4f} -> {after:.4f} ({change_str}, {rate_str})")
        
        logger.info(f"混淆矩阵: TP={self.initial_metrics['tp']}->{final_metrics['tp']}, FP={self.initial_metrics['fp']}->{final_metrics['fp']}, TN={self.initial_metrics['tn']}->{final_metrics['tn']}, FN={self.initial_metrics['fn']}->{final_metrics['fn']}")
        
        auc_drop = self.initial_metrics['auc'] - final_metrics['auc']
        f1_drop = self.initial_metrics['f1'] - final_metrics['f1']
        logger.info(
            f"{self.detector_label}原生指标变化: "
            f"AUC下降{auc_drop:.4f}, F1下降{f1_drop:.4f}"
        )

        # 最终主目标：固定阈值下目标检测器的 metadata TP 是否减少。
        initial_native_tp_nodes = set(self.detector.initial_true_tp_nodes)
        escaped_native_tp_nodes = {
            node_id for node_id in initial_native_tp_nodes
            if final_all_scores.get(node_id, float('-inf')) < best_thres
        }
        self.initial_tp_count = len(initial_native_tp_nodes)
        escaped_count = len(escaped_native_tp_nodes)
        native_tp_escape_rate = (
            escaped_count / self.initial_tp_count if self.initial_tp_count else 0.0
        )
        initial_native_tp = int(self.initial_metrics['tp'])
        final_native_tp = int(final_metrics['tp'])
        # Keep the sign for fair baselines: a strategy that increases TP must
        # report a negative reduction instead of being clipped to zero.
        tp_reduction_count = initial_native_tp - final_native_tp
        tp_reduction_rate = (
            tp_reduction_count / initial_native_tp if initial_native_tp else 0.0
        )
        direct_native_tp_escaped = escaped_native_tp_nodes & self.attacked_nodes
        indirect_native_tp_escaped = escaped_native_tp_nodes - self.attacked_nodes
        attacked_initial_tp_nodes = self.attacked_nodes & initial_native_tp_nodes
        attempted_initial_tp_nodes = (
            self.attempted_target_nodes & initial_native_tp_nodes
        )
        attacked_initial_fp_nodes = self.attacked_nodes & (
            self._initial_alert_nodes - initial_native_tp_nodes
        )
        attacked_late_alert_nodes = self.attacked_nodes - self._initial_alert_nodes
        escaped_attempted_initial_tp_nodes = (
            escaped_native_tp_nodes & attempted_initial_tp_nodes
        )
        direct_tp_asr = (
            len(escaped_attempted_initial_tp_nodes)
            / len(attempted_initial_tp_nodes)
            if attempted_initial_tp_nodes else None
        )
        direct_initial_fp_escaped = (
            direct_observable_escaped_nodes & attacked_initial_fp_nodes
        )
        direct_fp_evasion_rate = (
            len(direct_initial_fp_escaped) / len(attacked_initial_fp_nodes)
            if attacked_initial_fp_nodes else None
        )
        transient_crossed_initial_tp = (
            self.observable_crossed_nodes & initial_native_tp_nodes
        )
        transient_crossed_initial_fp = self.observable_crossed_nodes & (
            self._initial_alert_nodes - initial_native_tp_nodes
        )
        transient_crossed_late_alert = (
            self.observable_crossed_nodes - self._initial_alert_nodes
        )
        logger.info(f"[{self.detector_label} metadata TP减少评估]")
        logger.info(
            f"TP: {initial_native_tp}->{final_native_tp}, "
            f"净减少={tp_reduction_count}, TPReductionRate={tp_reduction_rate:.2%}"
        )
        logger.info(
            f"初始TP逃逸={escaped_count}/{self.initial_tp_count} "
            f"({native_tp_escape_rate:.2%}), "
            f"直接逃逸={len(direct_native_tp_escaped)}, "
            f"间接逃逸={len(indirect_native_tp_escaped)}"
        )
        direct_tp_asr_text = (
            f"{direct_tp_asr:.2%}" if direct_tp_asr is not None else "N/A"
        )
        direct_fp_rate_text = (
            f"{direct_fp_evasion_rate:.2%}"
            if direct_fp_evasion_rate is not None else "N/A"
        )
        logger.info(
            f"[TargetComposition] attacked_unique={len(self.attacked_nodes)}, "
            f"initial_metadata_TP={len(attacked_initial_tp_nodes)}, "
            f"initial_FP={len(attacked_initial_fp_nodes)}, "
            f"late_alert={len(attacked_late_alert_nodes)}"
        )
        logger.info(
            f"[DirectFinalEvasion] metadata_TP="
            f"{len(escaped_attempted_initial_tp_nodes)}/"
            f"{len(attempted_initial_tp_nodes)} "
            f"(DirectTP_ASR={direct_tp_asr_text}), initial_FP="
            f"{len(direct_initial_fp_escaped)}/{len(attacked_initial_fp_nodes)} "
            f"(DirectFP_EvasionRate={direct_fp_rate_text})"
        )
        logger.info(
            f"[TransientCrossing] metadata_TP={len(transient_crossed_initial_tp)}, "
            f"initial_FP={len(transient_crossed_initial_fp)}, "
            f"late_alert={len(transient_crossed_late_alert)}, "
            f"total={len(self.observable_crossed_nodes)}"
        )

        exact_tied_all = sum(
            float(score) == best_thres for score in self.initial_all_scores.values()
        )
        exact_tied_malicious = sum(
            self.initial_all_scores.get(node_id, float('-inf')) == best_thres
            for node_id in malicious
            if node_id < original_node_count
        )
        threshold_sensitivity = []
        offsets = [0.0]
        for epsilon in self.threshold_sensitivity_epsilons:
            epsilon = float(epsilon)
            if epsilon > 0:
                offsets.extend([-epsilon, epsilon])
        for offset in sorted(set(offsets)):
            audit_threshold = best_thres + offset
            initial_tp_at_threshold = sum(
                self.initial_all_scores.get(node_id, float('-inf')) >= audit_threshold
                for node_id in malicious
                if node_id < original_node_count
            )
            final_tp_at_threshold = sum(
                final_all_scores.get(node_id, float('-inf')) >= audit_threshold
                for node_id in malicious
                if node_id < original_node_count
            )
            reduction = initial_tp_at_threshold - final_tp_at_threshold
            threshold_sensitivity.append({
                'offset': float(offset),
                'threshold': float(audit_threshold),
                'initial_tp': int(initial_tp_at_threshold),
                'final_tp': int(final_tp_at_threshold),
                'signed_tp_reduction': int(reduction),
                'tp_reduction_rate': (
                    reduction / initial_tp_at_threshold
                    if initial_tp_at_threshold else None
                ),
            })
        logger.info(
            f"[ThresholdTieAudit] exact_tied_all={exact_tied_all}, "
            f"exact_tied_metadata_malicious={exact_tied_malicious}"
        )
        for audit in threshold_sensitivity:
            rate = audit['tp_reduction_rate']
            rate_text = f"{rate:.2%}" if rate is not None else "N/A"
            logger.info(
                f"[ThresholdSensitivity] offset={audit['offset']:+.6g}, "
                f"threshold={audit['threshold']:.12g}, "
                f"TP={audit['initial_tp']}->{audit['final_tp']}, "
                f"signed_reduction={audit['signed_tp_reduction']}, "
                f"rate={rate_text}"
            )

        indirect_tp_evasion_rate = (
            len(indirect_native_tp_escaped) / self.initial_tp_count
            if self.initial_tp_count else 0.0
        )
        attack_queries_excluding_baseline = max(attack_query_count - 1, 0)
        total_detector_queries = attack_query_count + len(clean_repeat_reports)
        tp_reduction_per_query = (
            tp_reduction_count / attack_queries_excluding_baseline
            if attack_queries_excluding_baseline else None
        )
        tp_reduction_per_action = (
            tp_reduction_count / self.total_actions
            if self.total_actions else None
        )
        total_edits = delta_e + delta_v
        primary_metrics = compute_primary_metrics(
            escaped_initial_tp=escaped_count,
            attacked_nodes=len(self.attacked_nodes),
            attacked_initial_tp=len(attempted_initial_tp_nodes),
            direct_escaped_initial_tp=len(escaped_attempted_initial_tp_nodes),
            delta_edges=delta_e,
            delta_nodes=delta_v,
            initial_edges=initial_edges,
            initial_nodes=initial_nodes,
        )
        direct_tp_asr = primary_metrics['direct_tp_asr']
        true_tp_evasion_amplification = primary_metrics['evasion_amplification']
        graph_modification_rate = primary_metrics['graph_modification_rate']
        average_attack_budget = primary_metrics['average_attack_budget']
        edits_per_reduced_tp = (
            total_edits / tp_reduction_count
            if tp_reduction_count > 0 else None
        )
        attack_elapsed_seconds = (
            time.time() - self._attack_started_at
            if self._attack_started_at is not None else 0.0
        )
        attack_elapsed_minutes = attack_elapsed_seconds / 60.0
        targeted_aab = (
            self.total_actions / len(escaped_attempted_initial_tp_nodes)
            if escaped_attempted_initial_tp_nodes else None
        )
        average_query_time_seconds = (
            attack_elapsed_seconds / attack_query_count
            if attack_query_count else None
        )
        policy_health = self._summarize_policy_health()

        def _metric_text(value, percent=False):
            if value is None:
                return "N/A"
            # GraphModifyRate is typically below 0.01%; two decimals would
            # misleadingly print a non-zero perturbation as 0.00%.
            return f"{value:.6%}" if percent else f"{value:.6f}"

        logger.info("[FinalMetrics] ===== 论文评估指标汇总 =====")
        logger.info(
            f"[FinalMetrics] strategy={self.action_policy}+{self.target_policy}, "
            f"seed={self.seed}, threshold={best_thres:.12g}"
        )
        logger.info("[FinalMetrics] ----- 论文四项主指标（固定口径）-----")
        logger.info(
            f"[FinalMetrics] DirectTP-ASR="
            f"{_metric_text(direct_tp_asr, percent=True)} "
            f"(escaped attempted initial TP="
            f"{len(escaped_attempted_initial_tp_nodes)}, "
            f"attempted initial TP={len(attempted_initial_tp_nodes)})"
        )
        logger.info(
            f"[FinalMetrics] TargetedAAB="
            f"{_metric_text(targeted_aab)} "
            f"(primitive operations={self.total_actions}, "
            f"successful attempted initial TP="
            f"{len(escaped_attempted_initial_tp_nodes)})"
        )
        logger.info(
            f"[FinalMetrics] AQT="
            f"{_metric_text(average_query_time_seconds)} seconds/query "
            f"(elapsed={attack_elapsed_seconds:.6f}s, queries={attack_query_count})"
        )
        logger.info(
            f"[FinalMetrics] EvasionAmplification="
            f"{_metric_text(true_tp_evasion_amplification)} "
            f"(all escaped initial TP={escaped_count}, "
            f"attacked unique nodes={len(self.attacked_nodes)})"
        )
        logger.info(
            f"[FinalMetrics] GraphModifyRate="
            f"{_metric_text(graph_modification_rate, percent=True)} "
            f"(deltaE+deltaV={delta_e}+{delta_v}={total_edits}, "
            f"E0+V0={initial_edges}+{initial_nodes}="
            f"{initial_edges + initial_nodes})"
        )
        logger.info(
            f"[FinalMetrics] AverageAttackBudget="
            f"{_metric_text(average_attack_budget)} "
            f"(deltaE+deltaV={total_edits}, "
            f"successfully evaded initial TP={escaped_count})"
        )
        logger.info("[FinalMetrics] ----- 辅助审计指标 -----")
        logger.info(
            f"[FinalMetrics] InitialTPEvasionRate="
            f"{_metric_text(native_tp_escape_rate, percent=True)} "
            f"({escaped_count}/{self.initial_tp_count}); "
            f"IndirectTPEvasionRate="
            f"{_metric_text(indirect_tp_evasion_rate, percent=True)} "
            f"({len(indirect_native_tp_escaped)}/{self.initial_tp_count})"
        )
        logger.info(
            f"[FinalMetrics] DirectAlertEvasionRate="
            f"{_metric_text(target_evasion_rate, percent=True)} "
            f"({direct_observable_count}/{len(eligible_attacked_nodes)}); "
            f"TPReductionPerQuery={_metric_text(tp_reduction_per_query)}, "
            f"TPReductionPerAction={_metric_text(tp_reduction_per_action)}, "
            f"EditsPerReducedTP={_metric_text(edits_per_reduced_tp)}"
        )
        logger.info(
            f"[FinalMetrics] attack_queries={attack_queries_excluding_baseline}, "
            f"total_detector_queries={total_detector_queries}, "
            f"reused_queries={self.reused_query_count}, "
            f"actions={self.total_actions}, edits={total_edits} "
            f"(deltaE={delta_e}, deltaV={delta_v}), "
            f"elapsed_seconds={attack_elapsed_seconds:.3f}"
        )
        logger.info(
            f"[FinalMetrics] TP={initial_native_tp}->{final_native_tp}, "
            f"FP={self.initial_metrics['fp']}->{final_metrics['fp']}, "
            f"Recall={self.initial_metrics['recall']:.6f}->{final_metrics['recall']:.6f}, "
            f"Precision={self.initial_metrics['precision']:.6f}->"
            f"{final_metrics['precision']:.6f}, "
            f"F1={self.initial_metrics['f1']:.6f}->{final_metrics['f1']:.6f}, "
            f"AUC={self.initial_metrics['auc']:.6f}->{final_metrics['auc']:.6f}"
        )
        logger.info(
            f"[FinalMetrics] target_composition: initialTP="
            f"{len(attacked_initial_tp_nodes)}, initialFP="
            f"{len(attacked_initial_fp_nodes)}, lateAlert="
            f"{len(attacked_late_alert_nodes)}; exactThresholdTies="
            f"{exact_tied_all}, maliciousTies={exact_tied_malicious}"
        )
        logger.info(
            f"[FinalMetrics] score_integrity: initialMissing="
            f"{score_integrity['initial_missing_original']}, finalMissing="
            f"{score_integrity['final_missing_original']}, initialNonfinite="
            f"{score_integrity['initial_nonfinite']}, finalNonfinite="
            f"{score_integrity['final_nonfinite']}"
        )
        logger.info("[PolicyHealth] ===== 策略非均匀性与奖励区分度 =====")
        policy_updates_current = self._policy_update_count()
        policy_updates_during_run = max(
            policy_updates_current - policy_updates_start, 0
        )
        logger.info(
            f"[PolicyHealth] action_policy={self.action_policy}, "
            f"selection_mode={policy_health['selection_mode']}, "
            f"selections={policy_health['selection_count']}, "
            f"updates_during_run={policy_updates_during_run}, "
            f"checkpoint_updates_total={policy_updates_current}"
        )
        logger.info(
            f"[PolicyHealth] NormalizedEntropy="
            f"mean={_metric_text(policy_health['mean_normalized_entropy'])}, "
            f"initial={_metric_text(policy_health['initial_normalized_entropy'])}, "
            f"final={_metric_text(policy_health['final_normalized_entropy'])}; "
            f"MeanMaxActionProbability="
            f"{_metric_text(policy_health['mean_max_action_probability'])}, "
            f"final={_metric_text(policy_health['final_max_action_probability'])}"
        )
        logger.info(
            f"[PolicyHealth] Top1Top2ProbabilityGap="
            f"mean={_metric_text(policy_health['mean_top1_top2_probability_gap'])}, "
            f"final={_metric_text(policy_health['final_top1_top2_probability_gap'])}; "
            f"KLToUniform="
            f"mean={_metric_text(policy_health['mean_kl_to_uniform'])}, "
            f"final={_metric_text(policy_health['final_kl_to_uniform'])}"
        )
        logger.info(
            f"[PolicyHealth] IndirectRewardNonzeroRate="
            f"{policy_health['indirect_reward_nonzero_rate']:.6%} "
            f"({policy_health['indirect_reward_nonzero_count']}/"
            f"{sum(record.get('kind') == 'attack_feedback' for record in self.query_records)}); "
            f"ActionHistogram={policy_health['action_histogram']}"
        )
        for action_stats in policy_health['per_action_reward']:
            logger.info(
                f"[PerActionReward] action_id={action_stats['action_id']}, "
                f"action={action_stats['action_name']}, "
                f"count={action_stats['count']}, "
                f"mean_reward={action_stats['mean_reward']:+.6f}, "
                f"direct_cross_rate="
                f"{action_stats['direct_cross_rate']:.6%}, "
                f"mean_indirect_reward="
                f"{action_stats['mean_indirect_reward']:+.6f}, "
                f"indirect_nonzero_rate="
                f"{action_stats['indirect_nonzero_rate']:.6%}"
            )
        final_normalized_entropy = policy_health[
            'final_normalized_entropy'
        ]
        if (
            self.action_policy == 'ppo_online'
            and policy_updates_during_run > 0
            and final_normalized_entropy is not None
            and final_normalized_entropy > 0.98
        ):
            logger.warning(
                "[PolicyHealthWarning] PPO更新后NormalizedEntropy仍高于0.98，"
                "策略仍接近当前合法动作集合上的均匀分布"
            )

        attack_success = tp_reduction_count > 0
        if attack_success:
            logger.info(
                f"最终判定: {self.detector_label} metadata TP减少，攻击目标达成"
            )
        else:
            logger.info(
                f"最终判定: {self.detector_label} metadata TP未减少，攻击目标未达成"
            )

        if self.stop_reason is None:
            self.stop_reason = 'max_rounds_completed'
        logger.info(
            f"停止原因: {self.stop_reason}; 攻击查询={attack_query_count}, "
            f"最终评估查询=0（复用最近观测）, "
            f"累计省去查询={self.reused_query_count}"
        )
        
        checkpoint_updates_end = self.ppo_agent.training_stats.get('total_updates', 0)
        policy_updates_end = self._policy_update_count()
        checkpoint_saved = False
        if self.action_policy == 'ppo_online':
            checkpoint_path = self.checkpoint_path
            self.ppo_agent.save_checkpoint(checkpoint_path)
            checkpoint_saved = True
            logger.info(f"PPO模型已保存到: {checkpoint_path}")
        elif self.action_policy == 'linucb':
            checkpoint_path = self.checkpoint_path
            self.linucb_policy.save_checkpoint(checkpoint_path)
            checkpoint_saved = True
            logger.info(f"LinUCB状态已保存到: {checkpoint_path}")
        elif self.action_policy == 'double_dqn':
            checkpoint_path = self.checkpoint_path
            self.double_dqn_policy.save_checkpoint(checkpoint_path)
            checkpoint_saved = True
            logger.info(f"Double DQN模型已保存到: {checkpoint_path}")
        else:
            logger.info(
                f"[PolicyNoCheckpointWrite] mode={self.action_policy}; "
                "本次实验不会覆盖PPO检查点"
            )
        
        return {
            # asr aliases are retained for callers; they now mean the escape
            # rate of initial MAGIC metadata TP nodes.
            'asr': native_tp_escape_rate,
            'asr_final': native_tp_escape_rate,
            'asr_numerator': escaped_count,
            'asr_denominator': self.initial_tp_count,
            'escaped_nodes': escaped_count,
            'total_tp_nodes': self.initial_tp_count,
            'initial_native_tp': initial_native_tp,
            'final_native_tp': final_native_tp,
            'initial_metrics': dict(self.initial_metrics),
            'final_metrics': dict(final_metrics),
            'tp_reduction_count': tp_reduction_count,
            'tp_reduction_rate': tp_reduction_rate,
            'native_tp_escape_rate': native_tp_escape_rate,
            'direct_native_tp_escaped': len(direct_native_tp_escaped),
            'indirect_native_tp_escaped': len(indirect_native_tp_escaped),
            'attacked_initial_tp_count': len(attacked_initial_tp_nodes),
            'attempted_initial_tp_count': len(attempted_initial_tp_nodes),
            'attacked_initial_fp_count': len(attacked_initial_fp_nodes),
            'attacked_late_alert_count': len(attacked_late_alert_nodes),
            'direct_tp_asr': direct_tp_asr,
            'targeted_asr': direct_tp_asr,
            'targeted_asr_numerator': len(escaped_attempted_initial_tp_nodes),
            'targeted_asr_denominator': len(attempted_initial_tp_nodes),
            'targeted_aab_primitive_operations': targeted_aab,
            'average_query_time_seconds': average_query_time_seconds,
            'direct_initial_fp_escaped': len(direct_initial_fp_escaped),
            'direct_fp_evasion_rate': direct_fp_evasion_rate,
            'indirect_tp_evasion_rate': indirect_tp_evasion_rate,
            'true_tp_evasion_amplification': true_tp_evasion_amplification,
            'evasion_amplification': true_tp_evasion_amplification,
            'graph_modification_rate': graph_modification_rate,
            'average_attack_budget': average_attack_budget,
            'attack_queries_excluding_baseline': attack_queries_excluding_baseline,
            'total_detector_queries': total_detector_queries,
            'tp_reduction_per_query': tp_reduction_per_query,
            'tp_reduction_per_action': tp_reduction_per_action,
            'edits_per_reduced_tp': edits_per_reduced_tp,
            'total_edits': total_edits,
            'attack_elapsed_seconds': attack_elapsed_seconds,
            'attack_elapsed_minutes': attack_elapsed_minutes,
            'transient_crossed_initial_tp': len(transient_crossed_initial_tp),
            'transient_crossed_initial_fp': len(transient_crossed_initial_fp),
            'transient_crossed_late_alert': len(transient_crossed_late_alert),
            'attack_success': attack_success,
            'stop_reason': self.stop_reason,
            'attack_query_count': attack_query_count,
            'evaluation_query_count': 0,
            'reused_query_count': self.reused_query_count,
            'attacked_unique_nodes': len(self.attacked_nodes),
            'attempted_unique_nodes': len(self.attempted_target_nodes),
            'eligible_attacked_alerts': len(eligible_attacked_nodes),
            'direct_observable_escaped': direct_observable_count,
            'indirect_observable_escaped': len(indirect_observable_escaped_nodes),
            'total_observable_escaped': total_observable_count,
            'target_evasion_rate': target_evasion_rate,
            'observable_evasion_amplification': observable_evasion_amplification,
            'indirect_evasion_factor': indirect_evasion_factor,
            'total_actions': self.total_actions,
            'macro_action_length': self.config.MACRO_ACTION_LENGTH,
            'macro_transition_count': sum(
                record.get('kind') == 'attack_feedback'
                for record in self.query_records
            ),
            'ppo_rollout_size': self.config.PPO_ROLLOUT_SIZE,
            'ppo_selection_mode': self.ppo_selection_mode,
            'ppo_policy_schema_version': self.ppo_agent.policy_schema_version,
            'policy_health': policy_health,
            'avg_steps_per_direct_evasion': avg_steps_per_direct,
            'avg_steps_per_total_evasion': avg_steps_per_total,
            'observable_crossed_nodes': len(self.observable_crossed_nodes),
            'action_policy': self.action_policy,
            'target_policy': self.target_policy,
            'target_margin_fraction': self.target_margin_fraction,
            'candidate_policy': self.candidate_policy,
            'experiment_variant': self.experiment_variant,
            'orthrus_full_aware': self.orthrus_full_aware,
            'orthrus_candidate_probe_size': (
                self.orthrus_candidate_probe_size
                if self.orthrus_full_aware else None
            ),
            'orthrus_full_aware_stats': dict(
                self.orthrus_full_aware_stats
            ),
            'protected_initial_tp_nodes': sorted(
                self._protected_initial_tp_nodes
            ),
            'action_schema_dataset': self.action_schema.dataset,
            'action_schema_version': self.action_schema.version,
            'action_dim': self.config.ACTION_DIM,
            'state_dim': self.config.STATE_DIM,
            'base_state_dim': StateEncoder.STATE_DIM,
            'node_type_slots': StateEncoder.NODE_TYPE_DIM,
            'action_definitions': [
                {
                    'action_id': spec.action_id,
                    'name': spec.name,
                    'edge_type': spec.edge_type,
                    'edge_name': spec.edge_name,
                    'valid_pairs': [list(pair) for pair in spec.valid_pairs],
                }
                for spec in self.action_schema.actions
            ],
            'reward_mode': self.reward_mode,
            'reward_config': dict(self.reward_config),
            'greedy_stats': self.greedy_policy.snapshot(),
            'linucb_stats': (
                self.linucb_policy.snapshot()
                if self.linucb_policy is not None else None
            ),
            'double_dqn_stats': (
                self.double_dqn_policy.snapshot()
                if self.double_dqn_policy is not None else None
            ),
            'benign_mimicry_stats': (
                self.benign_mimicry_policy.snapshot()
                if self.benign_mimicry_policy is not None else None
            ),
            'seed': self.seed,
            'checkpoint_updates_start': checkpoint_updates_start,
            'checkpoint_updates_end': checkpoint_updates_end,
            'policy_updates_start': policy_updates_start,
            'policy_updates_end': policy_updates_end,
            'checkpoint_saved': checkpoint_saved,
            'clean_repeat_reports': clean_repeat_reports,
            'score_integrity': score_integrity,
            'threshold_exact_tied_all': exact_tied_all,
            'threshold_exact_tied_metadata_malicious': exact_tied_malicious,
            'threshold_sensitivity': threshold_sensitivity,
            'escaped_native_tp_nodes': sorted(escaped_native_tp_nodes),
            'direct_native_tp_nodes': sorted(direct_native_tp_escaped),
            'indirect_native_tp_nodes': sorted(indirect_native_tp_escaped),
            'attacked_initial_tp_nodes': sorted(attacked_initial_tp_nodes),
            'attempted_initial_tp_nodes': sorted(attempted_initial_tp_nodes),
            'escaped_attempted_initial_tp_nodes': sorted(
                escaped_attempted_initial_tp_nodes
            ),
            'attacked_initial_fp_nodes': sorted(attacked_initial_fp_nodes),
            'attacked_late_alert_nodes': sorted(attacked_late_alert_nodes),
            'threshold': best_thres,
            'threshold_source': getattr(self.detector, 'threshold_source', 'detector_native'),
            'native_evaluation_scope': (
                'trace_original_multigraph'
                if self.dataset_name == 'trace' else 'attacked_test_graph'
            ),
            'native_evaluation_size': int(
                self.initial_metrics['tp'] + self.initial_metrics['fp']
                + self.initial_metrics['tn'] + self.initial_metrics['fn']
            ),
            'local_metadata_malicious_count': len(malicious),
            # Kept for older callers; the paper AAB now consistently means
            # edits per successfully evaded initial metadata TP.
            'aab': (
                average_attack_budget
                if average_attack_budget is not None else float('inf')
            ),
            'observable_aab': observable_aab,
            'delta_nodes': delta_nodes,
            'delta_edges': delta_edges
        }


# ============================================================================
# 主函数
# ============================================================================
import argparse


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MAGIC对抗攻击框架')
    parser.add_argument('--dataset', type=str, default=None,
                        help='数据集名称: cadets, theia, trace, streamspot, wget')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-K异常节点数量')
    parser.add_argument('--load_checkpoint', action='store_true',
                        help='是否加载预训练模型')
    parser.add_argument('--test-index', type=int, default=None,
                        help='测试图索引；默认使用最后一张图')
    parser.add_argument('--max-rounds', type=int, default=None)
    parser.add_argument('--threshold-source',
                        choices=('native', 'fixed'),
                        default='native')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='PPO检查点路径；默认按检测器和数据集命名')
    parser.add_argument('--warmup-updates', type=int, default=None,
                        help='使用均匀探索的PPO更新次数；默认0，立即使用策略')
    args = parser.parse_args()
    logger.info("MAGIC对抗攻击框架 - Directed Top-K模式")

    # 【关键修复】从配置读取参数
    config = DirectedAttackConfig()
    top_k = args.top_k if args.top_k is not None else config.TOP_K
    dataset_name = args.dataset if args.dataset is not None else config.DATASET_NAME  # 【新增】从命令行或配置读取数据集
    load_checkpoint = args.load_checkpoint  # 【新增】使用命令行参数
    checkpoint_path = args.checkpoint or Path(
        f'ppo_checkpoint_magic_{dataset_name}_directed_topk.pth'
    )

    logger.info(f"攻击配置: 数据集={dataset_name}, Top-K={top_k}, 加载预训练={load_checkpoint}")

    detector = DLDetector(
        dataset_name=dataset_name,
        top_k=top_k,
        threshold_source=args.threshold_source,
        threshold=args.threshold,
    )
    if args.threshold_source == 'fixed' and args.threshold is None:
        raise ValueError("--threshold-source fixed需要--threshold")

    # 初始化攻击智能体
    agent = DirectedTopKAttackAgent(
        dataset_name=dataset_name,  # 【修改】使用配置的数据集参数
        top_k=top_k,
        detector=detector,
    )
    if args.max_rounds is not None:
        agent.config.MAX_ROUNDS = args.max_rounds
    if args.warmup_updates is not None:
        if args.warmup_updates < 0:
            raise ValueError("--warmup-updates必须>=0")
        agent.config.WARMUP_UPDATES = args.warmup_updates
        agent.ppo_agent.config.WARMUP_UPDATES = args.warmup_updates
    agent.checkpoint_path = str(checkpoint_path)

    # 【新增】加载预训练模型(如果指定)
    if load_checkpoint and checkpoint_path.exists():
        logger.info(f"[PPO] 正在加载预训练模型: {checkpoint_path}")
        agent.ppo_agent.load_checkpoint(str(checkpoint_path))
    elif load_checkpoint:
        raise FileNotFoundError(f"PPO检查点文件不存在: {checkpoint_path}")

    # 加载测试图
    logger.info("加载测试图...")
    
    # 【优化】使用检测器缓存的 metadata，避免重复 load_metadata
    n_test = agent.detector.metadata['n_test']
    
    test_index = n_test - 1 if args.test_index is None else args.test_index
    if not 0 <= test_index < n_test:
        raise ValueError(f"--test-index必须在[0, {n_test - 1}]范围内")
    agent.detector.set_evaluation_context(test_index)
    logger.info(f"数据集有{n_test}个测试图，攻击索引={test_index}")
    test_graph = load_entity_level_dataset(dataset_name, 'test', test_index)
    logger.info(
        f"测试图(索引={test_index}): 节点={test_graph.number_of_nodes()}, "
        f"边={test_graph.number_of_edges()}"
    )

    # 执行攻击
    start_time = time.time()
    attack_result = agent.attack(test_graph)
    elapsed_time = time.time() - start_time

    logger.info(
        f"最终结果: 耗时={elapsed_time:.2f}s, "
        f"MAGIC TP={attack_result['initial_native_tp']}->{attack_result['final_native_tp']}, "
        f"净减少={attack_result['tp_reduction_count']}, "
        f"TPReductionRate={attack_result['tp_reduction_rate']:.2%}, "
        f"初始TP逃逸率={attack_result['native_tp_escape_rate']:.2%}, "
        f"图修改率={attack_result['graph_modification_rate']:.4f}, "
        f"AAB={attack_result['aab']:.4f}"
    )


if __name__ == '__main__':
    main()
