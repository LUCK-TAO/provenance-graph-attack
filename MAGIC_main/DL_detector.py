"""
DL Detector (PIDS) - 黑盒检测器封装
基于MAGIC模型的溯源图异常检测器
提供黑盒查询接口：输入修改后的溯源图，返回top-K异常节点及其分数
"""

import os
import logging
import pickle as pkl
from bisect import bisect_right
import torch
import numpy as np

logger = logging.getLogger('attack.DL_detector')

# 【性能优化】尝试使用cuML（GPU加速的KNN），如果不可用则回退到sklearn
try:
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    USE_CUML = True
except ImportError:
    from sklearn.neighbors import NearestNeighbors
    USE_CUML = False

from sklearn.metrics import roc_auc_score, precision_recall_curve
from utils.loaddata import load_metadata, load_entity_level_dataset
from model.autoencoder import build_model
from Detector.Detector_Interface import BaseDetector
from native_evaluation import (
    build_trace_evaluation_layout,
    classification_metrics as native_classification_metrics,
    localize_global_nodes,
    select_magic_threshold,
    trace_local_to_evaluation_positions,
)
import warnings

warnings.filterwarnings('ignore')


class DLDetector(BaseDetector):
    """
    深度学习检测器（黑盒封装）
    
    功能：
    1. 加载预训练的MAGIC模型权重
    2. 提取图嵌入表示
    3. 基于KNN计算节点异常分数
    4. 返回异常分数最高的top-K节点
    
    使用场景：
    - 攻击智能体的黑盒查询接口
    - 不暴露模型内部结构和训练数据
    """
    
    def __init__(self, dataset_name, device='cuda:0', top_k=30,
                 threshold_source='native', threshold=None,
                 native_magic_compat=False):
        """
        初始化检测器
        
        Args:
            dataset_name: 数据集名称 ('cadets', 'theia', 'trace')
            device: 设备 ('cuda:0' 或 'cpu')
            top_k: 返回异常分数最高的节点数量
        """
        self.dataset_name = dataset_name
        self.detector_family = 'distance'
        self.score_direction = 'higher_is_more_anomalous'
        # 【修复】正确处理设备选择逻辑
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')
            logger.warning("CUDA不可用，使用CPU")
        self.top_k = top_k
        
        # 模型和KNN相关参数
        self.model = None
        self.nbrs = None
        self.mean_distance = None
        self.x_train_mean = None
        self.x_train_std = None
        self.standardization_epsilon = 1e-6
        self.knn_score_space = "dl_v1"
        self.native_magic_compat = bool(native_magic_compat)
        
        # 【新增】TP追踪相关
        if threshold_source not in {'native', 'fixed'}:
            raise ValueError("threshold_source must be one of: native, fixed")
        if threshold_source == 'fixed' and threshold is None:
            raise ValueError("threshold_source='fixed' requires a numeric threshold")
        self.threshold_source = threshold_source
        self.configured_threshold = None if threshold is None else float(threshold)
        self.best_threshold = self.configured_threshold  # 攻击前固定，攻击过程中不再更新
        self.initial_tp_nodes = None  # 初始TP节点集合
        self.escaped_nodes_history = set()  # 【关键】历史逃逸节点集合(只增不减)（攻击全程不变）
        
        # 【性能优化】Embedding缓存 - 避免重复计算同一张图的嵌入
        self._embedding_cache = {}  # key: graph hash, value: embedding tensor
        self.test_index = 0
        self._native_evaluation_cache = None
        self._native_evaluation_scores = None
        self._native_evaluation_labels = None
        self._local_to_evaluation_position = {}
        self.last_baseline_cache_hit = False
        
        # 加载模型和KNN基准
        self._load_model()
        self._load_knn_baseline()
        if self.native_magic_compat:
            self.enable_native_magic_compatibility()
        
        # 【优化】缓存 metadata，避免每次 query/ASR 计算时重复加载
        self.metadata = load_metadata(self.dataset_name)
        self.native_malicious_nodes, _ = self.metadata['malicious']
        self.global_malicious_nodes = list(self.native_malicious_nodes)
        self.native_malicious_set = set(self.native_malicious_nodes)
        # Backward-compatible aliases for native MAGIC reproduction only.
        self.malicious_nodes = self.native_malicious_nodes
        self.malicious_set = self.native_malicious_set
        self.initial_alert_nodes = set()
        
        logger.info(f"[DL Detector] 初始化完成 - 数据集: {dataset_name}, Top-K: {top_k}")
        logger.info(f"[DL Detector] 阈值来源: {self.threshold_source}")
        
        # 【新增】初始化时计算并打印原始MAGIC的性能指标（作为攻击基准）
        # Evaluation is deferred until initialize_tp_baseline(), after the
        # caller has selected the actual test graph (important for TRACE).

    def enable_native_magic_compatibility(self):
        """Reconstruct TRACE's original MAGIC KNN score space in memory.

        ``distance_save_trace_dl.pkl`` stores a fitted sklearn KNN model whose
        training embeddings were divided by ``std + 1e-6``.  MAGIC's original
        evaluator divided by ``std`` and used the mean distance stored in
        ``distance_save_trace.pkl``.  Re-scaling the fitted training matrix and
        using zero epsilon restores that definition without querying any test
        graph or overwriting either cache file.
        """
        if self.dataset_name != "trace":
            raise ValueError(
                "native_magic_compat is currently implemented only for TRACE"
            )
        if self.knn_score_space == "magic_native_trace_v1":
            self.standardization_epsilon = 0.0
            logger.info(
                "[DL Detector] TRACE原生MAGIC KNN缓存已经对齐，无需再次转换"
            )
            return
        if self.x_train_std is None or self.nbrs is None:
            raise RuntimeError("TRACE KNN baseline is not initialized")
        fit_x = getattr(self.nbrs, "_fit_X", None)
        if fit_x is None:
            raise RuntimeError(
                "TRACE native compatibility requires the sklearn KNN cache "
                "with a fitted _fit_X matrix; regenerate the _dl cache with "
                "the CPU sklearn backend"
            )
        std = np.asarray(self.x_train_std, dtype=np.float64)
        if np.any(std == 0):
            raise RuntimeError(
                "Original MAGIC standardization is undefined because a "
                "TRACE training embedding dimension has zero variance"
            )
        # Convert (x-mean)/(std+1e-6) back to (x-mean)/std.  Mutating this
        # process-local unpickled matrix does not modify the on-disk _dl cache.
        scale = ((std + 1e-6) / std).astype(fit_x.dtype, copy=False)
        fit_x *= scale.reshape(1, -1)

        # sklearn's high-dimensional auto backend is normally brute force and
        # reads _fit_X directly.  Refit defensively if a tree backend was used.
        if getattr(self.nbrs, "_fit_method", "brute") != "brute":
            from sklearn.neighbors import NearestNeighbors
            rebuilt = NearestNeighbors(n_neighbors=10, n_jobs=-1)
            rebuilt.fit(fit_x)
            self.nbrs = rebuilt

        original_cache = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "MAGIC_main", "eval_result", "distance_save_trace.pkl",
        )
        if not os.path.exists(original_cache):
            raise FileNotFoundError(
                f"Original MAGIC TRACE distance cache is missing: {original_cache}"
            )
        with open(original_cache, "rb") as handle:
            saved = pkl.load(handle)
        if not isinstance(saved, (list, tuple)) or len(saved) != 2:
            raise ValueError(
                "distance_save_trace.pkl is not the original MAGIC cache format"
            )
        self.mean_distance = float(saved[0])
        self.standardization_epsilon = 0.0
        self.knn_score_space = "magic_native_trace_v1"
        logger.info(
            "[DL Detector] TRACE原生MAGIC KNN兼容模式已启用 - "
            f"epsilon=0, mean_distance={self.mean_distance:.12g}"
        )

    def persist_native_magic_knn_baseline(self):
        """Atomically replace TRACE's _dl cache after alignment has passed."""
        if self.dataset_name != "trace":
            raise ValueError("native TRACE KNN persistence is TRACE-only")
        if self.knn_score_space != "magic_native_trace_v1":
            raise RuntimeError("refusing to persist an unverified non-native KNN")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(
            root, "MAGIC_main", "eval_result", "distance_save_trace_dl.pkl"
        )
        temp_path = save_path + ".native.tmp"
        payload = {
            "version": 2,
            "dataset": "trace",
            "score_space": self.knn_score_space,
            "standardization_epsilon": 0.0,
            "mean_distance": float(self.mean_distance),
            "x_train_mean": self.x_train_mean,
            "x_train_std": self.x_train_std,
            "nbrs": self.nbrs,
        }
        try:
            with open(temp_path, "wb") as handle:
                pkl.dump(payload, handle, protocol=pkl.HIGHEST_PROTOCOL)
            os.replace(temp_path, save_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        logger.info(f"[DL Detector] TRACE原生MAGIC KNN缓存已原子覆盖: {save_path}")
        return save_path

    def rebuild_native_magic_knn_from_training(self):
        """Rebuild TRACE KNN from the four training graphs as MAGIC did.

        This is the definitive fallback when the fitted matrix in the newer
        ``_dl.pkl`` cache is stale.  It performs no test-graph query and leaves
        all on-disk caches untouched until an external alignment check passes.
        """
        if self.dataset_name != "trace":
            raise ValueError("native TRACE KNN rebuilding is TRACE-only")
        from sklearn.neighbors import NearestNeighbors

        metadata = load_metadata("trace")
        embeddings = []
        logger.info(
            "[DL Detector] 按原始MAGIC逻辑重建TRACE训练KNN: "
            f"train_graphs={int(metadata['n_train'])}"
        )
        with torch.no_grad():
            for index in range(int(metadata["n_train"])):
                graph = load_entity_level_dataset("trace", "train", index)
                graph = graph.to(self.device)
                embeddings.append(self.model.embed(graph).cpu().numpy())
                logger.info(
                    f"[DL Detector] TRACE train graph={index}, "
                    f"nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}"
                )
                del graph
        x_train = np.concatenate(embeddings, axis=0)
        del embeddings
        self.x_train_mean = x_train.mean(axis=0)
        self.x_train_std = x_train.std(axis=0)
        if np.any(self.x_train_std == 0):
            raise RuntimeError(
                "Original MAGIC standardization is undefined because a "
                "TRACE training embedding dimension has zero variance"
            )
        x_train = (x_train - self.x_train_mean) / self.x_train_std
        rebuilt = NearestNeighbors(n_neighbors=10, n_jobs=-1)
        rebuilt.fit(x_train)
        self.nbrs = rebuilt

        original_cache = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "MAGIC_main", "eval_result", "distance_save_trace.pkl",
        )
        with open(original_cache, "rb") as handle:
            saved = pkl.load(handle)
        self.mean_distance = float(saved[0])
        self.standardization_epsilon = 0.0
        self.knn_score_space = "magic_native_trace_v1"
        self._embedding_cache.clear()
        logger.info(
            "[DL Detector] TRACE原生训练KNN重建完成 - "
            f"train_nodes={x_train.shape[0]}, dims={x_train.shape[1]}, "
            f"mean_distance={self.mean_distance:.12g}"
        )

    def _native_evaluation_cache_path(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(
            root, "MAGIC_main", "eval_result",
            f"native_evaluation_{self.dataset_name}_dl.pkl",
        )

    def build_native_evaluation_cache(self, *, force=False):
        """Precompute MAGIC's native multi-graph evaluation baseline.

        TRACE's metadata IDs index the concatenation of five test graphs.  The
        original evaluator retains malicious nodes from the early partitions
        and every node from the final partition.  This cache is detector-side
        calibration and is intentionally built before an attack experiment so
        its graph queries are not charged to the attacker's black-box budget.
        """
        if self.dataset_name != "trace":
            raise ValueError("native multi-graph calibration is only required for TRACE")
        cache_path = self._native_evaluation_cache_path()
        if os.path.exists(cache_path) and not force:
            with open(cache_path, "rb") as handle:
                return pkl.load(handle)

        graph_sizes = []
        early_scores = []
        final_scores = None
        global_offset = 0
        malicious_global = sorted(int(node) for node in self.global_malicious_nodes)
        n_test = int(self.metadata["n_test"])
        logger.info(
            f"[DL Detector] 正在构建TRACE原生MAGIC评估基准: test_graphs={n_test}"
        )
        for index in range(n_test):
            graph = load_entity_level_dataset(
                self.dataset_name, "test", index
            )
            size = int(graph.number_of_nodes())
            graph_sizes.append(size)
            if index == n_test - 1:
                # MAGIC retains every node from the final TRACE partition.
                result = self.query(graph)
                final_scores = np.fromiter(
                    (result["all_scores"][node] for node in range(size)),
                    dtype=np.float64,
                    count=size,
                )
                selected_count = size
            else:
                # Benign nodes in early partitions are excluded by MAGIC's
                # evaluator.  Their KNN distances therefore need not be
                # computed; the GNN embedding still sees the complete graph.
                local_malicious = [
                    node - global_offset
                    for node in malicious_global
                    if global_offset <= node < global_offset + size
                ]
                selected_scores = self.query_selected_scores(
                    graph, local_malicious
                )
                early_scores.extend(
                    selected_scores[node] for node in local_malicious
                )
                result = None
                selected_count = len(local_malicious)
            logger.info(
                f"[DL Detector] TRACE calibration graph={index}, "
                f"nodes={size}, edges={graph.number_of_edges()}, "
                f"knn_scored_nodes={selected_count}"
            )
            # Calibration graphs are used once. Retaining all five embedding
            # matrices would consume roughly a gigabyte of host memory.
            self._embedding_cache.clear()
            del graph, result
            global_offset += size

        layout = build_trace_evaluation_layout(
            self.global_malicious_nodes, graph_sizes
        )
        if len(early_scores) != len(layout["early_malicious_global"]):
            raise RuntimeError(
                "TRACE early malicious score count mismatch: "
                f"scores={len(early_scores)}, "
                f"labels={len(layout['early_malicious_global'])}"
            )
        if final_scores is None:
            raise RuntimeError("TRACE final graph scores were not generated")
        evaluation_scores = np.concatenate([
            np.asarray(early_scores, dtype=np.float64), final_scores
        ])
        labels = np.asarray(layout["labels"], dtype=np.int8)
        if len(evaluation_scores) != len(labels):
            raise RuntimeError(
                "TRACE native evaluation score/label length mismatch: "
                f"scores={len(evaluation_scores)}, labels={len(labels)}"
            )
        threshold = select_magic_threshold(
            self.dataset_name, labels, evaluation_scores
        )
        metrics = native_classification_metrics(
            labels, evaluation_scores, threshold
        )
        cache = {
            "version": 1,
            "dataset": self.dataset_name,
            "graph_sizes": graph_sizes,
            "evaluation_scores": evaluation_scores,
            "threshold": float(threshold),
            "initial_metrics": metrics,
            "evaluation_size": int(len(labels)),
            "malicious_count": int(labels.sum()),
            "selection_rule": (
                "all malicious nodes from early partitions plus all nodes "
                "from the final partition"
            ),
            "knn_score_space": self.knn_score_space,
            "score_transform": "log1p(distance/mean_distance)",
        }
        with open(cache_path, "wb") as handle:
            pkl.dump(cache, handle, protocol=pkl.HIGHEST_PROTOCOL)
        logger.info(
            "[DL Detector] TRACE原生评估基准已保存: "
            f"path={cache_path}, threshold={threshold:.12g}, "
            f"F1={metrics['f1']:.6f}, TP={metrics['tp']}, FP={metrics['fp']}"
        )
        return cache

    def _load_trace_evaluation_context(self):
        cache_path = self._native_evaluation_cache_path()
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                "TRACE native MAGIC evaluation cache is missing. Run: "
                "python MAGIC_main/calibrate_native_magic.py --dataset trace"
            )
        with open(cache_path, "rb") as handle:
            cache = pkl.load(handle)
        if cache.get("dataset") != "trace" or int(cache.get("version", 0)) != 1:
            raise ValueError(f"invalid TRACE native evaluation cache: {cache_path}")
        graph_sizes = [int(value) for value in cache["graph_sizes"]]
        if len(graph_sizes) != int(self.metadata["n_test"]):
            raise ValueError("TRACE native evaluation cache graph count is stale")
        layout = build_trace_evaluation_layout(
            self.global_malicious_nodes, graph_sizes
        )
        scores = np.asarray(cache["evaluation_scores"], dtype=np.float64)
        labels = np.asarray(layout["labels"], dtype=np.int8)
        if len(scores) != len(labels):
            raise ValueError("TRACE native evaluation cache length is stale")
        self._native_evaluation_cache = cache
        self._native_evaluation_scores = scores
        self._native_evaluation_labels = labels
        self._local_to_evaluation_position = (
            trace_local_to_evaluation_positions(layout, self.test_index)
        )
        local_malicious = localize_global_nodes(
            self.global_malicious_nodes, graph_sizes, self.test_index
        )
        self.malicious_nodes = local_malicious
        self.malicious_set = set(local_malicious)
        self.native_malicious_set = set(local_malicious)
        logger.info(
            "[DL Detector] TRACE原生评估上下文已加载: "
            f"test_index={self.test_index}, local_nodes={graph_sizes[self.test_index]}, "
            f"local_malicious={len(local_malicious)}, "
            f"global_eval_nodes={len(labels)}, global_malicious={int(labels.sum())}"
        )

    def evaluate_native_scores(self, local_scores, *, threshold=None):
        """Evaluate one attacked graph under MAGIC's unchanged native scope."""
        if self.dataset_name != "trace":
            node_count = len(local_scores)
            labels = np.zeros(node_count, dtype=np.int8)
            valid = [node for node in self.malicious_nodes if node < node_count]
            if valid:
                labels[valid] = 1
            scores = np.asarray([
                local_scores.get(node, float("-inf"))
                for node in range(node_count)
            ])
        else:
            if self._native_evaluation_scores is None:
                self._load_trace_evaluation_context()
            scores = self._native_evaluation_scores.copy()
            labels = self._native_evaluation_labels
            for local_node, evaluation_position in (
                self._local_to_evaluation_position.items()
            ):
                if local_node in local_scores:
                    scores[evaluation_position] = float(local_scores[local_node])
        operating_threshold = (
            self.best_threshold if threshold is None else float(threshold)
        )
        return native_classification_metrics(
            labels, scores, operating_threshold
        )

    def _trace_clean_scores_from_native_cache(self, graph):
        """Return cached clean test4 scores when the graph is unmodified.

        TRACE's native cache contains every node score for the final test
        partition.  Reusing it avoids repeating a roughly 16-minute clean
        detector computation in every strategy run.  This is a physical cache
        hit only; callers still count the observation as one logical black-box
        query for fair query-budget accounting.
        """
        if (
            self.dataset_name != "trace"
            or self.threshold_source != "native"
            or self._native_evaluation_cache is None
            or self._native_evaluation_scores is None
        ):
            return None
        graph_sizes = [
            int(value) for value in self._native_evaluation_cache["graph_sizes"]
        ]
        if self.test_index != len(graph_sizes) - 1:
            return None
        expected_nodes = graph_sizes[self.test_index]
        if int(graph.number_of_nodes()) != expected_nodes:
            logger.warning(
                "[BaselineCacheMiss] TRACE graph node count differs from "
                f"calibration: current={graph.number_of_nodes()}, "
                f"expected={expected_nodes}"
            )
            return None
        prefix = len(self._native_evaluation_scores) - expected_nodes
        cached = np.asarray(
            self._native_evaluation_scores[prefix:], dtype=np.float64
        )
        if len(cached) != expected_nodes or not np.all(np.isfinite(cached)):
            logger.warning("[BaselineCacheMiss] TRACE cached test4 scores are invalid")
            return None
        logger.info(
            "[BaselineCacheHit] TRACE test4干净分数直接复用原生评估缓存 - "
            f"nodes={expected_nodes}, physical_queries_saved=1, "
            "logical_queries_charged=1"
        )
        return {
            "all_scores": {
                node: float(score) for node, score in enumerate(cached)
            },
            "threshold": self.best_threshold,
            "metadata": {
                "model": "MAGIC",
                "dataset": "trace",
                "source": "native_evaluation_cache",
            },
        }

    def _load_model(self):
        """加载预训练的MAGIC模型"""
        logger.info("[DL Detector] 正在加载预训练模型...")
        
        # 根据数据集设置模型参数
        if self.dataset_name in ['streamspot', 'wget']:
            num_hidden = 256
            num_layers = 4
        else:  # cadets, theia, trace
            num_hidden = 64
            num_layers = 3
        
        # 构建模型参数
        class Args:
            def __init__(self):
                self.num_hidden = num_hidden
                self.num_layers = num_layers
                self.n_dim = 0
                self.e_dim = 0
                self.pooling = 'mean'
                # 添加build_model所需的参数
                self.negative_slope = 0.2  # LeakyReLU的负斜率
                self.mask_rate = 0.5       # 掩码率
                self.alpha_l = 3           # sce损失的幂指数
                self.loss_fn = 'sce'       # 损失函数类型
        
        args = Args()
        
        # 加载metadata获取特征维度
        metadata = load_metadata(self.dataset_name)
        args.n_dim = metadata['node_feature_dim']
        args.e_dim = metadata['edge_feature_dim']
        
        # 构建模型并加载权重
        self.model = build_model(args)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = os.path.join(root, 'MAGIC_main', 'checkpoints', f"checkpoint-{self.dataset_name}.pt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"预训练权重文件不存在: {checkpoint_path}")
        
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"[DL Detector] 模型加载成功 - 隐藏维度: {num_hidden}, 层数: {num_layers}")
        logger.info(f"[DL Detector] 使用设备: {self.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    def _load_knn_baseline(self):
        """
        加载KNN基准距离
        这些基准是在正常数据上计算的，用于异常评分
        """
        logger.info("[DL Detector] 正在加载KNN基准...")
        
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dict_path = os.path.join(root, 'MAGIC_main', 'eval_result', f'distance_save_{self.dataset_name}_dl.pkl')
        
        if not os.path.exists(save_dict_path):
            # 如果KNN基准文件不存在，需要先生成
            logger.info("[DL Detector] KNN基准文件不存在，正在生成...")
            self._generate_knn_baseline()
        else:
            # 加载已有的KNN基准
            with open(save_dict_path, 'rb') as f:
                saved_data = pkl.load(f)
                # 【关键修复】支持新旧两种格式
                if isinstance(saved_data, dict):
                    self.mean_distance = saved_data['mean_distance']
                    self.x_train_mean = saved_data['x_train_mean']
                    self.x_train_std = saved_data['x_train_std']
                    self.nbrs = saved_data['nbrs']
                    self.standardization_epsilon = float(
                        saved_data.get('standardization_epsilon', 1e-6)
                    )
                    self.knn_score_space = str(
                        saved_data.get('score_space', 'dl_v1')
                    )
                else:
                    # 旧格式：只保存了mean_distance
                    self.mean_distance, _ = saved_data
                    logger.warning("使用旧格式KNN基准，将重新生成完整模型")
                    self._generate_knn_baseline()
                    return
            
            logger.info(f"[DL Detector] KNN基准加载成功 - 平均距离: {self.mean_distance:.4f}")
    
    def _generate_knn_baseline(self):
        """
        生成KNN基准（仅在没有基准文件时调用）
        使用训练集数据计算正常节点的平均KNN距离
        """
        # 加载metadata
        metadata = load_metadata(self.dataset_name)
        n_train = metadata['n_train']
        
        # 提取训练集特征
        x_train = []
        with torch.no_grad():
            for i in range(n_train):
                g = load_entity_level_dataset(self.dataset_name, 'train', i).to(self.device)
                x_train.append(self.model.embed(g).cpu().numpy())
                del g
        
        x_train = np.concatenate(x_train, axis=0)
        
        # 标准化
        self.x_train_mean = x_train.mean(axis=0)
        self.x_train_std = x_train.std(axis=0)
        x_train = ((x_train - self.x_train_mean) /
                   (self.x_train_std + self.standardization_epsilon))
        
        # 构建KNN模型
        if self.dataset_name == 'cadets':
            n_neighbors = 200
        else:
            n_neighbors = 10
        
        # 【性能优化】根据是否有cuML选择不同的KNN实现
        if USE_CUML:
            # GPU加速的KNN（cuML）
            self.nbrs = cuNearestNeighbors(n_neighbors=n_neighbors)
            # cuML需要GPU数组，将数据转移到GPU
            import cupy as cp
            x_train_gpu = cp.asarray(x_train)
            self.nbrs.fit(x_train_gpu)
            logger.info("[DL Detector] KNN模型已使用GPU加速 (cuML)")
        else:
            # CPU版KNN（sklearn）
            self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
            self.nbrs.fit(x_train)
        
        # 计算训练集内部平均距离（作为基准）
        idx = list(range(x_train.shape[0]))
        np.random.shuffle(idx)
        distances, _ = self.nbrs.kneighbors(
            x_train[idx][:min(50000, x_train.shape[0])], 
            n_neighbors=n_neighbors
        )
        self.mean_distance = distances.mean()
        
        # 保存基准（使用独立的文件名，避免覆盖原始MAGIC的缓存）
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dict_path = os.path.join(root, 'MAGIC_main', 'eval_result', f'distance_save_{self.dataset_name}_dl.pkl')
        with open(save_dict_path, 'wb') as f:
            # 【关键修复】保存完整的KNN模型，包括nbrs对象
            pkl.dump({
                'version': 2,
                'dataset': self.dataset_name,
                'score_space': 'dl_v1',
                'standardization_epsilon': self.standardization_epsilon,
                'mean_distance': self.mean_distance,
                'x_train_mean': self.x_train_mean,
                'x_train_std': self.x_train_std,
                'nbrs': self.nbrs
            }, f)
        
        logger.info(f"[DL Detector] KNN基准生成并保存 - 平均距离: {self.mean_distance:.4f}")
        logger.info(f"[DL Detector] 缓存文件: {save_dict_path}")
    
    def _extract_embeddings(self, graph):
        """
        提取图的嵌入表示（支持缓存优化）
        
        Args:
            graph: DGL图对象
            
        Returns:
            numpy.ndarray: 节点级嵌入向量 (num_nodes, hidden_dim)
        """
        # 【性能优化】尝试从缓存中获取embedding
        # Size-only keys collide for different topologies with the same edit
        # count.  Include the ordered edges and node features in the key.
        src, dst = graph.edges()
        edge_key = (tuple(src.detach().cpu().tolist()),
                    tuple(dst.detach().cpu().tolist()))
        attr = graph.ndata.get('attr')
        attr_key = (tuple(attr.detach().cpu().reshape(-1).tolist()),
                    tuple(attr.shape)) if attr is not None else None
        graph_hash = hash((graph.number_of_nodes(), edge_key, attr_key))
        if graph_hash in self._embedding_cache:
            return self._embedding_cache[graph_hash]
        
        with torch.no_grad():
            graph = graph.to(self.device)
            embedding = self.model.embed(graph)
            # 【验证】确认是节点级嵌入
            assert embedding.shape[0] == graph.number_of_nodes(), \
                f"Embedding shape {embedding.shape} doesn't match num_nodes {graph.number_of_nodes()}"
            embedding_np = embedding.cpu().numpy()
        
        # 【性能优化】缓存结果（限制缓存大小，避免内存泄漏）
        if len(self._embedding_cache) < 10:  # 最多缓存10张图
            self._embedding_cache[graph_hash] = embedding_np
        
        return embedding_np
    
    def query(self, graph, node_indices=None, return_tp=False):
        """
        黑盒查询接口：输入溯源图，返回top-K异常节点
        
        Args:
            graph: DGL图对象（修改后的溯源图）
            node_indices: 可选，指定需要查询的节点索引列表
                         如果为None，则返回全图top-K节点
            return_tp: 是否返回TP节点列表
        
        Returns:
            dict: {
                'top_k_nodes': list,      # top-K异常节点索引
                'anomaly_scores': list,   # 对应的异常分数
                'all_scores': dict,       # 所有节点的异常分数（节点索引->分数）
                'tp_nodes': list,         # TP节点列表（仅当return_tp=True时）
                'threshold': float        # 分类阈值（仅当return_tp=True时）
            }
        """
        logger.debug(f"query() - 节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
        
        # 提取图嵌入
        x_test = self._extract_embeddings(graph)
        
        if x_test.shape[0] >= 3:
            logger.debug(f"embedding[0:3] mean: {x_test[:3].mean(axis=1)}, std: {x_test[:3].std(axis=1)}")
        
        # 标准化（使用训练集的均值和标准差）
        if self.x_train_mean is None or self.x_train_std is None:
            # 如果没有标准化参数，使用当前数据的统计量
            x_mean = x_test.mean(axis=0)
            x_std = x_test.std(axis=0)
            x_test = (x_test - x_mean) / (x_std + self.standardization_epsilon)
        else:
            x_test = ((x_test - self.x_train_mean) /
                      (self.x_train_std + self.standardization_epsilon))
        
        # 计算KNN距离
        if self.nbrs is None:
            # 如果KNN模型未初始化，重新生成
            self._generate_knn_baseline()
        
        n_neighbors = 200 if self.dataset_name == 'cadets' else 10
        
        # 【性能优化】根据是否有cuML选择不同的处理方式
        if USE_CUML:
            import cupy as cp
            x_test_gpu = cp.asarray(x_test)
            distances, _ = self.nbrs.kneighbors(x_test_gpu, n_neighbors=n_neighbors)
            # cuML返回的是GPU数组，需要转回CPU
            distances = distances.get() if hasattr(distances, 'get') else distances
        else:
            distances, _ = self.nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        
        logger.debug(f"KNN距离[0:3]: {distances[:3].mean(axis=1)}")
        
        # 计算异常分数（归一化距离）
        raw_scores = distances.mean(axis=1) / self.mean_distance
        
        # 【优化】对mean_distance较小的数据集使用log变换，压缩极端值范围
        # Theia: 0.000668, Trace: 0.002972 (比Cadets的0.025125小很多)
        if self.dataset_name in ['theia', 'trace']:
            scores = np.log1p(raw_scores)  # log(1+x)，避免log(0)
        else:
            scores = raw_scores
        
        # 构建节点索引到分数的映射
        all_scores = {}
        for i, score in enumerate(scores):
            all_scores[i] = float(score)
        
        # 如果指定了节点索引，只返回这些节点的分数
        if node_indices is not None:
            filtered_scores = {idx: all_scores[idx] for idx in node_indices if idx in all_scores}
        else:
            filtered_scores = all_scores
        
        # 按异常分数降序排序
        sorted_nodes = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回top-K
        top_k_nodes = [node_idx for node_idx, _ in sorted_nodes[:self.top_k]]
        top_k_scores = [score for _, score in sorted_nodes[:self.top_k]]
        
        result = {
            'top_k_nodes': top_k_nodes,
            'anomaly_scores': top_k_scores,
            'all_scores': all_scores,
            'native_alerts': ([i for i, score in all_scores.items() if score >= self.best_threshold]
                              if self.best_threshold is not None else top_k_nodes),
            'threshold': self.best_threshold,
            'score_direction': self.score_direction,
            'detector_family': self.detector_family,
            'metadata': {'model': 'MAGIC', 'dataset': self.dataset_name}
        }
        
        # 【新增】如果需要，计算TP节点
        if return_tp:
            tp_nodes = self._compute_tp_nodes(all_scores)
            result['tp_nodes'] = tp_nodes
            result['threshold'] = self.best_threshold
        
        return result

    def query_selected_scores(self, graph, node_indices):
        """Score only selected nodes after embedding the complete graph.

        GNN message passing still uses the entire graph, so scores are exactly
        the same as slicing a full ``query`` result.  Only the expensive CPU
        KNN lookup is restricted to the requested nodes.
        """
        node_count = int(graph.number_of_nodes())
        selected = sorted({
            int(node) for node in node_indices
            if 0 <= int(node) < node_count
        })
        if not selected:
            return {}
        embeddings = self._extract_embeddings(graph)
        selected_array = np.asarray(selected, dtype=np.int64)
        x_test = embeddings[selected_array]
        if self.x_train_mean is None or self.x_train_std is None:
            raise RuntimeError(
                "selected-node KNN scoring requires training normalization stats"
            )
        x_test = ((x_test - self.x_train_mean) /
                  (self.x_train_std + self.standardization_epsilon))
        n_neighbors = 200 if self.dataset_name == 'cadets' else 10
        if USE_CUML:
            import cupy as cp
            distances, _ = self.nbrs.kneighbors(
                cp.asarray(x_test), n_neighbors=n_neighbors
            )
            distances = distances.get() if hasattr(distances, 'get') else distances
        else:
            distances, _ = self.nbrs.kneighbors(
                x_test, n_neighbors=n_neighbors
            )
        raw_scores = distances.mean(axis=1) / self.mean_distance
        if self.dataset_name in ['theia', 'trace']:
            scores = np.log1p(raw_scores)
        else:
            scores = raw_scores
        return {
            node: float(score) for node, score in zip(selected, scores)
        }
    
    def query_cached(self, graph, x_embedding=None, node_indices=None, return_tp=False):
        """
        优化的查询接口（支持传入已计算的嵌入，避免重复计算）
        
        Args:
            graph: DGL图对象
            x_embedding: 可选，已计算的嵌入向量（避免重复计算）
            node_indices: 可选，指定需要查询的节点索引列表
            
        Returns:
            dict: 查询结果
        """
        # 如果没有传入嵌入，则计算
        if x_embedding is None:
            x_test = self._extract_embeddings(graph)
        else:
            x_test = x_embedding
        
        # 标准化
        if self.x_train_mean is not None and self.x_train_std is not None:
            x_test = ((x_test - self.x_train_mean) /
                      (self.x_train_std + self.standardization_epsilon))
        
        # 计算KNN距离
        if self.nbrs is None:
            self._generate_knn_baseline()
        
        n_neighbors = 200 if self.dataset_name == 'cadets' else 10
        distances, _ = self.nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        
        # 计算异常分数
        raw_scores = distances.mean(axis=1) / self.mean_distance
        
        # 【优化】对mean_distance较小的数据集使用log变换，压缩极端值范围
        if self.dataset_name in ['theia', 'trace']:
            scores = np.log1p(raw_scores)  # log(1+x)，避免log(0)
        else:
            scores = raw_scores
        
        # 构建节点索引到分数的映射
        all_scores = {}
        for i, score in enumerate(scores):
            all_scores[i] = float(score)
        
        # 过滤节点
        if node_indices is not None:
            filtered_scores = {idx: all_scores[idx] for idx in node_indices if idx in all_scores}
        else:
            filtered_scores = all_scores
        
        # 按异常分数降序排序
        sorted_nodes = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回top-K
        top_k_nodes = [node_idx for node_idx, _ in sorted_nodes[:self.top_k]]
        top_k_scores = [score for _, score in sorted_nodes[:self.top_k]]
        
        result = {
            'top_k_nodes': top_k_nodes,
            'anomaly_scores': top_k_scores,
            'all_scores': all_scores,
            'native_alerts': ([i for i, score in all_scores.items() if score >= self.best_threshold]
                              if self.best_threshold is not None else top_k_nodes),
            'threshold': self.best_threshold,
            'score_direction': self.score_direction,
            'detector_family': self.detector_family,
            'metadata': {'model': 'MAGIC', 'dataset': self.dataset_name}
        }
        
        # 【新增】如果需要，计算TP节点
        if return_tp:
            tp_nodes = self._compute_tp_nodes(all_scores)
            result['tp_nodes'] = tp_nodes
            result['threshold'] = self.best_threshold
        
        return result
    
    def get_node_anomaly_score(self, graph, node_index):
        """
        获取单个节点的异常分数（优化版：避免重新计算KNN）
        
        Args:
            graph: DGL图对象
            node_index: 节点索引
            
        Returns:
            float: 节点的异常分数
        """
        # 提取图嵌入
        x_test = self._extract_embeddings(graph)
        
        # 标准化
        if self.x_train_mean is not None and self.x_train_std is not None:
            x_test = ((x_test - self.x_train_mean) /
                      (self.x_train_std + self.standardization_epsilon))
        
        # 计算KNN距离
        if self.nbrs is None:
            self._generate_knn_baseline()
        
        n_neighbors = 200 if self.dataset_name == 'cadets' else 10
        distances, _ = self.nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        
        # 计算异常分数
        raw_score = distances.mean(axis=1)[0] / self.mean_distance
        
        # 【优化】对mean_distance较小的数据集使用log变换，压缩极端值范围
        if self.dataset_name in ['theia', 'trace']:
            score = np.log1p(raw_score)  # log(1+x)，避免log(0)
        else:
            score = raw_score
        
        return float(score)
    
    def compute_confidence(self, graph, node_indices):
        """
        计算节点的可信度（用于智能体状态表示）
        
        Args:
            graph: DGL图对象
            node_indices: 节点索引列表
            
        Returns:
            dict: {节点索引: 可信度分数}
        """
        result = self.query(graph, node_indices=node_indices)
        confidence = {}
        
        for idx in node_indices:
            if idx in result['all_scores']:
                # 可信度 = 1 / (1 + 异常分数)，分数越低越可信
                confidence[idx] = 1.0 / (1.0 + result['all_scores'][idx])
            else:
                confidence[idx] = 1.0  # 默认高可信度
        
        return confidence
    
    def _print_initial_metrics(self):
        """
        初始化时计算并打印原始MAGIC的性能指标（作为攻击基准）
        与eval.py的输出格式对齐
        
        【修复】只在被攻击的最后一张测试图上计算初始指标，
        与攻击后评估和 initialize_tp_baseline() 保持一致。
        这样攻击前后的指标变化才是同一批样本之间的比较。
        """
        logger.info(f"\n{'=' * 40} 原始MAGIC检测器性能指标 (攻击前基准) {'=' * 40}")
        
        # 加载测试集数据
        metadata = load_metadata(self.dataset_name)
        n_test = metadata['n_test']
        malicious, _ = metadata['malicious']
        
        # 【修复】只加载最后一张测试图（与攻击目标和 initialize_tp_baseline 一致）
        logger.info(f"数据集共有{n_test}个测试图，只在最后一张图上计算初始指标（与攻击评估范围一致）...")
        
        g = load_entity_level_dataset(self.dataset_name, 'test', n_test - 1).to(self.device)
        x_test = self.model.embed(g).detach().cpu().numpy()
        last_graph_nodes = g.number_of_nodes()
        del g
        
        logger.info(f"最后一张测试图: {last_graph_nodes} 个节点")
        
        # 标准化
        if self.x_train_mean is not None and self.x_train_std is not None:
            x_test = ((x_test - self.x_train_mean) /
                      (self.x_train_std + self.standardization_epsilon))
        
        # 计算KNN距离和异常分数
        n_neighbors = 200 if self.dataset_name == 'cadets' else 10
        distances, _ = self.nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        raw_scores = distances.mean(axis=1) / self.mean_distance
        
        # 【优化】对mean_distance较小的数据集使用log变换，压缩极端值范围
        if self.dataset_name in ['theia', 'trace']:
            scores = np.log1p(raw_scores)  # log(1+x)，避免log(0)
        else:
            scores = raw_scores
        
        # 生成真实标签（只使用最后一张图范围内的恶意节点索引）
        num_nodes = len(scores)
        valid_malicious = [idx for idx in malicious if idx < num_nodes]
        y_test = np.zeros(num_nodes)
        if len(valid_malicious) > 0:
            y_test[valid_malicious] = 1.0
        
        # 计算AUC
        auc = roc_auc_score(y_test, scores)
        
        # 计算Precision-Recall曲线和F1
        prec, rec, threshold = precision_recall_curve(y_test, scores)
        f1 = 2 * prec * rec / (rec + prec + 1e-9)
        
        # 找到最佳阈值（对应最大F1）
        best_idx = -1
        for i in range(len(f1)):
            # 参考eval.py的逻辑
            if self.dataset_name == 'trace' and rec[i] < 0.99979:
                best_idx = i - 1
                break
            # Match the operating point used by the original MAGIC evaluation.
            if self.dataset_name == 'theia' and rec[i] < 0.99996:
                best_idx = i - 1
                break
            if self.dataset_name == 'cadets' and rec[i] < 0.9976:
                best_idx = i - 1
                break
        
        if best_idx < 0:
            best_idx = np.argmax(f1)
        
        best_thres = threshold[best_idx] if best_idx < len(threshold) else self.best_threshold
        
        # 计算混淆矩阵
        tn = fn = tp = fp = 0
        for i in range(len(y_test)):
            if y_test[i] == 1.0 and scores[i] >= best_thres:
                tp += 1
            elif y_test[i] == 1.0 and scores[i] < best_thres:
                fn += 1
            elif y_test[i] == 0.0 and scores[i] < best_thres:
                tn += 1
            elif y_test[i] == 0.0 and scores[i] >= best_thres:
                fp += 1
        
        # 打印指标（与eval.py格式完全对齐 - 大写、无额外装饰）
        logger.info('AUC: {}'.format(auc))
        logger.info('F1: {}'.format(f1[best_idx]))
        logger.info('PRECISION: {}'.format(prec[best_idx]))
        logger.info('RECALL: {}'.format(rec[best_idx]))
        logger.info('TN: {}'.format(tn))
        logger.info('FN: {}'.format(fn))
        logger.info('TP: {}'.format(tp))
        logger.info('FP: {}'.format(fp))
        logger.info(f"{'=' * 80}")
        
        # 保存基准指标供后续对比
        self.initial_metrics = {
            'auc': auc,
            'f1': f1[best_idx],
            'precision': prec[best_idx],
            'recall': rec[best_idx],
            'tn': tn,
            'fn': fn,
            'tp': tp,
            'fp': fp,
            'threshold': best_thres
        }
    
    def _compute_tp_nodes(self, all_scores):
        """
        根据当前分数和最佳阈值计算TP节点列表
        
        Args:
            all_scores: 所有节点的异常分数字典 {node_idx: score}
            
        Returns:
            list: TP节点索引列表(按node_idx排序,保证稳定性)
        """
        if self.best_threshold is None:
            # 如果还没有计算阈值，使用默认策略：分数>1.0视为异常
            threshold = 1.0
        else:
            threshold = self.best_threshold
        
        # 【关键修复】只返回分数>=阈值的节点
        tp_nodes = sorted([node_idx for node_idx, score in all_scores.items() 
                          if score >= threshold])
        
        return tp_nodes

    def _select_native_threshold(self, all_scores):
        """Reproduce MAGIC's native metadata-based operating threshold.

        This path exists for native MAGIC reproduction.  For a strict held-out
        experiment, callers should precompute a validation threshold and use
        ``threshold_source='fixed'`` instead.
        """
        if self.dataset_name == "trace":
            if self._native_evaluation_cache is None:
                self._load_trace_evaluation_context()
            return float(self._native_evaluation_cache["threshold"])
        num_nodes = len(all_scores)
        valid_native = [idx for idx in self.native_malicious_nodes if idx < num_nodes]
        if not valid_native:
            raise ValueError("native MAGIC threshold selection has no in-range metadata positives")
        labels = np.zeros(num_nodes, dtype=int)
        labels[valid_native] = 1
        scores = np.asarray([all_scores.get(i, 0.0) for i in range(num_nodes)])
        precision, recall, thresholds = precision_recall_curve(labels, scores)

        best_idx = -1
        for i, value in enumerate(recall):
            if self.dataset_name == 'trace' and value < 0.99979:
                best_idx = i - 1
                break
            if self.dataset_name == 'theia' and value < 0.99996:
                best_idx = i - 1
                break
            if self.dataset_name == 'cadets' and value < 0.9976:
                best_idx = i - 1
                break
        if best_idx < 0 or best_idx >= len(thresholds):
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            best_idx = min(int(np.argmax(f1)), len(thresholds) - 1)
        return float(thresholds[best_idx])

    @staticmethod
    def _classification_metrics(labels, scores, threshold):
        return native_classification_metrics(labels, scores, threshold)
    
    def initialize_tp_baseline(self, graph):
        """
        在攻击开始前初始化TP基准（只调用一次）
        
        【重要修复】初始TP节点应该是检测器判定为恶意的所有节点（TP+FP）
        因为攻击的目标是让这些节点逃逸检测（分数低于阈值）
        
        Args:
            graph: 初始测试图
            
        Returns:
            dict: 包含initial_tp_nodes和threshold
        """
        logger.info("[DL Detector] 正在初始化TP基准...")
        
        # 【关键】重置历史逃逸集合
        self.escaped_nodes_history = set()
        self._last_asr = 0.0
        
        # TRACE/test4的干净全节点分数已包含在一次性原生评估缓存中。
        # 物理计算可以复用，但仍按一次逻辑黑盒查询计入实验预算。
        result = self._trace_clean_scores_from_native_cache(graph)
        self.last_baseline_cache_hit = result is not None
        if result is None:
            result = self.query(graph, return_tp=False)
        all_scores = result['all_scores']

        # 【优化】使用缓存的 malicious 数据，避免重复 load_metadata
        num_nodes = len(all_scores)
        valid_malicious = [idx for idx in self.malicious_nodes if idx < num_nodes]
        
        # 生成当前被攻击分图的真实标签。TRACE标签已经由全局拼接ID
        # 转换为该分图的局部节点ID；它只用于定向TP/DASR审计。
        y_test = np.zeros(num_nodes)
        if len(valid_malicious) > 0:
            y_test[valid_malicious] = 1.0
        
        # 转换为数组形式用于计算
        scores_array = np.array([all_scores.get(i, 0.0) for i in range(num_nodes)])
        
        if self.threshold_source == 'fixed':
            self.best_threshold = float(self.configured_threshold)
        else:
            self.best_threshold = self._select_native_threshold(all_scores)
        
        # 【关键修复】初始TP节点 = 检测器判定为恶意的所有节点（分数>=阈值）
        # 这包括真正的TP节点和FP节点
        self.initial_tp_nodes = self._compute_tp_nodes(all_scores)
        
        # 缓存 MAGIC metadata 真值集合，仅供攻击前后离线评估使用。
        self.valid_malicious = valid_malicious
        self.true_malicious_set = set(valid_malicious)
        self.initial_true_tp_nodes = set(
            node_idx for node_idx in self.initial_tp_nodes
            if node_idx in self.true_malicious_set
        )
        
        # 记录固定阈值下的初始 MAGIC metadata TP 节点数。
        # 【重要说明】对于多图数据集(如trace)，这里只统计当前图(最后一个图)中的真正TP节点
        # 因为我们只攻击了最后一个图，ASR应该反映该图的逃逸率
        initial_tp_set = set(self.initial_tp_nodes)
        self.initial_alert_nodes = set(initial_tp_set)
        self.true_tp_count = len(self.initial_true_tp_nodes)
        self.false_positive_count = len(initial_tp_set - self.true_malicious_set)
        # 检测性能指标遵循原始MAGIC评估范围。TRACE使用前四图恶意节点
        # 加最后一图全部节点的全局集合；其他数据集仍使用当前单图。
        self.initial_metrics = self.evaluate_native_scores(
            all_scores, threshold=self.best_threshold
        )
        
        logger.info(f"[DL Detector] TP基准初始化完成:")
        logger.info(f"  最佳阈值: {self.best_threshold:.4f}")
        logger.info(f"  初始TP节点数(TP+FP): {len(self.initial_tp_nodes)}")
        logger.info(f"  其中真正TP: {self.true_tp_count}, FP: {self.false_positive_count}")
        logger.info(
            "  原始MAGIC评估: "
            f"AUC={self.initial_metrics['auc']:.6f}, "
            f"F1={self.initial_metrics['f1']:.6f}, "
            f"Precision={self.initial_metrics['precision']:.6f}, "
            f"Recall={self.initial_metrics['recall']:.6f}, "
            f"TP={self.initial_metrics['tp']}, FP={self.initial_metrics['fp']}"
        )
        
        return {
            'initial_tp_nodes': self.initial_tp_nodes,
            'initial_true_tp_nodes': sorted(self.initial_true_tp_nodes),
            'threshold': self.best_threshold,
            'asr_denominator': self.true_tp_count,
            'all_scores': all_scores,
            'ground_truth': 'metadata_native',
            'threshold_source': self.threshold_source,
        }
    
    def compute_asr(self, current_graph):
        """
        计算当前的ASR (Attack Success Rate)
        
        【重要】ASR是单调不减的累计值:
        - 节点第一次低于阈值时,计入逃逸
        - 即使后续又高于阈值,也不从逃逸集合中移除
        
        ASR = (历史逃逸的初始TP节点数) / (初始TP节点总数)
        
        Args:
            current_graph: 当前修改后的图
            
        Returns:
            float: ASR值 (0.0 - 1.0)
        """
        if self.initial_tp_nodes is None:
            raise RuntimeError("必须先调用initialize_tp_baseline()初始化TP基准")
        
        if len(self.initial_tp_nodes) == 0:
            return 0.0
        
        # 查询当前图的分数
        result = self.query(current_graph, return_tp=False)
        current_scores = result['all_scores']
        
        # 使用统一的计算方法(基于已有分数)
        asr = self._compute_asr_from_scores(current_scores)
        
        return asr
    
    def _compute_asr_from_scores(self, current_scores):
        """
        基于已有分数计算ASR,避免重复query
        
        【重要】ASR是单调不减的累计值:
        - 节点第一次低于阈值时,计入逃逸
        - 即使后续又高于阈值,也不从逃逸集合中移除
        
        【关键定义】ASR = (历史逃逸的真正TP节点数) / (当前图中的真正TP节点总数)
        - 分子: 初始时分数>=阈值(被检测到)，攻击后分数<阈值(逃逸)的真正TP节点数
        - 分母: 当前图(最后一个测试图)中的固定真正TP节点数，不随攻击变化
        
        【重要说明】对于多图数据集(如trace)，ASR只反映最后一个图的逃逸率，
        因为我们只攻击了最后一个图。其他图的节点状态不会改变。
        
        Args:
            current_scores: dict {node_idx: score} - 已经query得到的分数
            
        Returns:
            float: ASR值 (0.0 - 1.0)
        """
        if self.initial_tp_nodes is None:
            raise RuntimeError("必须先调用initialize_tp_baseline()初始化TP基准")
        
        if self.true_tp_count == 0:
            return 0.0
        
        # 【优化】直接使用缓存的 initial_true_tp_nodes，避免每次 load_metadata
        initial_true_tp_nodes = self.initial_true_tp_nodes
        
        # 【验证】确保数量正确
        if len(initial_true_tp_nodes) != self.true_tp_count:
            logger.warning(f"initial_true_tp_nodes数量({len(initial_true_tp_nodes)}) != true_tp_count({self.true_tp_count})")
        
        # 【关键修复】找出当前仍然被检测到的真正TP节点（分数>=阈值）
        current_detected_true_tp = set()
        for node_idx in initial_true_tp_nodes:
            if node_idx in current_scores and current_scores[node_idx] >= self.best_threshold:
                current_detected_true_tp.add(node_idx)
        
        # 【关键修复】新逃逸的真正TP节点 = 初始真正TP - 当前仍被检测到的真正TP
        newly_escaped_true_tp = initial_true_tp_nodes - current_detected_true_tp
        
        # 【关键修改】将新逃逸的真正TP节点加入历史集合(只增不减)
        old_escaped_count = len(self.escaped_nodes_history)
        
        # 【调试】计算真正新增的节点(之前没逃逸过的)
        truly_new_escaped = newly_escaped_true_tp - self.escaped_nodes_history
        
        self.escaped_nodes_history.update(truly_new_escaped)  # 【关键修复】只添加真正新增的节点
        new_escaped_count = len(self.escaped_nodes_history)
        
        # 【关键修复】ASR = 历史逃逸的真正TP节点数 / 当前图中的真正TP总数
        # 【重要说明】对于多图数据集，分母只包含最后一个图的TP节点，因为我们只攻击了该图
        # 确保ASR不超过1.0
        asr = min(len(self.escaped_nodes_history) / self.true_tp_count, 1.0)
        
        # 如果ASR发生变化,记录详细信息
        if hasattr(self, '_last_asr') and abs(asr - self._last_asr) > 1e-6:
            logger.debug(f"ASR {self._last_asr:.2%} -> {asr:.2%}")
            logger.debug(f"  初始TP: {len(initial_true_tp_nodes)}, 当前检测到: {len(current_detected_true_tp)}, 逃逸: {len(newly_escaped_true_tp)}, 历史逃逸: {len(self.escaped_nodes_history)}, 新增: {len(truly_new_escaped)}")
            
            if len(truly_new_escaped) > 0:
                truly_new_list = sorted(list(truly_new_escaped))[:10]
                for node_id in truly_new_list:
                    score = current_scores.get(node_id, 'N/A')
                    logger.debug(f"  新逃逸节点{node_id}: 分数={score:.4f}, 阈值={self.best_threshold:.4f}")
                
                if hasattr(self, '_last_attacked_nodes'):
                    directly_attacked = truly_new_escaped.intersection(self._last_attacked_nodes)
                    indirectly_affected = truly_new_escaped - self._last_attacked_nodes
                    logger.debug(f"  直接攻击: {len(directly_attacked)}, 间接影响: {len(indirectly_affected)}")
        
        self._last_asr = asr  # 保存上一次ASR
        
        return asr

    def compute_asr_from_scores(self, scores, *, final=False):
        """Stable public API; ``final`` returns non-cumulative final-state ASR."""
        if not final:
            return self._compute_asr_from_scores(scores)
        if self.true_tp_count == 0:
            return 0.0
        escaped = {
            node for node in self.initial_true_tp_nodes
            if scores.get(node, float('-inf')) < self.best_threshold
        }
        return len(escaped) / self.true_tp_count

    def set_evaluation_context(self, test_index):
        self.test_index = int(test_index)
        if not 0 <= self.test_index < int(self.metadata['n_test']):
            raise IndexError(
                f"test_index {self.test_index} outside [0, {self.metadata['n_test'] - 1}]"
            )
        if self.dataset_name == "trace":
            self._load_trace_evaluation_context()
        else:
            # Single-graph entity datasets already use local metadata IDs.
            self.malicious_nodes = list(self.global_malicious_nodes)
            self.malicious_set = set(self.malicious_nodes)
            self.native_malicious_set = set(self.malicious_nodes)
    


# 便捷函数：直接使用检测器
def create_detector(dataset_name='cadets', device='cuda:0', top_k=30):
    """
    创建检测器实例的便捷函数
    
    Args:
        dataset_name: 数据集名称
        device: 设备
        top_k: top-K节点数量
        
    Returns:
        DLDetector: 检测器实例
    """
    return DLDetector(dataset_name=dataset_name, device=device, top_k=top_k)


if __name__ == '__main__':
    # 测试代码
    print("=== DL Detector 测试 ===")
    
    # 创建检测器
    detector = create_detector(dataset_name='cadets', top_k=30)
    
    # 加载测试图
    from utils.loaddata import load_entity_level_dataset
    test_graph = load_entity_level_dataset('cadets', 'test', 0)
    
    # 查询异常节点
    result = detector.query(test_graph)
    
    print(f"\nTop-30 异常节点:")
    for i, (node_idx, score) in enumerate(zip(result['top_k_nodes'], result['anomaly_scores'])):
        print(f"  {i+1}. 节点 {node_idx}: 异常分数 {score:.4f}")
    
    print("\n测试完成！")
