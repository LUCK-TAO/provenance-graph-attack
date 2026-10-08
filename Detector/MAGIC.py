"""
DL Detector (PIDS) - 黑盒检测器封装
基于MAGIC模型的溯源图异常检测器
提供黑盒查询接口：输入修改后的溯源图，返回top-K异常节点及其分数
"""

import os
import pickle as pkl
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_curve
from MAGIC_main.utils.loaddata import load_metadata
from MAGIC_main.model.autoencoder import build_model
import warnings

warnings.filterwarnings('ignore')


class DLDetector:
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
    
    def __init__(self, dataset_name='cadets', device='cuda:0', top_k=30):
        """
        初始化检测器
        
        Args:
            dataset_name: 数据集名称 ('cadets', 'theia', 'trace')
            device: 设备 ('cuda:0' 或 'cpu')
            top_k: 返回异常分数最高的节点数量
        """
        self.dataset_name = dataset_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.top_k = top_k
        
        # 模型和KNN相关参数
        self.model = None
        self.nbrs = None
        self.mean_distance = None
        self.x_train_mean = None
        self.x_train_std = None
        
        # 【新增】TP追踪相关
        self.best_threshold = None  # 最佳分类阈值
        self.initial_tp_nodes = None  # 初始TP节点集合
        self.escaped_nodes_history = set()  # 【关键】历史逃逸节点集合(只增不减)（攻击全程不变）
        
        # 加载模型和KNN基准
        self._load_model()
        self._load_knn_baseline()
        
        print(f"[DL Detector] 初始化完成 - 数据集: {dataset_name}, Top-K: {top_k}")
        
        # 【新增】初始化时计算并打印原始MAGIC的性能指标（作为攻击基准）
        self._print_initial_metrics()
    
    def _load_model(self):
        """加载预训练的MAGIC模型"""
        print(f"[DL Detector] 正在加载预训练模型...")
        
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
        checkpoint_path = f"./checkpoints/checkpoint-{self.dataset_name}.pt"
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"预训练权重文件不存在: {checkpoint_path}")
        
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"[DL Detector] 模型加载成功 - 隐藏维度: {num_hidden}, 层数: {num_layers}")
    
    def _load_knn_baseline(self):
        """
        加载KNN基准距离
        这些基准是在正常数据上计算的，用于异常评分
        """
        print(f"[DL Detector] 正在加载KNN基准...")
        
        save_dict_path = f'./eval_result/distance_save_{self.dataset_name}_dl.pkl'  # 使用独立的文件名
        
        if not os.path.exists(save_dict_path):
            # 如果KNN基准文件不存在，需要先生成
            print(f"[DL Detector] KNN基准文件不存在，正在生成...")
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
                else:
                    # 旧格式：只保存了mean_distance
                    self.mean_distance, _ = saved_data
                    print(f"[警告] 使用旧格式KNN基准，将重新生成完整模型")
                    self._generate_knn_baseline()
                    return
            
            print(f"[DL Detector] KNN基准加载成功 - 平均距离: {self.mean_distance:.4f}")
    
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
        x_train = (x_train - self.x_train_mean) / (self.x_train_std + 1e-6)
        
        # 构建KNN模型
        if self.dataset_name == 'cadets':
            n_neighbors = 200
        else:
            n_neighbors = 10
        
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
        save_dict_path = f'./eval_result/distance_save_{self.dataset_name}_dl.pkl'  # 添加_dl后缀
        with open(save_dict_path, 'wb') as f:
            # 【关键修复】保存完整的KNN模型，包括nbrs对象
            pkl.dump({
                'mean_distance': self.mean_distance,
                'x_train_mean': self.x_train_mean,
                'x_train_std': self.x_train_std,
                'nbrs': self.nbrs
            }, f)
        
        print(f"[DL Detector] KNN基准生成并保存 - 平均距离: {self.mean_distance:.4f}")
        print(f"[DL Detector] 缓存文件: {save_dict_path}")
    
    def _extract_embeddings(self, graph):
        """
        提取图的嵌入表示
        
        Args:
            graph: DGL图对象
            
        Returns:
            numpy.ndarray: 节点级嵌入向量 (num_nodes, hidden_dim)
        """
        with torch.no_grad():
            graph = graph.to(self.device)
            embedding = self.model.embed(graph)
            # 【验证】确认是节点级嵌入
            assert embedding.shape[0] == graph.number_of_nodes(), \
                f"Embedding shape {embedding.shape} doesn't match num_nodes {graph.number_of_nodes()}"
            return embedding.cpu().numpy()
    
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
        # 【关键调试】打印图的基本信息
        print(f"[DEBUG DL] query() - 节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
        
        # 提取图嵌入
        x_test = self._extract_embeddings(graph)
        
        # 【关键调试】打印前3个节点的embedding统计信息
        if x_test.shape[0] >= 3:
            print(f"[DEBUG DL] embedding[0:3] mean: {x_test[:3].mean(axis=1)}, std: {x_test[:3].std(axis=1)}")
        
        # 标准化（使用训练集的均值和标准差）
        if self.x_train_mean is None or self.x_train_std is None:
            # 如果没有标准化参数，使用当前数据的统计量
            x_mean = x_test.mean(axis=0)
            x_std = x_test.std(axis=0)
            x_test = (x_test - x_mean) / (x_std + 1e-6)
        else:
            x_test = (x_test - self.x_train_mean) / (self.x_train_std + 1e-6)
        
        # 计算KNN距离
        if self.nbrs is None:
            # 如果KNN模型未初始化，重新生成
            self._generate_knn_baseline()
        
        distances, _ = self.nbrs.kneighbors(x_test, n_neighbors=200 if self.dataset_name == 'cadets' else 10)
        
        # 【关键调试】打印前3个节点的KNN距离
        print(f"[DEBUG DL] KNN距离[0:3]: {distances[:3].mean(axis=1)}")
        
        # 计算异常分数（归一化距离）
        scores = distances.mean(axis=1) / self.mean_distance
        
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
            'all_scores': all_scores
        }
        
        # 【新增】如果需要，计算TP节点
        if return_tp:
            tp_nodes = self._compute_tp_nodes(all_scores)
            result['tp_nodes'] = tp_nodes
            result['threshold'] = self.best_threshold
        
        return result
    
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
            x_test = (x_test - self.x_train_mean) / (self.x_train_std + 1e-6)
        
        # 计算KNN距离
        if self.nbrs is None:
            self._generate_knn_baseline()
        
        n_neighbors = 200 if self.dataset_name == 'cadets' else 10
        distances, _ = self.nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        
        # 计算异常分数
        scores = distances.mean(axis=1) / self.mean_distance
        
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
            'all_scores': all_scores
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
            x_test = (x_test - self.x_train_mean) / (self.x_train_std + 1e-6)
        
        # 计算KNN距离
        if self.nbrs is None:
            self._generate_knn_baseline()
        
        n_neighbors = 200 if self.dataset_name == 'cadets' else 10
        distances, _ = self.nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        
        # 计算异常分数
        score = distances.mean(axis=1)[0] / self.mean_distance
        
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
        """
        print(f"\n{'=' * 80}")
        print(f"原始MAGIC检测器性能指标 (攻击前基准)")
        print(f"{'=' * 80}")
        
        # 加载测试集数据
        metadata = load_metadata(self.dataset_name)
        n_test = metadata['n_test']
        malicious, _ = metadata['malicious']
        
        # 提取测试集嵌入
        x_test = []
        with torch.no_grad():
            for i in range(n_test):
                g = load_entity_level_dataset(self.dataset_name, 'test', i).to(self.device)
                x_test.append(self.model.embed(g).cpu().numpy())
                del g
        
        x_test = np.concatenate(x_test, axis=0)
        
        # 标准化
        if self.x_train_mean is not None and self.x_train_std is not None:
            x_test = (x_test - self.x_train_mean) / (self.x_train_std + 1e-6)
        
        # 计算KNN距离和异常分数
        n_neighbors = 200 if self.dataset_name == 'cadets' else 10
        distances, _ = self.nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        scores = distances.mean(axis=1) / self.mean_distance
        
        # 生成真实标签
        num_nodes = len(scores)
        y_test = np.zeros(num_nodes)
        y_test[malicious] = 1.0
        
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
        print('AUC: {}'.format(auc))
        print('F1: {}'.format(f1[best_idx]))
        print('PRECISION: {}'.format(prec[best_idx]))
        print('RECALL: {}'.format(rec[best_idx]))
        print('TN: {}'.format(tn))
        print('FN: {}'.format(fn))
        print('TP: {}'.format(tp))
        print('FP: {}'.format(fp))
        print(f"{'=' * 80}\n")
        
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
        print("[DL Detector] 正在初始化TP基准...")
        
        # 【关键】重置历史逃逸集合
        self.escaped_nodes_history = set()
        self._last_asr = 0.0
        
        # 查询初始图的异常分数
        result = self.query(graph, return_tp=False)
        all_scores = result['all_scores']
        
        # 【关键修复】使用与eval.py完全一致的阈值计算逻辑
        # 需要加载真实标签来计算最佳阈值
        metadata = load_metadata(self.dataset_name)
        malicious, _ = metadata['malicious']
        
        # 生成真实标签
        num_nodes = len(all_scores)
        y_test = np.zeros(num_nodes)
        y_test[malicious] = 1.0
        
        # 转换为数组形式用于计算
        scores_array = np.array([all_scores.get(i, 0.0) for i in range(num_nodes)])
        
        # 计算Precision-Recall曲线
        prec, rec, threshold_curve = precision_recall_curve(y_test, scores_array)
        
        # 使用与eval.py相同的阈值选择策略
        best_idx = -1
        for i in range(len(rec)):
            # To repeat peak performance
            if self.dataset_name == 'trace' and rec[i] < 0.99979:
                best_idx = i - 1
                break
            if self.dataset_name == 'theia' and rec[i] < 0.99996:
                best_idx = i - 1
                break
            if self.dataset_name == 'cadets' and rec[i] < 0.9976:
                best_idx = i - 1
                break
        
        if best_idx < 0 or best_idx >= len(threshold_curve):
            best_idx = np.argmax(2 * prec * rec / (rec + prec + 1e-9))
        
        self.best_threshold = threshold_curve[best_idx] if best_idx < len(threshold_curve) else 1.0
        
        # 【关键修复】初始TP节点 = 检测器判定为恶意的所有节点（分数>=阈值）
        # 这包括真正的TP节点和FP节点
        self.initial_tp_nodes = self._compute_tp_nodes(all_scores)
        
        # 【关键修复】记录真正的TP节点数（用于ASR计算的分母）
        initial_tp_set = set(self.initial_tp_nodes)
        true_malicious_set = set(malicious)
        self.true_tp_count = len(initial_tp_set.intersection(true_malicious_set))  # 12816
        self.false_positive_count = len(initial_tp_set - true_malicious_set)  # 759
        
        print(f"[DL Detector] TP基准初始化完成:")
        print(f"  最佳阈值: {self.best_threshold:.4f}")
        print(f"  初始TP节点数(TP+FP): {len(self.initial_tp_nodes)}")
        print(f"  其中真正TP: {self.true_tp_count}, FP: {self.false_positive_count}")
        print(f"  ASR分母(固定): {self.true_tp_count}")
        
        return {
            'initial_tp_nodes': self.initial_tp_nodes,
            'threshold': self.best_threshold,
            'all_scores': all_scores
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
        
        【关键定义】ASR = (初始TP节点减少数) / 12816
        - 分子: 初始时分数>=阈值(被检测到)，攻击后分数<阈值(逃逸)的真正TP节点数
        - 分母: 固定的真正TP节点数(12816)，不随攻击变化
        
        Args:
            current_scores: dict {node_idx: score} - 已经query得到的分数
            
        Returns:
            float: ASR值 (0.0 - 1.0)
        """
        if self.initial_tp_nodes is None:
            raise RuntimeError("必须先调用initialize_tp_baseline()初始化TP基准")
        
        if self.true_tp_count == 0:
            return 0.0
        
        # 【关键修复】只统计初始TP节点中真正的TP节点（12816个）
        metadata = load_metadata(self.dataset_name)
        malicious, _ = metadata['malicious']
        true_malicious_set = set(malicious)
        
        # 【关键修复】找出初始TP节点中真正的TP节点集合
        initial_true_tp_nodes = set()
        for node_idx in self.initial_tp_nodes:
            if node_idx in true_malicious_set:
                initial_true_tp_nodes.add(node_idx)
        
        # 【验证】确保数量正确
        if len(initial_true_tp_nodes) != self.true_tp_count:
            print(f"[WARNING] initial_true_tp_nodes数量({len(initial_true_tp_nodes)}) != true_tp_count({self.true_tp_count})")
        
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
        
        # 【关键修复】ASR = 历史逃逸的真正TP节点数 / 固定的真正TP总数(12816)
        # 确保ASR不超过1.0
        asr = min(len(self.escaped_nodes_history) / self.true_tp_count, 1.0)
        
        # 【调试】如果ASR发生变化,打印详细信息
        if hasattr(self, '_last_asr') and abs(asr - self._last_asr) > 1e-6:
            print(f"\n  [DEBUG ASR] {self._last_asr:.2%} -> {asr:.2%}")
            print(f"    初始真正TP节点数: {len(initial_true_tp_nodes)}")
            print(f"    当前检测到真正TP数: {len(current_detected_true_tp)}")
            print(f"    本次计算的逃逸节点数: {len(newly_escaped_true_tp)}")
            print(f"    历史逃逸节点数: {len(self.escaped_nodes_history)}")
            print(f"    本次真正新增节点数: {len(truly_new_escaped)}")
            print(f"    ASR分母(固定): {self.true_tp_count}")
            
            if len(truly_new_escaped) > 0:
                # 【关键修复】打印真正新增的节点(之前没逃逸过的)
                truly_new_list = sorted(list(truly_new_escaped))[:10]
                print(f"    本次真正新逃逸节点(前{len(truly_new_list)}个):")
                for node_id in truly_new_list:
                    score = current_scores.get(node_id, 'N/A')
                    print(f"      - 节点{node_id}: 分数={score:.4f}, 阈值={self.best_threshold:.4f}")
                
                # 【新增】区分直接攻击节点和间接受影响节点
                if hasattr(self, '_last_attacked_nodes'):
                    directly_attacked = truly_new_escaped.intersection(self._last_attacked_nodes)
                    indirectly_affected = truly_new_escaped - self._last_attacked_nodes
                    print(f"    其中直接攻击节点: {len(directly_attacked)}个")
                    print(f"    其中间接受影响节点: {len(indirectly_affected)}个")
                    if len(indirectly_affected) > 0:
                        indirect_list = sorted(list(indirectly_affected))[:5]
                        print(f"    间接受影响节点示例(前{len(indirect_list)}个):")
                        for node_id in indirect_list:
                            score = current_scores.get(node_id, 'N/A')
                            print(f"      - 节点{node_id}: 分数={score:.4f}")
        
        self._last_asr = asr  # 保存上一次ASR
        
        return asr
    


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
    from MAGIC_main.utils.loaddata import load_entity_level_dataset
    test_graph = load_entity_level_dataset('cadets', 'test', 0)
    
    # 查询异常节点
    result = detector.query(test_graph)
    
    print(f"\nTop-30 异常节点:")
    for i, (node_idx, score) in enumerate(zip(result['top_k_nodes'], result['anomaly_scores'])):
        print(f"  {i+1}. 节点 {node_idx}: 异常分数 {score:.4f}")
    
    print("\n测试完成！")
