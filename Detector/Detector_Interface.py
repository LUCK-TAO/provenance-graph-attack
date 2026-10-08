"""统一的黑盒检测器接口。

适配器必须复用目标检测器的原生预处理、权重和推理规则。连续分数只用于
PPO 反馈，``native_alerts`` 才是原生告警结果；二者不能混用。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional
import math


class BaseDetector(ABC):
    score_direction = "higher_is_more_anomalous"
    detector_family = "unknown"

    @abstractmethod
    def __init__(self, dataset_name: str = "cadets", device: str = "cuda:0", top_k: int = 50, **kwargs):
        self.dataset_name = dataset_name
        self.device = device
        self.top_k = int(top_k)
        self.best_threshold = kwargs.get("threshold")
        self.initial_tp_nodes: List[int] = []
        self.initial_alert_nodes = set()
        self.escaped_nodes_history = set()

    @abstractmethod
    def query(self, graph, node_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """输入修改后的图，返回统一结果字典。"""

    def _make_result(self, scores: Dict[int, float], *, threshold: Optional[float] = None,
                     native_alerts: Optional[Iterable[int]] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     node_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        scores = {int(k): float(v) for k, v in scores.items() if math.isfinite(float(v))}
        visible = scores if node_indices is None else {int(i): scores[int(i)] for i in node_indices if int(i) in scores}
        ranked = sorted(visible.items(), key=lambda kv: (-kv[1], kv[0]))
        alerts = sorted({int(x) for x in (native_alerts if native_alerts is not None else []) if int(x) in scores})
        if threshold is None:
            threshold = self.best_threshold
        return {
            "top_k_nodes": [n for n, _ in ranked[:self.top_k]],
            "anomaly_scores": [s for _, s in ranked[:self.top_k]],
            "all_scores": scores,
            "native_alerts": alerts,
            "threshold": None if threshold is None else float(threshold),
            "score_direction": self.score_direction,
            "detector_family": self.detector_family,
            "metadata": metadata or {},
        }

    def get_node_anomaly_score(self, graph, node_index: int) -> float:
        return float(self.query(graph, node_indices=[node_index])["all_scores"].get(int(node_index), 0.0))

    def initialize_tp_baseline(self, graph) -> Dict[str, Any]:
        result = self.query(graph)
        self.initial_tp_nodes = list(result["native_alerts"] or result["top_k_nodes"])
        self.initial_alert_nodes = set(self.initial_tp_nodes)
        self.best_threshold = result.get("threshold")
        if self.best_threshold is None and self.initial_tp_nodes:
            self.best_threshold = min(result["all_scores"].get(n, float("inf")) for n in self.initial_tp_nodes)
        self.escaped_nodes_history = set()
        self.true_tp_count = len(self.initial_tp_nodes)
        return {"initial_tp_nodes": self.initial_tp_nodes, "threshold": self.best_threshold,
                "all_scores": result["all_scores"], "native_alerts": result["native_alerts"]}

    def compute_asr(self, graph_or_scores, threshold: Optional[float] = None,
                    true_malicious_set: Optional[set] = None, *, final: bool = False) -> float:
        if isinstance(graph_or_scores, dict):
            result = {"all_scores": graph_or_scores, "native_alerts": []}
        else:
            result = self.query(graph_or_scores)
        scores = result["all_scores"]
        targets = set(true_malicious_set or self.initial_tp_nodes)
        if not targets:
            return 0.0
        if result.get("native_alerts"):
            escaped = {n for n in targets if n not in set(result["native_alerts"])}
        else:
            th = self.best_threshold if threshold is None else threshold
            escaped = {n for n in targets if th is not None and scores.get(n, float("inf")) < th}
        if final:
            return len(escaped) / len(targets)
        self.escaped_nodes_history.update(escaped)
        return len(self.escaped_nodes_history) / len(targets)

    def record_attack_modification(self, **_kwargs) -> None:
        return None

    def compute_asr_from_scores(self, scores: Dict[int, float], *, final: bool = False) -> float:
        """Public score-only ASR hook used by detector-agnostic attacks."""
        return self.compute_asr(scores, final=final)

    def set_evaluation_context(self, test_index: int) -> None:
        """Select a per-graph calibration baseline, if the adapter has one."""
        self.test_index = int(test_index)

    def encode_attack_state(self, base_state, **_kwargs):
        return base_state
