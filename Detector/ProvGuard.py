"""Faithful black-box adapter for the entity-level ProvGuard detector.

The bundled ProvGuard evaluation code embeds every node with a two-layer GAT,
standardises the embeddings with training statistics, and uses the normalised
distance to the nearest KMeans centre as the anomaly score.  This adapter keeps
that score definition and exposes it through the common detector interface.

Ground-truth metadata is used only to reproduce ProvGuard's native evaluation
operating point and to report offline metrics.  It is never returned by
``query`` and is not available to the attack policy.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score

from Detector.Detector_Interface import BaseDetector
from MAGIC_main.utils.loaddata import load_metadata
from ProvGuard_main.model.model import build_model


logger = logging.getLogger("attack.detector.provguard")
ROOT = Path(__file__).resolve().parents[1]
PROVGUARD_ROOT = ROOT / "ProvGuard_main"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ProvGuardDetector(BaseDetector):
    """Entity-level ProvGuard adapter using the detector's native score rule."""

    score_direction = "higher_is_more_anomalous"
    detector_family = "nearest_kmeans_center"

    def __init__(
        self,
        dataset_name: str = "cadets",
        device: str = "cuda:0",
        top_k: int = 50,
        checkpoint_path: Optional[str] = None,
        threshold_source: str = "native",
        threshold: Optional[float] = None,
        score_batch_size: int = 65_536,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            device=device,
            top_k=top_k,
            threshold=threshold,
            **kwargs,
        )
        if threshold_source not in {"native", "fixed"}:
            raise ValueError("threshold_source must be 'native' or 'fixed'")
        if threshold_source == "fixed" and threshold is None:
            raise ValueError("threshold_source='fixed' requires threshold")
        if score_batch_size <= 0:
            raise ValueError("score_batch_size must be positive")

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is unavailable; ProvGuard will run on CPU")
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.threshold_source = threshold_source
        self.configured_threshold = threshold
        self.score_batch_size = int(score_batch_size)
        self.test_index = 0
        self.last_baseline_cache_hit = False

        self.metadata = load_metadata(dataset_name)
        self.node_feature_dim = int(self.metadata["node_feature_dim"])
        self.edge_feature_dim = int(self.metadata["edge_feature_dim"])
        malicious = self.metadata.get("malicious", [[], []])[0]
        self.global_malicious_nodes = [int(node) for node in malicious]
        self.malicious_nodes = list(self.global_malicious_nodes)
        self.malicious_set = set(self.malicious_nodes)
        self.true_malicious_set = set(self.malicious_nodes)
        self.initial_true_tp_nodes: set[int] = set()
        self.valid_malicious: List[int] = []
        self.false_positive_count = 0
        self.true_tp_count = 0
        self.initial_metrics: Dict[str, float] = {}

        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self.params_path = (
            PROVGUARD_ROOT
            / "data"
            / "eval_result"
            / f"{dataset_name}_trained_params.pkl"
        )
        self._load_model()
        self._load_scoring_parameters()
        self.model_sha256 = _sha256(self.checkpoint_path)
        self.params_sha256 = _sha256(self.params_path)

        logger.info(
            "[ProvGuard] initialized: dataset=%s, device=%s, node_dim=%d, "
            "edge_dim=%d, threshold_source=%s",
            dataset_name,
            self.device,
            self.node_feature_dim,
            self.edge_feature_dim,
            self.threshold_source,
        )

    def _resolve_checkpoint(self, explicit: Optional[str]) -> Path:
        candidates = []
        if explicit:
            candidates.append(Path(explicit).expanduser())
        candidates.extend(
            [
                PROVGUARD_ROOT
                / "data"
                / "res_models"
                / f"graphcl_model_{self.dataset_name}.pth",
                PROVGUARD_ROOT
                / "checkpoints"
                / f"checkpoint-{self.dataset_name}.pt",
                PROVGUARD_ROOT
                / "checkpoints"
                / f"checkpoint-{self.dataset_name}.pth",
            ]
        )
        for candidate in candidates:
            resolved = candidate if candidate.is_absolute() else ROOT / candidate
            if resolved.is_file():
                return resolved.resolve()
        searched = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "ProvGuard pretrained encoder weights are missing. The original "
            "evaluation code expects graphcl_model_<dataset>.pth. Searched:\n"
            f"  - {searched}\n"
            "Do not run an attack with randomly initialised weights. Supply the "
            "official/reproduced checkpoint with --detector-checkpoint."
        )

    def _load_model(self) -> None:
        args = SimpleNamespace(
            n_dim=self.node_feature_dim,
            e_dim=self.edge_feature_dim,
            num_hidden=64,
            num_layers=2,
        )
        self.model = build_model(args)
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise TypeError(
                f"unsupported ProvGuard checkpoint payload: {type(payload).__name__}"
            )
        if payload and all(str(key).startswith("module.") for key in payload):
            payload = {str(key)[7:]: value for key, value in payload.items()}
        self.model.load_state_dict(payload, strict=True)
        self.model.to(self.device)
        self.model.eval()
        logger.info("[ProvGuard] loaded encoder weights: %s", self.checkpoint_path)

    def _load_scoring_parameters(self) -> None:
        if not self.params_path.is_file():
            raise FileNotFoundError(
                "ProvGuard KMeans scoring parameters are missing: "
                f"{self.params_path}"
            )
        with self.params_path.open("rb") as handle:
            params = pickle.load(handle)
        required = {
            "cluster_centers",
            "global_mean_distance",
            "x_train_mean",
            "x_train_std",
        }
        missing = required - set(params)
        if missing:
            raise ValueError(
                f"incomplete ProvGuard scoring parameters; missing {sorted(missing)}"
            )
        self.cluster_centers = np.asarray(params["cluster_centers"], dtype=np.float32)
        self.mean_distance = float(params["global_mean_distance"])
        self.x_train_mean = np.asarray(params["x_train_mean"], dtype=np.float32)
        self.x_train_std = np.asarray(params["x_train_std"], dtype=np.float32)
        if self.cluster_centers.ndim != 2:
            raise ValueError("cluster_centers must be a two-dimensional array")
        embedding_dim = self.cluster_centers.shape[1]
        if self.x_train_mean.shape != (embedding_dim,):
            raise ValueError("x_train_mean shape does not match cluster centres")
        if self.x_train_std.shape != (embedding_dim,):
            raise ValueError("x_train_std shape does not match cluster centres")
        if not np.isfinite(self.mean_distance) or self.mean_distance <= 0:
            raise ValueError("global_mean_distance must be finite and positive")
        if np.any(self.x_train_std == 0):
            raise ValueError(
                "x_train_std contains zero entries; the bundled native scoring "
                "rule would divide by zero"
            )
        logger.info(
            "[ProvGuard] loaded native KMeans parameters: centres=%d, dim=%d, "
            "mean_distance=%.8g",
            self.cluster_centers.shape[0],
            embedding_dim,
            self.mean_distance,
        )

    def _validate_graph(self, graph) -> None:
        if "attr" not in graph.ndata:
            raise KeyError("ProvGuard graph is missing ndata['attr']")
        if "attr" not in graph.edata:
            raise KeyError("ProvGuard graph is missing edata['attr']")
        node_dim = int(graph.ndata["attr"].shape[1])
        edge_dim = int(graph.edata["attr"].shape[1])
        if node_dim != self.node_feature_dim or edge_dim != self.edge_feature_dim:
            raise ValueError(
                "ProvGuard graph feature dimensions do not match the checkpoint: "
                f"graph=({node_dim}, {edge_dim}), "
                f"expected=({self.node_feature_dim}, {self.edge_feature_dim})"
            )

    def _extract_embeddings(self, graph) -> np.ndarray:
        self._validate_graph(graph)
        with torch.inference_mode():
            device_graph = graph.to(self.device)
            embedding = self.model.embed(
                device_graph,
                device_graph.ndata["attr"],
                device_graph.edata["attr"],
            )
            result = embedding.detach().cpu().numpy().astype(np.float32, copy=False)
            del embedding, device_graph
        return result

    def _score_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        normalised = (embeddings - self.x_train_mean) / self.x_train_std
        centres = self.cluster_centers
        centre_norm = np.sum(centres * centres, axis=1, dtype=np.float32)
        scores = np.empty(normalised.shape[0], dtype=np.float32)
        for start in range(0, len(normalised), self.score_batch_size):
            stop = min(start + self.score_batch_size, len(normalised))
            batch = normalised[start:stop]
            squared = (
                np.sum(batch * batch, axis=1, keepdims=True, dtype=np.float32)
                + centre_norm[None, :]
                - 2.0 * np.matmul(batch, centres.T)
            )
            np.maximum(squared, 0.0, out=squared)
            scores[start:stop] = np.sqrt(squared.min(axis=1)) / self.mean_distance
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("ProvGuard produced non-finite anomaly scores")
        return scores

    def query(self, graph, node_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        scores_array = self._score_embeddings(self._extract_embeddings(graph))
        scores = {int(node): float(score) for node, score in enumerate(scores_array)}
        alerts = []
        if self.best_threshold is not None:
            alerts = [
                node for node, score in scores.items()
                if score >= float(self.best_threshold)
            ]
        return self._make_result(
            scores,
            threshold=self.best_threshold,
            native_alerts=alerts,
            node_indices=node_indices,
            metadata={
                "model": "ProvGuard",
                "dataset": self.dataset_name,
                "scoring": "nearest_kmeans_center_normalised_by_train_mean_distance",
            },
        )

    @staticmethod
    def _classification_metrics(
        labels: np.ndarray, scores: np.ndarray, threshold: float
    ) -> Dict[str, float]:
        predicted = scores >= float(threshold)
        positive = labels == 1
        tp = int(np.sum(predicted & positive))
        fp = int(np.sum(predicted & ~positive))
        fn = int(np.sum(~predicted & positive))
        tn = int(np.sum(~predicted & ~positive))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")
        return {
            "auc": auc,
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "tn": tn,
            "fn": fn,
            "tp": tp,
            "fp": fp,
            "threshold": float(threshold),
        }

    @staticmethod
    def _select_native_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        if len(thresholds) == 0:
            raise ValueError("cannot calibrate a threshold from constant labels/scores")
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        best_index = min(int(np.argmax(f1)), len(thresholds) - 1)
        return float(thresholds[best_index])

    def _cache_path(self) -> Path:
        return (
            PROVGUARD_ROOT
            / "data"
            / "eval_result"
            / f"native_evaluation_{self.dataset_name}_test{self.test_index}.pkl"
        )

    def _load_clean_cache(self, graph) -> Optional[Dict[str, Any]]:
        path = self._cache_path()
        if not path.is_file():
            return None
        with path.open("rb") as handle:
            cache = pickle.load(handle)
        expected = {
            "version": 1,
            "dataset": self.dataset_name,
            "test_index": self.test_index,
            "graph_nodes": int(graph.number_of_nodes()),
            "graph_edges": int(graph.number_of_edges()),
            "model_sha256": self.model_sha256,
            "params_sha256": self.params_sha256,
        }
        if any(cache.get(key) != value for key, value in expected.items()):
            logger.warning("[ProvGuard] ignored stale native cache: %s", path)
            return None
        scores = np.asarray(cache.get("scores"), dtype=np.float64)
        if len(scores) != graph.number_of_nodes() or not np.all(np.isfinite(scores)):
            return None
        return cache

    def _save_clean_cache(
        self,
        graph,
        scores: np.ndarray,
        threshold: float,
        metrics: Dict[str, float],
    ) -> None:
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = {
            "version": 1,
            "dataset": self.dataset_name,
            "test_index": self.test_index,
            "graph_nodes": int(graph.number_of_nodes()),
            "graph_edges": int(graph.number_of_edges()),
            "model_sha256": self.model_sha256,
            "params_sha256": self.params_sha256,
            "scores": np.asarray(scores, dtype=np.float32),
            "threshold": float(threshold),
            "initial_metrics": dict(metrics),
            "selection_rule": "maximum F1 on the native metadata labels",
        }
        with path.open("wb") as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("[ProvGuard] saved native clean evaluation cache: %s", path)

    def set_evaluation_context(self, test_index: int) -> None:
        self.test_index = int(test_index)
        n_test = int(self.metadata["n_test"])
        if not 0 <= self.test_index < n_test:
            raise IndexError(f"test_index {self.test_index} outside [0, {n_test - 1}]")
        if n_test != 1:
            raise NotImplementedError(
                "ProvGuard multi-graph native evaluation requires a cross-graph "
                "calibration cache. The current audited adapter supports the "
                "single-test-graph CADETS/THEIA setting only."
            )
        self.malicious_nodes = list(self.global_malicious_nodes)
        self.malicious_set = set(self.malicious_nodes)
        self.true_malicious_set = set(self.malicious_nodes)

    def initialize_tp_baseline(self, graph) -> Dict[str, Any]:
        self._validate_graph(graph)
        cache = self._load_clean_cache(graph)
        self.last_baseline_cache_hit = cache is not None
        if cache is not None:
            scores_array = np.asarray(cache["scores"], dtype=np.float64)
            logger.info(
                "[ProvGuard] reused clean native scores: physical_queries_saved=1, "
                "logical_queries_charged=1"
            )
        else:
            result = self.query(graph)
            scores_array = np.asarray(
                [result["all_scores"][node] for node in range(graph.number_of_nodes())],
                dtype=np.float64,
            )

        labels = np.zeros(len(scores_array), dtype=np.int8)
        self.valid_malicious = [
            node for node in self.malicious_nodes if 0 <= node < len(scores_array)
        ]
        if not self.valid_malicious:
            raise ValueError("ProvGuard native evaluation has no in-range malicious nodes")
        labels[self.valid_malicious] = 1
        self.true_malicious_set = set(self.valid_malicious)

        if self.threshold_source == "fixed":
            self.best_threshold = float(self.configured_threshold)
        elif cache is not None:
            self.best_threshold = float(cache["threshold"])
        else:
            self.best_threshold = self._select_native_threshold(labels, scores_array)

        self.initial_metrics = self._classification_metrics(
            labels, scores_array, self.best_threshold
        )
        self.initial_alert_nodes = {
            node for node, score in enumerate(scores_array)
            if score >= self.best_threshold
        }
        self.initial_tp_nodes = sorted(self.initial_alert_nodes)
        self.initial_true_tp_nodes = self.initial_alert_nodes & self.true_malicious_set
        self.true_tp_count = len(self.initial_true_tp_nodes)
        self.false_positive_count = len(
            self.initial_alert_nodes - self.true_malicious_set
        )
        self.escaped_nodes_history = set()

        if cache is None and self.threshold_source == "native":
            self._save_clean_cache(
                graph, scores_array, self.best_threshold, self.initial_metrics
            )

        logger.info(
            "[ProvGuard] native baseline: threshold=%.8g, alerts=%d, TP=%d, "
            "FP=%d, F1=%.6f, AUC=%.6f",
            self.best_threshold,
            len(self.initial_alert_nodes),
            self.true_tp_count,
            self.false_positive_count,
            self.initial_metrics["f1"],
            self.initial_metrics["auc"],
        )
        all_scores = {
            int(node): float(score) for node, score in enumerate(scores_array)
        }
        return {
            "initial_tp_nodes": self.initial_tp_nodes,
            "initial_true_tp_nodes": sorted(self.initial_true_tp_nodes),
            "threshold": self.best_threshold,
            "asr_denominator": self.true_tp_count,
            "all_scores": all_scores,
            "ground_truth": "metadata_native",
            "threshold_source": self.threshold_source,
        }

    def evaluate_native_scores(
        self, local_scores: Dict[int, float], *, threshold: Optional[float] = None
    ) -> Dict[str, float]:
        node_count = len(local_scores)
        labels = np.zeros(node_count, dtype=np.int8)
        valid = [node for node in self.malicious_nodes if 0 <= node < node_count]
        labels[valid] = 1
        scores = np.asarray(
            [local_scores.get(node, float("-inf")) for node in range(node_count)],
            dtype=np.float64,
        )
        operating_threshold = (
            self.best_threshold if threshold is None else float(threshold)
        )
        return self._classification_metrics(labels, scores, operating_threshold)

    def compute_asr_from_scores(
        self, scores: Dict[int, float], *, final: bool = False
    ) -> float:
        if self.true_tp_count == 0:
            return 0.0
        escaped = {
            node for node in self.initial_true_tp_nodes
            if scores.get(node, float("-inf")) < self.best_threshold
        }
        if final:
            return len(escaped) / self.true_tp_count
        self.escaped_nodes_history.update(escaped)
        return len(self.escaped_nodes_history) / self.true_tp_count
