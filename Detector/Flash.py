"""Native black-box adapter for FLASH.

The official CADETS notebook trains a sequence of GraphSAGE node-type
classifiers and reports a node only when *every* classifier misclassifies its
observable entity type.  This adapter preserves that rule while exposing a
continuous score for black-box optimisation:

    score(v) = min_m [max_{c != y_v} z_m(v, c) - z_m(v, y_v)]

Consequently ``score(v) > 0`` is equivalent to persistent misclassification by
all models (ties are disambiguated with the model's actual argmax).  The native
threshold is fixed at zero and never calibrated from malicious ground truth.
"""

from __future__ import annotations

import hashlib
import logging
import math
import pickle
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from Detector.Detector_Interface import BaseDetector
from Detector.Flash_State import FlashSemanticState
from MAGIC_main.action_schema import (
    build_observed_action_schema,
    get_action_schema,
)
from MAGIC_main.utils.loaddata import load_metadata


logger = logging.getLogger("attack.detector.flash")
ROOT = Path(__file__).resolve().parents[1]
FLASH_ROOT = ROOT / "Flash_IDS_main"


def _sha256_files(paths: List[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class FlashDetector(BaseDetector):
    """Entity-level FLASH adapter with the original ensemble semantics."""

    attack_state_extra_dim = 5
    attack_state_dim = 55
    score_direction = "higher_is_more_anomalous"
    detector_family = "persistent_node_type_misclassification"

    CLASS_COUNTS = {"cadets": 6, "theia": 5, "trace": 11}
    NATIVE_ENSEMBLE_SIZES = {"cadets": 22, "theia": 20, "trace": 10}

    def __init__(
        self,
        dataset_name: str = "cadets",
        device: str = "cuda:0",
        top_k: int = 50,
        checkpoint_path: Optional[str] = None,
        threshold_source: str = "native",
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if dataset_name not in self.CLASS_COUNTS:
            raise ValueError(f"FLASH does not support dataset {dataset_name!r}")
        if threshold_source not in {"native", "fixed"}:
            raise ValueError("threshold_source must be 'native' or 'fixed'")
        if threshold_source == "fixed" and threshold is None:
            raise ValueError("threshold_source='fixed' requires --threshold")
        native_threshold = 0.0 if threshold_source == "native" else float(threshold)
        super().__init__(
            dataset_name=dataset_name,
            device=device,
            top_k=top_k,
            threshold=native_threshold,
            **kwargs,
        )

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is unavailable; FLASH will run on CPU")
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.threshold_source = threshold_source
        self.configured_threshold = native_threshold
        self.test_index = 0
        self.semantic_state: Optional[FlashSemanticState] = None
        self.last_baseline_cache_hit = False

        self.metadata = load_metadata(dataset_name)
        self.global_malicious_nodes = [
            int(node) for node in self.metadata.get("malicious", [[], []])[0]
        ]
        self.malicious_nodes = list(self.global_malicious_nodes)
        self.malicious_set = set(self.malicious_nodes)
        self.true_malicious_set = set(self.malicious_nodes)
        self.valid_malicious: List[int] = []
        self.initial_true_tp_nodes: set[int] = set()
        self.false_positive_count = 0
        self.true_tp_count = 0
        self.initial_metrics: Dict[str, float] = {}

        self._load_models(checkpoint_path)
        logger.info(
            "[FLASH] initialized: dataset=%s, device=%s, models=%d, "
            "native_threshold=0, rule=persistent_misclassification",
            self.dataset_name,
            self.device,
            len(self.model_ensemble),
        )

    @staticmethod
    def _weight_number(path: Path) -> int:
        match = re.search(r"(\d+)_E3\.pth$", path.name)
        if not match:
            raise ValueError(f"unexpected FLASH checkpoint name: {path.name}")
        return int(match.group(1))

    def _load_models(self, checkpoint_path: Optional[str]) -> None:
        try:
            # Gensim < 4.4 imports scipy.linalg.triu, removed by newer SciPy.
            # NumPy's implementation is API-compatible for this use.
            import scipy.linalg
            if not hasattr(scipy.linalg, "triu"):
                scipy.linalg.triu = np.triu
            from gensim.models import Word2Vec
        except ImportError as exc:
            raise RuntimeError(
                "FLASH requires gensim in the STGAN environment"
            ) from exc

        class PureSAGEConv(torch.nn.Module):
            """PyG SAGEConv(mean, normalize=True) without compiled PyG wheels."""

            def __init__(self, in_channels: int, out_channels: int) -> None:
                super().__init__()
                self.lin_l = torch.nn.Linear(in_channels, out_channels, bias=True)
                self.lin_r = torch.nn.Linear(in_channels, out_channels, bias=False)

            def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
                src, dst = edge_index[0], edge_index[1]
                aggregate = torch.zeros_like(x)
                aggregate.index_add_(0, dst, x.index_select(0, src))
                degree = torch.bincount(dst, minlength=x.shape[0]).to(x.dtype)
                aggregate = aggregate / degree.clamp_min_(1).unsqueeze(1)
                output = self.lin_l(aggregate) + self.lin_r(x)
                return torch.nn.functional.normalize(output, p=2.0, dim=-1)

        class FlashGraphSAGE(torch.nn.Module):
            def __init__(self, in_channels: int, out_channels: int) -> None:
                super().__init__()
                self.conv1 = PureSAGEConv(in_channels, 32)
                self.conv2 = PureSAGEConv(32, out_channels)

            def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
                x = self.conv1(x, edge_index).relu()
                # The official notebook applies dropout only in training.
                x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
                return self.conv2(x, edge_index)

        weight_dir = FLASH_ROOT / "trained_weights" / self.dataset_name
        w2v_path = weight_dir / f"word2vec_{self.dataset_name}_E3.model"
        if not w2v_path.is_file():
            raise FileNotFoundError(f"FLASH Word2Vec model is missing: {w2v_path}")
        self.w2v_model = Word2Vec.load(str(w2v_path))

        if checkpoint_path:
            explicit = Path(checkpoint_path).expanduser()
            explicit = explicit if explicit.is_absolute() else ROOT / explicit
            if explicit.is_dir():
                weights = list(explicit.glob(f"lword2vec_gnn_{self.dataset_name}*_E3.pth"))
            elif explicit.is_file():
                weights = [explicit]
            else:
                raise FileNotFoundError(f"FLASH checkpoint does not exist: {explicit}")
        else:
            weights = list(weight_dir.glob(f"lword2vec_gnn_{self.dataset_name}*_E3.pth"))
        weights = sorted(weights, key=self._weight_number)
        expected = self.NATIVE_ENSEMBLE_SIZES[self.dataset_name]
        if len(weights) < expected:
            raise FileNotFoundError(
                f"FLASH native evaluation needs {expected} checkpoints for "
                f"{self.dataset_name}, found {len(weights)} in {weight_dir}"
            )
        # TRACE ships 20 training snapshots but its official evaluation cell uses 10.
        weights = weights[:expected]

        classes = self.CLASS_COUNTS[self.dataset_name]
        self.model_ensemble = []
        for path in weights:
            model = FlashGraphSAGE(30, classes)
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict) and "state_dict" in payload:
                payload = payload["state_dict"]
            model.load_state_dict(payload, strict=True)
            model.to(self.device).eval()
            self.model_ensemble.append(model)
        self.weight_paths = weights
        self.w2v_path = w2v_path
        self.model_signature = _sha256_files([w2v_path, *weights])

    def attach_semantic_state(self, state: FlashSemanticState) -> None:
        if state.dataset and state.dataset != self.dataset_name:
            raise ValueError(
                f"FLASH state dataset={state.dataset!r}, detector={self.dataset_name!r}"
            )
        if int(state.test_index) != int(self.test_index):
            raise ValueError(
                "FLASH state/test graph mismatch: "
                f"state=test{state.test_index}, requested=test{self.test_index}"
            )
        self.semantic_state = state
        offset = int(state.global_node_offset)
        upper = offset + int(state.original_node_count)
        self.malicious_nodes = sorted(
            {
                int(node) - offset
                for node in self.global_malicious_nodes
                if offset <= int(node) < upper
            }
        )
        self.malicious_set = set(self.malicious_nodes)
        self.true_malicious_set = set(self.malicious_nodes)
        logger.info(
            "[FLASH] semantic state attached: source=%s, test=%d, nodes=%d, "
            "native_edges=%s, localized_malicious=%d",
            state.source or "legacy-memory",
            state.test_index,
            state.original_node_count,
            (
                int(state.native_edge_index.shape[1])
                if state.native_edge_index is not None
                else "DGL-fallback"
            ),
            len(self.malicious_nodes),
        )

    def build_action_schema(self, graph, *, action_dim: int = 9):
        """Build legal MAGIC edits annotated with native CDM semantics."""
        if self.semantic_state is None:
            raise RuntimeError("attach a FLASH semantic state before schema creation")
        return build_observed_action_schema(
            graph,
            dataset=self.dataset_name,
            action_dim=action_dim,
            edge_type_names=self.semantic_state.magic_edge_type_names,
            node_type_names=self.semantic_state.node_type_names,
        )

    def encode_attack_state(self, base_state, *, target_node: int, **_kwargs):
        if self.semantic_state is None:
            raise RuntimeError("attach a FLASH semantic state before attacking")
        vocabulary = set(self.w2v_model.wv.key_to_index)
        semantic = torch.tensor(
            self.semantic_state.features(target_node, vocabulary),
            dtype=base_state.dtype,
            device=base_state.device,
        )
        return torch.cat((base_state, semantic))

    @staticmethod
    def _position_vector(position: int, dim: int = 30) -> np.ndarray:
        values = np.zeros(dim, dtype=np.float32)
        scale = np.exp(
            np.arange(0, dim, 2, dtype=np.float32)
            * (-math.log(10000.0) / dim)
        )
        values[0::2] = np.sin(float(position) * scale)
        values[1::2] = np.cos(float(position) * scale)
        return values

    def record_attack_modification(self, *, kind, action_type, graph, **kwargs):
        if kind != "edge" or self.semantic_state is None:
            return
        schema = get_action_schema(self.dataset_name, allow_observed_fallback=True)
        if schema is not None:
            action = schema.get(int(action_type)).edge_name
        else:
            edge_type = int(kwargs.get("edge_type", -1))
            if edge_type < 0:
                raise ValueError("FLASH edge modification is missing edge_type")
            action = self.semantic_state.event_name_for_edge_type(edge_type)

        src, dst = int(kwargs["src"]), int(kwargs["dst"])
        # FLASH's CDM graph is actor -> object. MAGIC reverses read/receive edges
        # to represent information flow, so recover the native direction using
        # the observable process node type.
        reverse_for_magic = any(
            token in action.upper() for token in ("READ", "RECV", "LOAD")
        )
        native_src, native_dst = (dst, src) if reverse_for_magic else (src, dst)

        vector = None
        if action in self.w2v_model.wv:
            vector = np.asarray(self.w2v_model.wv[action], dtype=np.float32)
            if self.semantic_state.token_counts is not None:
                positions = {
                    endpoint: (
                        self._position_vector(
                            int(self.semantic_state.token_counts[endpoint])
                        )
                        if self.semantic_state.document_counts is None
                        or int(self.semantic_state.document_counts[endpoint]) + 1 < 100_000
                        else None
                    )
                    for endpoint in {native_src, native_dst}
                }
                self.semantic_state.add_edge(
                    native_src,
                    native_dst,
                    action,
                    word_vector=vector,
                    positional_vectors=positions,
                )
                return
        self.semantic_state.add_edge(native_src, native_dst, action)

    def _validate_graph(self, graph) -> None:
        if self.semantic_state is None:
            raise RuntimeError(
                "FLASH requires a semantic state. Run "
                "'python Flash_IDS_main/flash_cadets.py prepare' first and pass "
                "the resulting file with --flash-semantic-state."
            )
        self.semantic_state.validate_for_graph(graph, self.dataset_name)
        if "type" not in graph.ndata:
            raise KeyError("FLASH requires graph.ndata['type'] entity labels")
        labels = self.semantic_state.labels_for_graph(graph)
        if labels.size != graph.number_of_nodes():
            raise ValueError("FLASH node label count does not match graph size")
        if int(labels.min()) < 0:
            raise ValueError("FLASH node labels must be non-negative")
        class_count = self.CLASS_COUNTS[self.dataset_name]
        if int(labels.max()) > class_count:
            raise ValueError("FLASH node type IDs do not match the checkpoint classes")
        unavailable = int(np.sum(labels >= class_count))
        if unavailable:
            logger.warning(
                "[FLASH] %d nodes use official label %d, while the %s "
                "checkpoint exposes classes 0..%d; matching the official "
                "argmax rule, these nodes are persistently misclassified",
                unavailable,
                class_count,
                self.dataset_name,
                class_count - 1,
            )

    def _edge_index(self, graph) -> torch.Tensor:
        native = self.semantic_state.edge_index() if self.semantic_state else None
        if native is not None:
            return torch.as_tensor(native, dtype=torch.long, device=self.device)
        src, dst = graph.edges()
        return torch.stack((src, dst)).long().to(self.device)

    @staticmethod
    def _margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classes = int(logits.shape[1])
        valid = labels.lt(classes)
        margin = torch.empty(len(labels), dtype=logits.dtype, device=logits.device)
        if bool(valid.any()):
            valid_logits = logits[valid]
            valid_labels = labels[valid]
            true_logits = valid_logits.gather(1, valid_labels[:, None]).squeeze(1)
            competitors = valid_logits.clone()
            competitors.scatter_(1, valid_labels[:, None], float("-inf"))
            valid_margin = competitors.max(dim=1).values - true_logits
            pred = valid_logits.argmax(dim=1)
            # Preserve the exact hard decision at the zero boundary.
            tied = valid_margin == 0
            eps = torch.finfo(valid_margin.dtype).eps
            valid_margin = torch.where(
                tied & pred.eq(valid_labels), -eps, valid_margin
            )
            valid_margin = torch.where(
                tied & pred.ne(valid_labels), eps, valid_margin
            )
            margin[valid] = valid_margin
        if bool((~valid).any()):
            # The official THEIA/TRACE notebooks compare argmax directly with
            # labels that may equal the output width. Such a label can never be
            # predicted and is therefore always anomalous. Use a finite,
            # positive confidence surrogate so metrics and caches remain
            # numeric while preserving that exact hard decision at threshold 0.
            unavailable_logits = logits[~valid]
            top_two = unavailable_logits.topk(k=2, dim=1).values
            confidence_gap = (top_two[:, 0] - top_two[:, 1]).abs()
            margin[~valid] = 1.0 + confidence_gap
        return margin

    def _score_graph(self, graph) -> tuple[np.ndarray, np.ndarray]:
        self._validate_graph(graph)
        features = torch.as_tensor(
            self.semantic_state.embeddings(self.w2v_model),
            dtype=torch.float32,
            device=self.device,
        )
        labels = torch.as_tensor(
            self.semantic_state.labels_for_graph(graph),
            dtype=torch.long,
            device=self.device,
        )
        edge_index = self._edge_index(graph)
        persistent = torch.ones(len(labels), dtype=torch.bool, device=self.device)
        score = torch.full(
            (len(labels),), float("inf"), dtype=torch.float32, device=self.device
        )
        with torch.inference_mode():
            for model in self.model_ensemble:
                logits = model(features, edge_index)
                model_margin = self._margin(logits, labels)
                score = torch.minimum(score, model_margin)
                persistent &= logits.argmax(dim=1).ne(labels)
                del logits, model_margin
        scores = score.cpu().numpy().astype(np.float32, copy=False)
        alerts = persistent.nonzero(as_tuple=False).view(-1).cpu().numpy().astype(np.int64)
        del features, labels, edge_index, score, persistent
        return scores, alerts

    def query(self, graph, node_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        scores_array, native_alerts = self._score_graph(graph)
        scores = {int(node): float(value) for node, value in enumerate(scores_array)}
        threshold = float(self.best_threshold)
        if self.threshold_source == "fixed":
            native_alerts = np.flatnonzero(scores_array > threshold)
        return self._make_result(
            scores,
            threshold=threshold,
            native_alerts=native_alerts.tolist(),
            node_indices=node_indices,
            metadata={
                "model": "FLASH",
                "dataset": self.dataset_name,
                "scoring": "minimum_true_class_logit_margin_across_native_ensemble",
                "alert_rule": "misclassified_by_every_native_model",
                "ensemble_size": len(self.model_ensemble),
            },
        )

    @staticmethod
    def _classification_metrics(
        labels: np.ndarray, scores: np.ndarray, threshold: float
    ) -> Dict[str, float]:
        predicted = scores > float(threshold)
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

    def set_evaluation_context(self, test_index: int) -> None:
        self.test_index = int(test_index)
        n_test = int(self.metadata["n_test"])
        if not 0 <= self.test_index < n_test:
            raise IndexError(f"test_index {test_index} outside [0, {n_test - 1}]")
        native_test_index = {"cadets": 0, "theia": 0, "trace": 4}[
            self.dataset_name
        ]
        if self.test_index != native_test_index:
            raise ValueError(
                f"FLASH {self.dataset_name} native evaluation uses test"
                f"{native_test_index}; requested test{self.test_index}. "
                "This restriction keeps the attack aligned with the official "
                "FLASH evaluation split."
            )
        self.malicious_nodes = list(self.global_malicious_nodes)
        self.malicious_set = set(self.malicious_nodes)
        self.true_malicious_set = set(self.malicious_nodes)

    def _cache_path(self) -> Path:
        return (
            FLASH_ROOT
            / "eval_result"
            / f"native_evaluation_{self.dataset_name}_test{self.test_index}.pkl"
        )

    def _state_signature(self) -> Dict[str, Any]:
        if self.semantic_state is None:
            return {}
        source = Path(self.semantic_state.source) if self.semantic_state.source else None
        return {
            "semantic_state": str(source.resolve()) if source and source.exists() else "memory",
            "semantic_state_size": source.stat().st_size if source and source.exists() else None,
            "semantic_state_mtime_ns": source.stat().st_mtime_ns if source and source.exists() else None,
        }

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
            "model_signature": self.model_signature,
            **self._state_signature(),
        }
        if any(cache.get(key) != value for key, value in expected.items()):
            logger.warning("[FLASH] ignored stale native cache: %s", path)
            return None
        scores = np.asarray(cache.get("scores"), dtype=np.float64)
        if len(scores) != graph.number_of_nodes() or not np.all(np.isfinite(scores)):
            return None
        return cache

    def _save_clean_cache(
        self,
        graph,
        scores: np.ndarray,
        native_alerts: np.ndarray,
        metrics: Dict[str, float],
    ) -> None:
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "dataset": self.dataset_name,
            "test_index": self.test_index,
            "graph_nodes": int(graph.number_of_nodes()),
            "graph_edges": int(graph.number_of_edges()),
            "model_signature": self.model_signature,
            **self._state_signature(),
            "scores": np.asarray(scores, dtype=np.float32),
            "native_alerts": np.asarray(native_alerts, dtype=np.int64),
            "threshold": float(self.best_threshold),
            "initial_metrics": dict(metrics),
            "selection_rule": "persistent node-type misclassification; threshold fixed at zero",
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("[FLASH] saved native clean evaluation cache: %s", path)

    def initialize_tp_baseline(self, graph) -> Dict[str, Any]:
        self._validate_graph(graph)
        cache = self._load_clean_cache(graph)
        self.last_baseline_cache_hit = cache is not None
        if cache is None:
            scores_array, native_alerts = self._score_graph(graph)
        else:
            scores_array = np.asarray(cache["scores"], dtype=np.float64)
            native_alerts = np.asarray(cache["native_alerts"], dtype=np.int64)
            logger.info(
                "[FLASH] reused clean native scores: physical_queries_saved=1, "
                "logical_queries_charged=1"
            )

        labels = np.zeros(len(scores_array), dtype=np.int8)
        self.valid_malicious = [
            node for node in self.malicious_nodes if 0 <= node < len(scores_array)
        ]
        if not self.valid_malicious:
            raise ValueError("FLASH evaluation has no in-range malicious metadata nodes")
        labels[self.valid_malicious] = 1
        self.true_malicious_set = set(self.valid_malicious)
        self.best_threshold = float(self.configured_threshold)
        if self.threshold_source == "fixed":
            native_alerts = np.flatnonzero(scores_array > self.best_threshold)

        self.initial_metrics = self._classification_metrics(
            labels, scores_array, self.best_threshold
        )
        self.initial_alert_nodes = set(int(node) for node in native_alerts)
        self.initial_tp_nodes = sorted(self.initial_alert_nodes)
        self.initial_true_tp_nodes = self.initial_alert_nodes & self.true_malicious_set
        self.true_tp_count = len(self.initial_true_tp_nodes)
        self.false_positive_count = len(self.initial_alert_nodes - self.true_malicious_set)
        self.escaped_nodes_history = set()

        if cache is None:
            self._save_clean_cache(
                graph, scores_array, native_alerts, self.initial_metrics
            )
        logger.info(
            "[FLASH] native baseline: threshold=%.1f, alerts=%d, TP=%d, FP=%d, "
            "F1=%.6f, AUC=%.6f",
            self.best_threshold,
            len(self.initial_alert_nodes),
            self.true_tp_count,
            self.false_positive_count,
            self.initial_metrics["f1"],
            self.initial_metrics["auc"],
        )
        return {
            "initial_tp_nodes": self.initial_tp_nodes,
            "initial_true_tp_nodes": sorted(self.initial_true_tp_nodes),
            "threshold": self.best_threshold,
            "asr_denominator": self.true_tp_count,
            "all_scores": {
                int(node): float(score) for node, score in enumerate(scores_array)
            },
            "ground_truth": "metadata_evaluation_only",
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
        operating_threshold = self.best_threshold if threshold is None else float(threshold)
        return self._classification_metrics(labels, scores, operating_threshold)

    def compute_asr_from_scores(
        self, scores: Dict[int, float], *, final: bool = False
    ) -> float:
        if self.true_tp_count == 0:
            return 0.0
        escaped = {
            node
            for node in self.initial_true_tp_nodes
            if scores.get(node, float("-inf")) <= self.best_threshold
        }
        if final:
            return len(escaped) / self.true_tp_count
        self.escaped_nodes_history.update(escaped)
        return len(self.escaped_nodes_history) / self.true_tp_count
