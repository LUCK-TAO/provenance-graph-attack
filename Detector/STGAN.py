"""Native black-box adapter for the released STGAN E3 evaluators.

The released evaluator applies 50 GAT/TGN/self-attention snapshots to the
native CDM multigraph. A node is alerted only when every snapshot
misclassifies its observable entity type. This adapter preserves that hard
rule and exposes a continuous, label-free feedback score:

    score(v) = min [max competing logit - observable-type logit]

Thus ``score(v) > 0`` is equivalent to persistent misclassification. The
observable entity type is detector input, not malicious ground truth.

The attack-facing metrics deliberately use the raw node decisions before the
released evaluator's ground-truth-aware two-hop tolerance.  This matches the
FLASH adapter and keeps target selection, DASR, and before/after detector
metrics on one common black-box evaluation scope.  The released two-hop
metrics remain available as a separate reproduction-only audit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from Detector.Detector_Interface import BaseDetector
from Detector.STGAN_State import STGANSemanticState
from Detector.STGAN_WordVectors import STGANWordVectors
from MAGIC_main.action_schema import build_observed_action_schema, get_action_schema


logger = logging.getLogger("attack.detector.stgan")
ROOT = Path(__file__).resolve().parents[1]
STGAN_ROOT = ROOT / "STGAN_master"

# Import through the repository-qualified package name.  Several detector
# implementations expose a top-level package named ``model``; importing
# ``model.GAT`` therefore depends on detector initialization order and fails
# when another detector has already populated ``sys.modules['model']``.
from STGAN_master.model.GAT import GAT  # noqa: E402
from STGAN_master.model.MultiHeadSelfAttention import MultiHeadSelfAttentionModel  # noqa: E402
from STGAN_master.model.TGN import TemporalGraphNetwork  # noqa: E402


def _file_signature(paths: List[Path]) -> str:
    """Cheap cache signature; checkpoint contents are not read at startup."""
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _load_released_state(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    """Load checkpoints saved by the PyG version used in the release.

    Old homogeneous ``GATConv`` stored one ``lin.weight``. PyG 2.1 exposes the
    same transform through the aliased ``lin_src``/``lin_dst`` names. Copying
    that matrix to both names is the exact architecture-preserving migration.
    """
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    translated = dict(payload)
    for key in list(payload):
        if not key.endswith(".lin.weight"):
            continue
        stem = key[: -len("lin.weight")]
        translated[f"{stem}lin_src.weight"] = payload[key]
        translated[f"{stem}lin_dst.weight"] = payload[key]
        del translated[key]
    model.load_state_dict(translated, strict=True)


class STGANDetector(BaseDetector):
    """Released STGAN persistent node-type-misclassification detector."""

    score_direction = "higher_is_more_anomalous"
    detector_family = "persistent_node_type_misclassification"
    attack_state_extra_dim = 0
    attack_state_dim = 50
    NATIVE_BATCH_SIZE = 26000
    CLASS_COUNTS = {"cadets": 6, "theia": 6, "trace": 12}
    NATIVE_SNAPSHOT_COUNTS = {"cadets": 50, "theia": 20, "trace": 50}
    TEST_INDICES = {"cadets": (0,), "theia": (0,), "trace": (4,)}

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
            raise ValueError(
                "The released STGAN implementation has been reproduced for "
                "CADETS, THEIA, and TRACE."
            )
        if threshold_source not in {"native", "fixed"}:
            raise ValueError("threshold_source must be 'native' or 'fixed'")
        if threshold_source == "fixed" and threshold is None:
            raise ValueError("threshold_source='fixed' requires --threshold")
        configured_threshold = 0.0 if threshold_source == "native" else float(threshold)
        super().__init__(
            dataset_name=dataset_name,
            device=device,
            top_k=top_k,
            threshold=configured_threshold,
            **kwargs,
        )
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is unavailable; STGAN will run on CPU")
            requested = torch.device("cpu")
        self.device = requested
        self.threshold_source = threshold_source
        self.configured_threshold = configured_threshold
        self.test_index = int(self.TEST_INDICES[dataset_name][0])
        self.semantic_state: Optional[STGANSemanticState] = None
        self.last_baseline_cache_hit = False

        # STGAN's released evaluator defines malicious truth with its
        # own UUID annotation file, not MAGIC metadata.  Keep that detector-
        # native truth for attack evaluation and target-set construction.
        uuid_path = (
            ROOT / "MAGIC_main" / "data" / dataset_name
            / f"uuid_to_index_test{self.test_index}.json"
        )
        gt_path = STGAN_ROOT / "data" / dataset_name / f"{dataset_name}.json"
        if not uuid_path.is_file():
            raise FileNotFoundError(
                f"STGAN UUID map is missing: {uuid_path}. Prepare the native "
                f"{dataset_name.upper()} state before starting an attack."
            )
        if not gt_path.is_file():
            raise FileNotFoundError(f"STGAN ground truth is missing: {gt_path}")
        uuid_map = json.loads(uuid_path.read_text(encoding="utf-8"))
        gt_uuids = set(json.loads(gt_path.read_text(encoding="utf-8")))
        self.global_malicious_nodes = sorted(
            int(uuid_map[node_uuid])
            for node_uuid in gt_uuids
            if node_uuid in uuid_map
        )
        self.ground_truth_source = "stgan_official_uuid_annotations"
        self.malicious_nodes = list(self.global_malicious_nodes)
        self.malicious_set = set(self.malicious_nodes)
        self.true_malicious_set = set(self.malicious_nodes)
        self.valid_malicious: List[int] = []
        self.initial_true_tp_nodes: set[int] = set()
        self.false_positive_count = 0
        self.true_tp_count = 0
        self.initial_metrics: Dict[str, float] = {}
        self.raw_initial_metrics: Dict[str, float] = {}
        self.released_initial_metrics: Dict[str, float] = {}
        # Frozen detector weights never change during an attack.  Keep every
        # released snapshot resident after the first physical query instead of
        # reading 150 checkpoint files again for every graph modification.
        self.snapshot_models = None
        self._last_attack_timestamp: Optional[float] = None

        self._discover_native_assets(checkpoint_path)
        logger.info(
            "[STGAN] initialized: dataset=%s, device=%s, snapshots=%d, "
            "threshold=%.1f, rule=persistent_misclassification",
            dataset_name,
            self.device,
            len(self.snapshot_paths),
            self.best_threshold,
        )

    @staticmethod
    def _weight_number(path: Path) -> int:
        match = re.search(r"(\d+)\.pth$", path.name)
        if not match:
            raise ValueError(f"unexpected STGAN checkpoint name: {path.name}")
        return int(match.group(1))

    def _discover_native_assets(self, checkpoint_path: Optional[str]) -> None:
        weight_dir = (
            Path(checkpoint_path).expanduser()
            if checkpoint_path
            else STGAN_ROOT / "trained_weights" / self.dataset_name
        )
        if not weight_dir.is_absolute():
            weight_dir = ROOT / weight_dir
        if not weight_dir.is_dir():
            raise FileNotFoundError(f"STGAN checkpoint directory is missing: {weight_dir}")
        groups = []
        expected_snapshots = self.NATIVE_SNAPSHOT_COUNTS[self.dataset_name]
        for prefix in ("gat", "tgn", "selfattention"):
            paths = sorted(weight_dir.glob(f"{prefix}*.pth"), key=self._weight_number)
            if len(paths) != expected_snapshots:
                raise FileNotFoundError(
                    f"STGAN native {self.dataset_name.upper()} evaluation requires "
                    f"{expected_snapshots} {prefix} "
                    f"checkpoints; found {len(paths)} in {weight_dir}"
                )
            groups.append(paths)
        self.snapshot_paths = list(zip(*groups))
        self.source_w2v_path = (
            STGAN_ROOT / "train" / f"word2vec_{self.dataset_name}.model"
        )
        self.w2v_path = (
            STGAN_ROOT / "train" / f"word2vec_{self.dataset_name}_compat.npz"
        )
        if not self.w2v_path.is_file():
            raise FileNotFoundError(
                f"STGAN neutral Word2Vec file is missing: {self.w2v_path}. "
                "Run STGAN_master/convert_word2vec.py once."
            )
        all_paths = [self.w2v_path]
        if self.source_w2v_path.is_file():
            all_paths.append(self.source_w2v_path)
        for triplet in self.snapshot_paths:
            all_paths.extend(triplet)
        self.model_signature = _file_signature(all_paths)

        # Clean embeddings are precomputed in the semantic state. Neutral word
        # vectors are retained only to update endpoints after a legal edit.
        self.word_vectors = STGANWordVectors(self.w2v_path)

    def set_evaluation_context(self, test_index: int) -> None:
        allowed = self.TEST_INDICES[self.dataset_name]
        if int(test_index) not in allowed:
            raise ValueError(
                f"Released STGAN {self.dataset_name.upper()} evaluation uses "
                f"test indices {allowed}"
            )
        self.test_index = int(test_index)

    def attach_semantic_state(self, state: STGANSemanticState) -> None:
        if state.dataset != self.dataset_name or int(state.test_index) != self.test_index:
            raise ValueError("STGAN semantic state does not match dataset/test split")
        self.semantic_state = state
        edge_time = state.edge_time()
        self._last_attack_timestamp = (
            float(np.max(edge_time)) if edge_time.size else -1.0
        )
        offset = int(state.global_node_offset)
        upper = offset + int(state.original_node_count)
        self.malicious_nodes = sorted(
            int(node) - offset
            for node in self.global_malicious_nodes
            if offset <= int(node) < upper
        )
        self.malicious_set = set(self.malicious_nodes)
        self.true_malicious_set = set(self.malicious_nodes)
        logger.info(
            "[STGAN] native state attached: source=%s, nodes=%d, native_edges=%d",
            state.source or "memory",
            state.original_node_count,
            state.native_edge_index.shape[1],
        )

    def build_action_schema(self, graph, *, action_dim: int = 9):
        if self.semantic_state is None:
            raise RuntimeError("attach an STGAN semantic state before schema creation")
        return build_observed_action_schema(
            graph,
            dataset=self.dataset_name,
            action_dim=action_dim,
            edge_type_names=self.semantic_state.magic_edge_type_names,
            node_type_names=self.semantic_state.node_type_names,
        )

    def record_attack_modification(self, *, kind, action_type, graph, **kwargs):
        if kind != "edge" or self.semantic_state is None:
            return
        schema = get_action_schema(self.dataset_name, allow_observed_fallback=True)
        if schema is not None:
            action = schema.get(int(action_type)).edge_name
        else:
            # THEIA/TRACE use a graph-observed runtime schema rather than the
            # static CADETS profile.  The resolved native edge type is passed
            # by the candidate selector and its audited CDM name is stored in
            # the attached semantic state.
            edge_type = int(kwargs.get("edge_type", -1))
            if edge_type not in self.semantic_state.magic_edge_type_names:
                raise ValueError(
                    "STGAN edge modification lacks an audited native event "
                    f"name for edge type {edge_type}"
                )
            action = self.semantic_state.magic_edge_type_names[edge_type]
        src, dst = int(kwargs["src"]), int(kwargs["dst"])
        reverse = any(token in action.upper() for token in ("READ", "RECV", "LOAD"))
        native_src, native_dst = (dst, src) if reverse else (src, dst)
        vector = (
            np.asarray(self.word_vectors[action], dtype=np.float32)
            if action in self.word_vectors
            else None
        )
        # The previous implementation concatenated and scanned more than two
        # million timestamps for every primitive edit. Attack timestamps are
        # strictly increasing, so maintaining the last value is equivalent.
        if self._last_attack_timestamp is None:
            edge_time = self.semantic_state.edge_time()
            self._last_attack_timestamp = (
                float(np.max(edge_time)) if edge_time.size else -1.0
            )
        self._last_attack_timestamp += 1.0
        timestamp = self._last_attack_timestamp
        self.semantic_state.add_edge(
            native_src,
            native_dst,
            action,
            word_vector=vector,
            timestamp=timestamp,
        )

    def _validate_graph(self, graph) -> None:
        if self.semantic_state is None:
            raise RuntimeError(
                "STGAN requires a native semantic state. Run "
                f"'python STGAN_master/stgan_{self.dataset_name}.py prepare' "
                "and pass it "
                "with --stgan-semantic-state."
            )
        self.semantic_state.validate_for_graph(graph, self.dataset_name, self.test_index)

    @staticmethod
    def _margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        true_logits = logits.gather(1, labels[:, None]).squeeze(1)
        competitors = logits.clone()
        competitors.scatter_(1, labels[:, None], float("-inf"))
        margin = competitors.max(dim=1).values - true_logits
        pred = logits.argmax(dim=1)
        tied = margin == 0
        eps = torch.finfo(margin.dtype).eps
        margin = torch.where(tied & pred.eq(labels), -eps, margin)
        return torch.where(tied & pred.ne(labels), eps, margin)

    def _new_models(self):
        classes = self.CLASS_COUNTS[self.dataset_name]
        gat = GAT(in_channel=classes, out_channel=30).to(self.device)
        tgn = TemporalGraphNetwork(in_channels=60, out_channels=32).to(self.device)
        attention = MultiHeadSelfAttentionModel(
            in_channels=32, out_channels=classes, num_classes=classes, num_heads=4
        ).to(self.device)
        return gat, tgn, attention

    def _ensure_snapshot_models(self):
        """Load all frozen snapshot triplets once and reuse them for the run."""
        if self.snapshot_models is not None:
            return self.snapshot_models
        logger.info(
            "[STGAN] loading %d frozen detector snapshots once",
            len(self.snapshot_paths),
        )
        models = []
        for gat_path, tgn_path, attention_path in self.snapshot_paths:
            gat, tgn, attention = self._new_models()
            _load_released_state(gat, gat_path, self.device)
            _load_released_state(tgn, tgn_path, self.device)
            _load_released_state(attention, attention_path, self.device)
            gat.eval()
            tgn.eval()
            attention.eval()
            models.append((gat, tgn, attention))
        self.snapshot_models = models
        logger.info(
            "[STGAN] frozen detector snapshots loaded and ready for reuse"
        )
        return self.snapshot_models

    def _score_graph(self, graph) -> tuple[np.ndarray, np.ndarray]:
        self._validate_graph(graph)
        query_started = time.perf_counter()
        state = self.semantic_state
        node_count = int(state.original_node_count)
        labels = np.asarray(state.model_labels, dtype=np.int64)
        classes = self.CLASS_COUNTS[self.dataset_name]
        one_hot = np.eye(classes, dtype=np.float32)[labels]
        pyg_graph = Data(
            x=torch.from_numpy(one_hot),
            y=torch.from_numpy(labels),
            edge_index=torch.as_tensor(state.edge_index(), dtype=torch.long),
            edge_time=torch.as_tensor(state.edge_time(), dtype=torch.float32),
        )
        pyg_graph.n_id = torch.arange(node_count, dtype=torch.long)
        embeddings = torch.as_tensor(state.base_embeddings, dtype=torch.float32)
        scores = np.full(node_count, np.inf, dtype=np.float32)
        persistent = np.ones(node_count, dtype=bool)
        logger.debug(
            "[STGAN] detector query started: nodes=%d, edges=%d, snapshots=%d",
            node_count,
            int(pyg_graph.edge_index.shape[1]),
            len(self.snapshot_paths),
        )
        snapshot_models = self._ensure_snapshot_models()
        # The graph and batching policy are identical for all frozen snapshots.
        # Reusing the loader avoids rebuilding its configuration 50 times.
        loader = NeighborLoader(
            pyg_graph,
            num_neighbors=[-1, -1],
            batch_size=self.NATIVE_BATCH_SIZE,
            shuffle=False,
        )

        # Sampling and host-to-device transfer depend only on the graph, not
        # on the frozen snapshot. Keep the native mathematics identical while
        # doing those operations once per subgraph instead of 50 times.
        inference_started = time.perf_counter()
        with torch.inference_mode():
            for batch_index, subgraph in enumerate(loader):
                node_ids = subgraph.n_id.cpu().numpy().astype(np.int64, copy=False)
                x = subgraph.x.to(self.device)
                edge_index = subgraph.edge_index.to(self.device)
                edge_time = subgraph.edge_time.to(self.device)
                y = subgraph.y.to(self.device)
                text = embeddings.index_select(0, subgraph.n_id).to(self.device)
                for gat, tgn, attention in snapshot_models:
                    output = gat(x, edge_index)
                    output = torch.cat((output, text), dim=1)
                    output = tgn(output, edge_index, edge_time)
                    logits = attention(output.unsqueeze(1))
                    margin = self._margin(logits, y).cpu().numpy()
                    np.minimum.at(scores, node_ids, margin)
                    correct = logits.argmax(dim=1).eq(y).cpu().numpy()
                    persistent[node_ids[correct]] = False
                    del output, logits, margin, correct
                logger.debug(
                    "[STGAN] native batch %d completed across %d snapshots",
                    batch_index + 1,
                    len(snapshot_models),
                )

        if not np.all(np.isfinite(scores)):
            missing = int(np.sum(~np.isfinite(scores)))
            raise RuntimeError(f"STGAN native loader did not score {missing} nodes")
        alerts = np.flatnonzero(persistent).astype(np.int64)
        query_elapsed = time.perf_counter() - query_started
        logger.info(
            "[STGANQueryTiming] total=%.2fs, inference=%.2fs, alerts=%d",
            query_elapsed,
            time.perf_counter() - inference_started,
            len(alerts),
        )
        return scores, alerts

    def query(self, graph, node_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        scores_array, native_alerts = self._score_graph(graph)
        if self.threshold_source == "fixed":
            native_alerts = np.flatnonzero(scores_array > self.best_threshold)
        return self._make_result(
            {int(i): float(value) for i, value in enumerate(scores_array)},
            threshold=float(self.best_threshold),
            native_alerts=native_alerts.tolist(),
            node_indices=node_indices,
            metadata={
                "model": "STGAN",
                "dataset": self.dataset_name,
                "ensemble_size": len(self.snapshot_paths),
                "scoring": "minimum_true_class_logit_margin_across_native_snapshots",
                "alert_rule": "misclassified_by_every_native_snapshot",
            },
        )

    @staticmethod
    def _classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float):
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
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tn": tn,
            "fn": fn,
            "tp": tp,
            "fp": fp,
            "threshold": float(threshold),
        }

    @staticmethod
    def _two_hop_mask(seeds: set[int], edge_index: np.ndarray, node_count: int):
        reached = np.zeros(node_count, dtype=bool)
        frontier = np.zeros(node_count, dtype=bool)
        if seeds:
            frontier[np.fromiter(seeds, dtype=np.int64)] = True
        src, dst = edge_index[0], edge_index[1]
        for _ in range(2):
            touching = frontier[src] | frontier[dst]
            next_frontier = np.zeros(node_count, dtype=bool)
            next_frontier[src[touching]] = True
            next_frontier[dst[touching]] = True
            reached |= next_frontier
            frontier = next_frontier
        return reached

    def _released_code_metrics(self, scores: np.ndarray, alerts: set[int]):
        """Apply the two-hop helper used by the released STGAN evaluator.

        Ground truth enters only this offline reporting function. It never
        changes detector scores, alerts, threshold, attack targets, or reward.
        """
        ground_truth = {
            int(node) for node in self.malicious_nodes
            if 0 <= int(node) < int(self.semantic_state.original_node_count)
        }
        alerts = set(int(node) for node in alerts)
        tp = alerts & ground_truth
        fp = alerts - ground_truth
        fn = ground_truth - alerts
        node_count = int(self.semantic_state.original_node_count)
        gt_neighborhood = self._two_hop_mask(
            ground_truth, self.semantic_state.native_edge_index, node_count
        )
        tp_neighborhood = self._two_hop_mask(
            tp, self.semantic_state.native_edge_index, node_count
        )
        filtered_fp = {node for node in fp if not gt_neighborhood[node]}
        recovered_fn = {node for node in fn if tp_neighborhood[node]}
        tp_count = len(tp | recovered_fn)
        fp_count = len(filtered_fp)
        fn_count = len(fn - recovered_fn)
        tn_count = node_count - len(alerts | ground_truth)
        precision = tp_count / (tp_count + fp_count) if tp_count + fp_count else 0.0
        recall = tp_count / (tp_count + fn_count) if tp_count + fn_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        labels = np.zeros(node_count, dtype=np.int8)
        if ground_truth:
            labels[np.fromiter(ground_truth, dtype=np.int64)] = 1
        # The released helper does not define AUC. Keep the continuous-margin
        # AUC as a separately scoped compatibility value for the common report.
        margin_auc = float(roc_auc_score(labels, scores))
        return {
            # The released CADETS helper reports no AUC. Do not present the
            # continuous feedback diagnostic as an official detector metric.
            "auc": float("nan"),
            "margin_auc_diagnostic": margin_auc,
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "tn": int(tn_count),
            "fn": int(fn_count),
            "tp": int(tp_count),
            "fp": int(fp_count),
            "threshold": float(self.best_threshold),
            "metric_scope": (
                f"released_{self.dataset_name}_two_hop_postprocessing"
            ),
            "auc_scope": (
                f"not_defined_by_released_{self.dataset_name}_evaluator"
            ),
        }

    def evaluate_released_code_scores(
        self, local_scores: Dict[int, float], *, threshold=None
    ) -> Dict[str, float]:
        """Return the released two-hop metrics for offline reproduction only.

        This method is intentionally separate from ``evaluate_native_scores``
        so that the common attack pipeline never mixes ground-truth-aware
        post-processing into its target set or before/after metrics.
        """
        scores = np.asarray(
            [local_scores.get(i, float("-inf")) for i in range(len(local_scores))]
        )
        operating = self.best_threshold if threshold is None else float(threshold)
        alerts = set(np.flatnonzero(scores > operating).tolist())
        return self._released_code_metrics(scores, alerts)

    def _cache_path(self) -> Path:
        return (
            STGAN_ROOT / "eval_result"
            / f"native_evaluation_{self.dataset_name}_test{self.test_index}.pkl"
        )

    def _state_signature(self) -> Dict[str, Any]:
        source = Path(self.semantic_state.source) if self.semantic_state and self.semantic_state.source else None
        return {
            "semantic_state": str(source.resolve()) if source and source.exists() else "memory",
            "semantic_state_size": source.stat().st_size if source and source.exists() else None,
            "semantic_state_mtime_ns": source.stat().st_mtime_ns if source and source.exists() else None,
        }

    def _load_clean_cache(self, graph):
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
            "model_signature": self.model_signature,
            **self._state_signature(),
        }
        if any(cache.get(key) != value for key, value in expected.items()):
            logger.warning("[STGAN] ignored stale native cache: %s", path)
            return None
        scores = np.asarray(cache.get("scores"), dtype=np.float32)
        if len(scores) != graph.number_of_nodes() or not np.all(np.isfinite(scores)):
            return None
        return cache

    def _save_clean_cache(self, graph, scores, alerts, metrics):
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "dataset": self.dataset_name,
            "test_index": self.test_index,
            "graph_nodes": int(graph.number_of_nodes()),
            "model_signature": self.model_signature,
            **self._state_signature(),
            "scores": np.asarray(scores, dtype=np.float32),
            "native_alerts": np.asarray(alerts, dtype=np.int64),
            "threshold": float(self.best_threshold),
            "initial_metrics": dict(metrics),
            "selection_rule": "persistent node-type misclassification; threshold zero",
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("[STGAN] saved native clean evaluation cache: %s", path)

    def initialize_tp_baseline(self, graph) -> Dict[str, Any]:
        self._validate_graph(graph)
        cache = self._load_clean_cache(graph)
        self.last_baseline_cache_hit = cache is not None
        if cache is None:
            scores, alerts = self._score_graph(graph)
        else:
            scores = np.asarray(cache["scores"], dtype=np.float32)
            alerts = np.asarray(cache["native_alerts"], dtype=np.int64)
            logger.info("[STGAN] reused clean native scores from cache")
        # Baseline initialization happens before the attack loop.  Preload the
        # immutable detector snapshots here so the first modified-graph query
        # does not pay checkpoint I/O and every later query reuses them.
        self._ensure_snapshot_models()
        labels = np.zeros(len(scores), dtype=np.int8)
        self.valid_malicious = [node for node in self.malicious_nodes if 0 <= node < len(scores)]
        if not self.valid_malicious:
            raise ValueError("STGAN evaluation has no in-range malicious metadata nodes")
        labels[self.valid_malicious] = 1
        self.best_threshold = float(self.configured_threshold)
        if self.threshold_source == "fixed":
            alerts = np.flatnonzero(scores > self.best_threshold)
        self.raw_initial_metrics = self._classification_metrics(
            labels, scores, self.best_threshold
        )
        self.initial_alert_nodes = set(int(x) for x in alerts)
        self.initial_tp_nodes = sorted(self.initial_alert_nodes)
        self.initial_true_tp_nodes = self.initial_alert_nodes & set(self.valid_malicious)
        self.true_tp_count = len(self.initial_true_tp_nodes)
        self.false_positive_count = len(self.initial_alert_nodes - set(self.valid_malicious))
        self.true_malicious_set = set(self.valid_malicious)
        self.escaped_nodes_history = set()
        # Main attack reporting uses raw node decisions, exactly as FLASH does.
        # Two-hop post-processing depends on malicious ground truth and is kept
        # only as a separately named reproduction audit.
        self.initial_metrics = dict(self.raw_initial_metrics)
        self.released_initial_metrics = self._released_code_metrics(
            scores, self.initial_alert_nodes
        )
        if cache is None:
            self._save_clean_cache(graph, scores, alerts, self.initial_metrics)
        logger.info(
            "[STGAN] raw native baseline (no two-hop): alerts=%d, TP=%d, FP=%d, "
            "F1=%.6f, AUC=%.6f",
            len(alerts),
            self.true_tp_count,
            self.false_positive_count,
            self.initial_metrics["f1"],
            self.initial_metrics["auc"],
        )
        logger.info(
            "[STGAN] released two-hop audit only: TP=%d, FP=%d, F1=%.6f, "
            "official AUC=not-defined",
            self.released_initial_metrics["tp"],
            self.released_initial_metrics["fp"],
            self.released_initial_metrics["f1"],
        )
        return {
            "initial_tp_nodes": self.initial_tp_nodes,
            "initial_true_tp_nodes": sorted(self.initial_true_tp_nodes),
            "threshold": self.best_threshold,
            "asr_denominator": self.true_tp_count,
            "all_scores": {int(i): float(v) for i, v in enumerate(scores)},
            "ground_truth": self.ground_truth_source,
            "threshold_source": self.threshold_source,
        }

    def evaluate_native_scores(self, local_scores: Dict[int, float], *, threshold=None):
        scores = np.asarray(
            [local_scores.get(i, float("-inf")) for i in range(len(local_scores))]
        )
        operating = self.best_threshold if threshold is None else float(threshold)
        labels = np.zeros(len(scores), dtype=np.int8)
        valid = [node for node in self.malicious_nodes if 0 <= node < len(scores)]
        labels[valid] = 1
        return self._classification_metrics(labels, scores, operating)

    def compute_asr_from_scores(self, scores: Dict[int, float], *, final: bool = False) -> float:
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
