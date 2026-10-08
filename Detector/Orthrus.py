"""Orthrus adapter for the detector-agnostic graph attack framework.

The attack target pool is read from a reproduced native Orthrus attack-state
file. Candidate construction, macro actions, budgets, rewards and three-seed
evaluation remain those of the shared attack framework; only black-box graph
scoring is delegated to the released Orthrus model and weights.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from Detector.Detector_Interface import BaseDetector
from Detector.Orthrus_State import OrthrusAttackState
from MAGIC_main.action_schema import build_observed_action_schema


ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("attack.detector.orthrus")


class OrthrusDetector(BaseDetector):
    score_direction = "higher_is_more_anomalous"
    detector_family = "temporal_edge_type_prediction"
    attack_state_extra_dim = 0
    attack_state_dim = 50
    force_observed_action_schema = True

    def __init__(
        self,
        dataset_name: str = "cadets",
        device: str = "cuda:0",
        top_k: int = 8,
        attack_state_path: Optional[str] = None,
        threshold_source: str = "native",
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if dataset_name not in {"cadets", "theia"}:
            raise ValueError("the Orthrus attack adapter supports cadets and theia")
        if threshold_source not in {"native", "fixed"}:
            raise ValueError("threshold_source must be 'native' or 'fixed'")
        if threshold_source == "fixed" and threshold is None:
            raise ValueError("threshold_source='fixed' requires --threshold")
        super().__init__(
            dataset_name=dataset_name,
            device=device,
            top_k=top_k,
            threshold=threshold,
            **kwargs,
        )
        state_names = {
            "cadets": "cadets_e3_tw228_229.npz",
            "theia": "theia_e3_native_full_tp.npz",
        }
        default_state = (
            ROOT / "Orthrus_official" / "artifacts" / "attack_state"
            / state_names[dataset_name]
        )
        self.attack_state_path = Path(attack_state_path or default_state).resolve()
        self.state: Optional[OrthrusAttackState] = None
        self.threshold_source = threshold_source
        self.configured_threshold = threshold
        self.test_index = 0
        self.modifications: List[Dict[str, int]] = []
        self.initial_true_tp_nodes: set[int] = set()
        self.true_malicious_set: set[int] = set()
        self.valid_malicious: List[int] = []
        self.false_positive_count = 0
        self.true_tp_count = 0
        self.initial_metrics: Dict[str, float] = {}
        self.raw_initial_metrics: Dict[str, float] = {}
        self.native_reproduction_metrics: Dict[str, float] = {}
        self.last_baseline_cache_hit = True
        self.full_aware_enabled = False
        self._process = None
        self._connection = None

    def set_evaluation_context(self, test_index: int) -> None:
        if int(test_index) != 0:
            raise ValueError("Orthrus E3 attack states use test context 0")
        self.test_index = 0

    def attach_attack_state(self, state: OrthrusAttackState) -> None:
        if state.dataset != self.dataset_name:
            raise ValueError("Orthrus attack state dataset mismatch")
        self.state = state
        truth_positions = np.flatnonzero(state.baseline_result_truth == 1)
        truth_global = set(int(value) for value in state.baseline_result_ids[truth_positions])
        mapping = state.global_to_local
        self.valid_malicious = sorted(
            mapping[node] for node in truth_global if node in mapping
        )
        self.true_malicious_set = set(self.valid_malicious)
        logger.info(
            "[Orthrus] attack state attached: nodes=%d, edges=%d, targets=%d, windows=%s",
            state.node_count,
            state.edge_count,
            len(state.target_local_ids),
            state.selected_windows.tolist(),
        )

    def build_action_schema(self, graph, *, action_dim: int = 9):
        if self.state is None:
            raise RuntimeError("attach the Orthrus attack state before schema creation")
        return build_observed_action_schema(
            graph,
            dataset=self.dataset_name,
            action_dim=action_dim,
            edge_type_names={
                index: str(name)
                for index, name in enumerate(self.state.edge_type_names.tolist())
            },
            node_type_names={
                index: str(name)
                for index, name in enumerate(self.state.node_type_names.tolist())
            },
        )

    def enable_full_aware(self, enabled: bool = True) -> None:
        """Enable Orthrus-specific temporal diagnostics and edge probes."""
        self.full_aware_enabled = bool(enabled)

    def record_attack_modification(
        self,
        *,
        kind,
        src,
        dst,
        edge_type,
        before_time=None,
        window=None,
        probe_id=None,
        **_kwargs,
    ):
        if kind != "edge" or self.state is None:
            return
        src = int(src)
        dst = int(dst)
        target_windows = self.state.target_local_to_window
        if window is None:
            window = target_windows.get(src, target_windows.get(dst))
        if window is None:
            window = int(self.state.selected_windows[0])
        if probe_id is None:
            probe_id = len(self.modifications)
        self.modifications.append({
            "src": src,
            "dst": dst,
            "edge_type": int(edge_type),
            "window": int(window),
            "probe_id": int(probe_id),
            **(
                {"before_time": int(before_time)}
                if before_time is not None and int(before_time) >= 0 else {}
            ),
        })

    def rollback_attack_modifications(self, keep_count: int) -> None:
        """Discard all committed adapter edits after ``keep_count``."""
        keep_count = max(0, int(keep_count))
        del self.modifications[keep_count:]

    def _worker_query(self, modifications, inspect_nodes=()):
        self._ensure_worker()
        self._connection.send({
            "command": "query",
            "modifications": list(modifications),
            "inspect_nodes": [int(node) for node in inspect_nodes],
        })
        message = self._connection.recv()
        if message.get("status") != "ok":
            raise RuntimeError(
                f"Orthrus query failed: {message.get('error')}\n"
                f"{message.get('traceback', '')}"
            )
        return message["result"]

    def probe_edge_candidates(self, candidates, *, inspect_nodes=()):
        """Score provisional edges with native Orthrus without committing them."""
        if not self.full_aware_enabled:
            raise RuntimeError("Orthrus full-aware probing is not enabled")
        provisional = []
        offset = len(self.modifications)
        for index, candidate in enumerate(candidates):
            row = dict(candidate)
            row["probe_id"] = int(row.get("probe_id", offset + index))
            provisional.append(row)
        payload = self._worker_query(
            [*self.modifications, *provisional],
            inspect_nodes=inspect_nodes,
        )
        return {
            "candidate_losses": {
                int(key): float(value)
                for key, value in payload.get("added_edge_losses", {}).items()
                if int(key) >= offset
            },
            "max_loss_edges": payload.get("max_loss_edges", {}),
            "local_scores": payload.get("local_scores", ()),
        }

    def _baseline_alerts(self) -> List[int]:
        if self.state is None:
            raise RuntimeError("Orthrus attack state is not attached")
        if self.threshold_source == "fixed":
            cutoff = float(self.configured_threshold)
            return np.flatnonzero(
                self.state.local_baseline_scores > cutoff
            ).astype(np.int64).tolist()
        return sorted(int(value) for value in self.state.target_local_ids.tolist())

    def _native_alert_rule(self) -> str:
        if self.threshold_source == "fixed":
            return "fixed_threshold"
        if self.state is not None and self.state.alert_mode == "full":
            return "native_max_validation_loss_threshold"
        return "native_top20_kmeans_with_clean_global_scores"

    def _baseline_result(self, node_indices=None) -> Dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Orthrus attack state is not attached")
        scores = {
            index: float(value)
            for index, value in enumerate(self.state.local_baseline_scores.tolist())
        }
        alerts = self._baseline_alerts()
        if self.best_threshold is None:
            native_cutoff = (
                float(self.state.native_threshold)
                if self.state.alert_mode == "full"
                else min(scores[node] for node in alerts)
            )
            self.best_threshold = (
                float(self.configured_threshold)
                if self.threshold_source == "fixed"
                else float(native_cutoff)
            )
        return self._make_result(
            scores,
            threshold=float(self.best_threshold),
            native_alerts=alerts,
            node_indices=node_indices,
            metadata={
                "model": "Orthrus",
                "dataset": f"{self.dataset_name.upper()}_E3",
                "query_scope": "native_temporal_selected_windows",
                "alert_rule": self._native_alert_rule(),
                "added_edges": 0,
            },
        )

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        from Detector.Orthrus_Worker import worker_main

        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=worker_main,
            args=(child, str(self.attack_state_path), str(self.device)),
            daemon=True,
            name="orthrus-attack-inference",
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent
        started = time.perf_counter()
        while True:
            if parent.poll(30.0):
                message = parent.recv()
                if message.get("status") == "ready":
                    logger.info(
                        "[Orthrus] native inference worker ready in %.2fs on %s",
                        time.perf_counter() - started,
                        message.get("device"),
                    )
                    return
                raise RuntimeError(
                    "Orthrus worker initialization failed: "
                    f"{message.get('error')}\n{message.get('traceback', '')}"
                )
            if not process.is_alive():
                raise RuntimeError(
                    f"Orthrus worker exited during initialization (exit={process.exitcode})"
                )
            logger.info(
                "[Orthrus] preparing native temporal state (elapsed %.1fs)",
                time.perf_counter() - started,
            )

    def query(self, graph, node_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        if self.state is None:
            raise RuntimeError("attach an Orthrus attack state before querying")
        self.state.validate_graph(graph)
        if not self.modifications and not (
            self.full_aware_enabled and node_indices
        ):
            return self._baseline_result(node_indices=node_indices)
        payload = self._worker_query(
            self.modifications,
            inspect_nodes=node_indices or (),
        )
        local_scores = np.asarray(payload["local_scores"], dtype=np.float64)
        alerts = [int(value) for value in payload["native_alerts"]]
        if self.threshold_source == "fixed":
            alerts = np.flatnonzero(local_scores > float(self.best_threshold)).tolist()
        return self._make_result(
            {index: float(value) for index, value in enumerate(local_scores.tolist())},
            threshold=float(self.best_threshold),
            native_alerts=alerts,
            node_indices=node_indices,
            metadata={
                "model": "Orthrus",
                "dataset": f"{self.dataset_name.upper()}_E3",
                "query_scope": "native_temporal_selected_windows",
                "alert_rule": self._native_alert_rule(),
                "added_edges": len(self.modifications),
                "global_alert_count": int(payload["global_alert_count"]),
                "cluster_centers": payload["cluster_centers"],
                "max_loss_edges": payload.get("max_loss_edges", {}),
                "added_edge_losses": payload.get("added_edge_losses", {}),
            },
        )

    def initialize_tp_baseline(self, graph) -> Dict[str, Any]:
        if self.state is None:
            raise RuntimeError("attach an Orthrus attack state before baseline setup")
        self.state.validate_graph(graph)
        self.modifications.clear()
        baseline = self._baseline_result()
        target_nodes = set(int(value) for value in self.state.target_local_ids.tolist())
        alerts = set(int(value) for value in baseline["native_alerts"])
        self.initial_alert_nodes = alerts
        self.initial_tp_nodes = sorted(alerts)
        self.initial_true_tp_nodes = alerts & target_nodes
        self.true_tp_count = len(self.initial_true_tp_nodes)
        self.false_positive_count = len(alerts - target_nodes)
        self.true_malicious_set = set(self.initial_true_tp_nodes)
        self.escaped_nodes_history = set()

        alert_scores = [baseline["all_scores"][node] for node in alerts]
        native_cutoff = (
            float(self.state.native_threshold)
            if self.state.alert_mode == "full"
            else min(alert_scores)
        )
        self.best_threshold = (
            float(self.configured_threshold)
            if self.threshold_source == "fixed"
            else float(native_cutoff)
        )
        native_truth = self.state.baseline_result_truth.astype(np.int8)
        native_predicted = self.state.baseline_result_hat.astype(np.int8)
        native_scores = self.state.baseline_result_scores.astype(np.float64)
        native_tp = int(np.sum((native_truth == 1) & (native_predicted == 1)))
        native_fp = int(np.sum((native_truth == 0) & (native_predicted == 1)))
        native_fn = int(np.sum((native_truth == 1) & (native_predicted == 0)))
        native_tn = int(np.sum((native_truth == 0) & (native_predicted == 0)))
        self.native_reproduction_metrics = {
            "auc": float(roc_auc_score(native_truth, native_scores)),
            "tp": native_tp,
            "fp": native_fp,
            "tn": native_tn,
            "fn": native_fn,
        }

        # The attack experiment is evaluated on its fixed local native-TP
        # target universe, not on Orthrus' full benchmark node universe.
        truth = np.zeros(self.state.node_count, dtype=np.int8)
        truth[np.asarray(sorted(target_nodes), dtype=np.int64)] = 1
        predicted = np.zeros(self.state.node_count, dtype=np.int8)
        predicted[np.asarray(sorted(alerts), dtype=np.int64)] = 1
        scores = self.state.local_baseline_scores.astype(np.float64)
        tp = int(np.sum((truth == 1) & (predicted == 1)))
        fp = int(np.sum((truth == 0) & (predicted == 1)))
        fn = int(np.sum((truth == 1) & (predicted == 0)))
        tn = int(np.sum((truth == 0) & (predicted == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        self.initial_metrics = {
            "auc": float(roc_auc_score(truth, scores)),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "tn": tn,
            "fn": fn,
            "tp": tp,
            "fp": fp,
            "threshold": float(self.best_threshold),
            "metric_scope": "orthrus_attack_framework_fixed_initial_tp_pool",
        }
        self.raw_initial_metrics = dict(self.initial_metrics)
        logger.info(
            "[Orthrus] fixed attack baseline: targets=%d, TP=%d, FP=%d, "
            "feedback_cutoff=%.6f",
            len(target_nodes), tp, fp, self.best_threshold,
        )
        return {
            "initial_tp_nodes": self.initial_tp_nodes,
            "initial_true_tp_nodes": sorted(self.initial_true_tp_nodes),
            "threshold": self.best_threshold,
            "all_scores": baseline["all_scores"],
            "native_alerts": baseline["native_alerts"],
        }

    def close(self) -> None:
        if self._connection is not None and self._process is not None:
            try:
                if self._process.is_alive():
                    self._connection.send({"command": "close"})
                    if self._connection.poll(30.0):
                        self._connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            self._process.join(timeout=30.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=15.0)
            if self._process.is_alive() and hasattr(self._process, "kill"):
                self._process.kill()
                self._process.join(timeout=15.0)
            try:
                self._connection.close()
            except OSError:
                pass
        self._connection = None
        self._process = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
