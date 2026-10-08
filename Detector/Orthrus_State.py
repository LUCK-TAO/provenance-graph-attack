"""Compact editable graph view used by the Orthrus attack adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch


@dataclass
class OrthrusAttackState:
    source: str
    dataset: str
    alert_mode: str
    native_threshold: float
    global_node_ids: np.ndarray
    node_types: np.ndarray
    local_src: np.ndarray
    local_dst: np.ndarray
    edge_types: np.ndarray
    edge_times: np.ndarray
    edge_window_ids: np.ndarray
    selected_windows: np.ndarray
    selected_window_files: np.ndarray
    window_edge_counts: np.ndarray
    baseline_result_ids: np.ndarray
    baseline_result_scores: np.ndarray
    baseline_result_truth: np.ndarray
    baseline_result_hat: np.ndarray
    baseline_result_tw: np.ndarray
    local_baseline_scores: np.ndarray
    local_outside_max_scores: np.ndarray
    target_global_ids: np.ndarray
    target_local_ids: np.ndarray
    target_windows: np.ndarray
    node_type_names: np.ndarray
    edge_type_names: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "OrthrusAttackState":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Orthrus attack state not found: {source}")
        payload = np.load(source, allow_pickle=False)
        version = int(payload["version"][0])
        if version != 1:
            raise ValueError(f"unsupported Orthrus attack-state version: {version}")
        names = (
            "global_node_ids", "node_types", "local_src", "local_dst",
            "edge_types", "edge_times", "edge_window_ids", "selected_windows",
            "selected_window_files", "window_edge_counts", "baseline_result_ids",
            "baseline_result_scores", "baseline_result_truth", "baseline_result_hat",
            "baseline_result_tw", "local_baseline_scores",
            "local_outside_max_scores", "target_global_ids", "target_local_ids",
            "target_windows", "node_type_names", "edge_type_names",
        )
        values = {name: np.asarray(payload[name]) for name in names}
        state = cls(
            source=str(source),
            dataset=str(payload["dataset"][0]),
            alert_mode=(
                str(payload["alert_mode"][0])
                if "alert_mode" in payload.files else "ano"
            ),
            native_threshold=(
                float(payload["native_threshold"][0])
                if "native_threshold" in payload.files else float("nan")
            ),
            **values,
        )
        state._validate()
        return state

    def _validate(self) -> None:
        node_count = len(self.global_node_ids)
        edge_count = len(self.local_src)
        if self.dataset not in {"cadets", "theia"}:
            raise ValueError(
                f"unsupported Orthrus attack-state dataset: {self.dataset}"
            )
        if self.alert_mode not in {"ano", "full"}:
            raise ValueError(f"unsupported Orthrus alert mode: {self.alert_mode}")
        if self.alert_mode == "full" and not np.isfinite(self.native_threshold):
            raise ValueError("Orthrus full mode requires a finite native threshold")
        if len(np.unique(self.global_node_ids)) != node_count:
            raise ValueError("Orthrus global-node mapping is not one-to-one")
        if len(self.node_types) != node_count:
            raise ValueError("Orthrus node-type array length mismatch")
        if not (
            len(self.local_dst) == edge_count
            and len(self.edge_types) == edge_count
            and len(self.edge_times) == edge_count
            and len(self.edge_window_ids) == edge_count
        ):
            raise ValueError("Orthrus edge arrays have inconsistent lengths")
        if edge_count and (
            min(self.local_src.min(), self.local_dst.min()) < 0
            or max(self.local_src.max(), self.local_dst.max()) >= node_count
        ):
            raise ValueError("Orthrus local edge endpoint is out of range")
        if np.any(self.node_types < 0) or np.any(self.node_types >= len(self.node_type_names)):
            raise ValueError("Orthrus node type is out of range")
        if np.any(self.edge_types < 0) or np.any(self.edge_types >= len(self.edge_type_names)):
            raise ValueError("Orthrus edge type is out of range")
        expected_targets = set(int(value) for value in self.target_global_ids)
        mapped_targets = {
            int(self.global_node_ids[int(local)]) for local in self.target_local_ids
        }
        if expected_targets != mapped_targets:
            raise ValueError("Orthrus TP local/global mappings disagree")

    @property
    def node_count(self) -> int:
        return int(len(self.global_node_ids))

    @property
    def edge_count(self) -> int:
        return int(len(self.local_src))

    @property
    def global_to_local(self) -> Dict[int, int]:
        return {
            int(global_id): local_id
            for local_id, global_id in enumerate(self.global_node_ids.tolist())
        }

    @property
    def target_local_to_window(self) -> Dict[int, int]:
        return {
            int(local): int(window)
            for local, window in zip(self.target_local_ids, self.target_windows)
        }

    def to_dgl(self):
        import dgl

        graph = dgl.graph(
            (
                torch.as_tensor(self.local_src, dtype=torch.int64),
                torch.as_tensor(self.local_dst, dtype=torch.int64),
            ),
            num_nodes=self.node_count,
        )
        graph.ndata["attr"] = torch.nn.functional.one_hot(
            torch.as_tensor(self.node_types, dtype=torch.int64),
            num_classes=len(self.node_type_names),
        ).to(torch.float32)
        graph.edata["attr"] = torch.nn.functional.one_hot(
            torch.as_tensor(self.edge_types, dtype=torch.int64),
            num_classes=len(self.edge_type_names),
        ).to(torch.float32)
        return graph

    def validate_graph(self, graph) -> None:
        if int(graph.number_of_nodes()) != self.node_count:
            raise ValueError(
                f"Orthrus attack graph node mismatch: state={self.node_count}, "
                f"graph={graph.number_of_nodes()}"
            )
        if int(graph.number_of_edges()) < self.edge_count:
            raise ValueError("Orthrus attack graph lost native edges")
