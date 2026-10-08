"""Serializable native input state for the released STGAN evaluator.

The attack graph is a MAGIC DGL graph, while STGAN consumes a different view:
the original actor-to-object CDM multigraph, per-event timestamps, observable
entity-type labels, and positional Word2Vec summaries.  Keeping this data in a
sidecar prevents accidental use of MAGIC features or fabricated zero vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np


@dataclass
class STGANSemanticState:
    original_node_count: int
    base_embeddings: np.ndarray
    token_counts: np.ndarray
    model_labels: np.ndarray
    native_edge_index: np.ndarray
    native_edge_time: np.ndarray
    magic_edge_type_names: Dict[int, str] = field(default_factory=dict)
    node_type_names: Dict[int, str] = field(default_factory=dict)
    dataset: str = "cadets"
    test_index: int = 0
    global_node_offset: int = 0
    use_positional_encoding: bool = True
    word2vec_source: str = ""
    source: str = ""
    _added_edges: list[tuple[int, int]] = field(default_factory=list)
    _added_times: list[float] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "STGANSemanticState":
        path = Path(path)
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "base_embeddings", "token_counts", "model_labels",
                "native_edge_index", "native_edge_time", "original_node_count",
            }
            missing = required - set(payload.files)
            if missing:
                raise ValueError(f"STGAN state is incomplete; missing {sorted(missing)}")
            edge_names = {}
            if {"magic_edge_type_ids", "magic_edge_type_names"}.issubset(payload.files):
                edge_names = {
                    int(key): str(value)
                    for key, value in zip(
                        np.asarray(payload["magic_edge_type_ids"]).tolist(),
                        np.asarray(payload["magic_edge_type_names"]).tolist(),
                    )
                }
            node_names = {}
            if {"node_type_ids", "node_type_names"}.issubset(payload.files):
                node_names = {
                    int(key): str(value)
                    for key, value in zip(
                        np.asarray(payload["node_type_ids"]).tolist(),
                        np.asarray(payload["node_type_names"]).tolist(),
                    )
                }
            state = cls(
                original_node_count=int(np.asarray(payload["original_node_count"]).item()),
                base_embeddings=np.asarray(payload["base_embeddings"], dtype=np.float32).copy(),
                token_counts=np.asarray(payload["token_counts"], dtype=np.int64).copy(),
                model_labels=np.asarray(payload["model_labels"], dtype=np.int64).copy(),
                native_edge_index=np.asarray(payload["native_edge_index"], dtype=np.int64).copy(),
                native_edge_time=np.asarray(payload["native_edge_time"], dtype=np.float32).copy(),
                magic_edge_type_names=edge_names,
                node_type_names=node_names,
                dataset=str(np.asarray(payload["dataset"]).item()) if "dataset" in payload.files else "cadets",
                test_index=int(np.asarray(payload["test_index"]).item()) if "test_index" in payload.files else 0,
                global_node_offset=int(np.asarray(payload["global_node_offset"]).item()) if "global_node_offset" in payload.files else 0,
                use_positional_encoding=bool(
                    np.asarray(payload["use_positional_encoding"]).item()
                ) if "use_positional_encoding" in payload.files else True,
                word2vec_source=str(
                    np.asarray(payload["word2vec_source"]).item()
                ) if "word2vec_source" in payload.files else "",
                source=str(path.resolve()),
            )
        state.validate_arrays()
        return state

    def validate_arrays(self) -> None:
        n = int(self.original_node_count)
        if self.base_embeddings.shape != (n, 30):
            raise ValueError(f"STGAN embeddings must have shape ({n}, 30)")
        if self.token_counts.shape != (n,) or self.model_labels.shape != (n,):
            raise ValueError("STGAN node arrays do not match original_node_count")
        if self.native_edge_index.ndim != 2 or self.native_edge_index.shape[0] != 2:
            raise ValueError("STGAN native_edge_index must have shape (2, E)")
        if self.native_edge_time.shape != (self.native_edge_index.shape[1],):
            raise ValueError("STGAN edge_time length does not match edge_index")
        class_counts = {"cadets": 6, "theia": 6, "trace": 12}
        classes = class_counts.get(self.dataset)
        if classes is None:
            raise ValueError(f"unsupported STGAN dataset in state: {self.dataset!r}")
        if n and (
            int(self.model_labels.min()) < 0
            or int(self.model_labels.max()) >= classes
        ):
            raise ValueError(
                f"STGAN {self.dataset} entity labels must be in [0, {classes - 1}]"
            )

    def validate_for_graph(self, graph, dataset: str, test_index: int) -> None:
        self.validate_arrays()
        if self.dataset != dataset:
            raise ValueError(f"STGAN state dataset={self.dataset!r}, requested={dataset!r}")
        if int(self.test_index) != int(test_index):
            raise ValueError(
                f"STGAN state=test{self.test_index}, requested=test{test_index}"
            )
        if int(graph.number_of_nodes()) != int(self.original_node_count):
            raise ValueError(
                "STGAN state/graph node mismatch: "
                f"state={self.original_node_count}, graph={graph.number_of_nodes()}"
            )

    def to_npz(self, path: str | Path) -> Path:
        self.validate_arrays()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        edge_items = sorted(self.magic_edge_type_names.items())
        node_items = sorted(self.node_type_names.items())
        np.savez_compressed(
            path,
            version=np.asarray(1, dtype=np.int64),
            dataset=np.asarray(self.dataset),
            test_index=np.asarray(self.test_index, dtype=np.int64),
            global_node_offset=np.asarray(self.global_node_offset, dtype=np.int64),
            use_positional_encoding=np.asarray(
                self.use_positional_encoding, dtype=np.bool_
            ),
            word2vec_source=np.asarray(self.word2vec_source),
            original_node_count=np.asarray(self.original_node_count, dtype=np.int64),
            base_embeddings=np.asarray(self.base_embeddings, dtype=np.float32),
            token_counts=np.asarray(self.token_counts, dtype=np.int64),
            model_labels=np.asarray(self.model_labels, dtype=np.int64),
            native_edge_index=np.asarray(self.native_edge_index, dtype=np.int64),
            native_edge_time=np.asarray(self.native_edge_time, dtype=np.float32),
            magic_edge_type_ids=np.asarray([key for key, _ in edge_items], dtype=np.int64),
            magic_edge_type_names=np.asarray([value for _, value in edge_items], dtype=np.str_),
            node_type_ids=np.asarray([key for key, _ in node_items], dtype=np.int64),
            node_type_names=np.asarray([value for _, value in node_items], dtype=np.str_),
        )
        self.source = str(path.resolve())
        return path

    def edge_index(self) -> np.ndarray:
        if not self._added_edges:
            return self.native_edge_index
        return np.concatenate(
            (self.native_edge_index, np.asarray(self._added_edges, dtype=np.int64).T),
            axis=1,
        )

    def edge_time(self) -> np.ndarray:
        if not self._added_times:
            return self.native_edge_time
        return np.concatenate(
            (self.native_edge_time, np.asarray(self._added_times, dtype=np.float32))
        )

    @staticmethod
    def _position_vector(position: int, dim: int = 30) -> np.ndarray:
        result = np.zeros(dim, dtype=np.float32)
        scale = np.exp(
            np.arange(0, dim, 2, dtype=np.float32)
            * (-np.log(10000.0) / dim)
        )
        result[0::2] = np.sin(float(position) * scale)
        result[1::2] = np.cos(float(position) * scale)
        return result

    def add_edge(
        self,
        src: int,
        dst: int,
        action: str,
        *,
        word_vector: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        src, dst = int(src), int(dst)
        if not (0 <= src < self.original_node_count and 0 <= dst < self.original_node_count):
            raise IndexError("STGAN currently supports edge-only edits between original nodes")
        self._added_edges.append((src, dst))
        if timestamp is None:
            timestamp = float(self.native_edge_time[-1]) if self.native_edge_time.size else 0.0
        self._added_times.append(float(timestamp))
        if word_vector is None:
            return
        value = np.asarray(word_vector, dtype=np.float32)
        # Released STGAN appends the event phrase once for each endpoint role;
        # a self-loop therefore contributes twice.
        for node in (src, dst):
            count = int(self.token_counts[node])
            enriched = value
            if self.use_positional_encoding:
                enriched = enriched + self._position_vector(count)
            self.base_embeddings[node] = (
                self.base_embeddings[node] * count + enriched
            ) / float(count + 1)
            self.token_counts[node] = count + 1
