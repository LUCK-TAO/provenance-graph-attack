"""Serializable semantic/topology state for the native FLASH adapter.

FLASH does not consume MAGIC's numeric node attributes.  Its CADETS model uses
Word2Vec summaries of ``exec/action/path`` event phrases and the original CDM
edge direction.  This sidecar therefore keeps the detector input separate from
the graph that the attack framework edits.

The preferred format is a compressed ``.npz`` file produced by
``Flash_IDS_main/flash_cadets.py prepare``.  Legacy JSON phrase files are still
accepted so old experiments fail explicitly only when their content is
insufficient, not merely because the extension changed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np


@dataclass
class FlashSemanticState:
    """Mutable detector state keyed by DGL node index."""

    node_phrases: Dict[int, List[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    node_types: Dict[int, str] = field(default_factory=dict)
    edge_actions: List[tuple[int, int, str]] = field(default_factory=list)
    original_node_count: int = 0
    base_embeddings: Optional[np.ndarray] = None
    token_counts: Optional[np.ndarray] = None
    document_counts: Optional[np.ndarray] = None
    native_edge_index: Optional[np.ndarray] = None
    incident_counts: Optional[np.ndarray] = None
    distinct_action_counts: Optional[np.ndarray] = None
    # FLASH class labels are not guaranteed to use the same numeric ordering
    # as MAGIC's one-hot node attributes.  Native E3 sidecars therefore carry
    # the checkpoint label for every MAGIC node index explicitly.
    model_labels: Optional[np.ndarray] = None
    # Mapping from MAGIC edge-type IDs to the original CDM event names.  It is
    # used both for readable/actionable schemas and for Word2Vec updates after
    # an attack edit.
    magic_edge_type_names: Dict[int, str] = field(default_factory=dict)
    node_type_names: Dict[int, str] = field(default_factory=dict)
    test_index: int = 0
    global_node_offset: int = 0
    dataset: str = ""
    source: str = ""
    _added_edges: List[tuple[int, int]] = field(default_factory=list)
    _added_actions: Dict[int, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    @classmethod
    def from_event_rows(
        cls,
        rows: Iterable[Mapping[str, str]],
        node_to_index: Mapping[str, int],
    ) -> "FlashSemanticState":
        state = cls(original_node_count=len(node_to_index))
        for row in rows:
            tokens = [str(row.get(key, "")) for key in ("exec", "action", "path")]
            tokens = [token for token in tokens if token]
            action = str(row.get("action", ""))
            src = node_to_index.get(str(row.get("actorID", "")))
            dst = node_to_index.get(str(row.get("objectID", "")))
            for node in (src, dst):
                if node is not None:
                    state.node_phrases[node].extend(tokens)
            if src is not None and dst is not None:
                state.edge_actions.append((src, dst, action))
        return state

    @classmethod
    def from_json(cls, path: str | Path) -> "FlashSemanticState":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            node_phrases={
                int(key): list(value)
                for key, value in payload["node_phrases"].items()
            },
            node_types={
                int(key): str(value)
                for key, value in payload.get("node_types", {}).items()
            },
            edge_actions=[
                (int(src), int(dst), str(action))
                for src, dst, action in payload.get("edge_actions", [])
            ],
            original_node_count=int(payload["original_node_count"]),
            dataset=str(payload.get("dataset", "")),
            source=str(path),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "FlashSemanticState":
        path = Path(path)
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "base_embeddings",
                "token_counts",
                "native_edge_index",
                "original_node_count",
            }
            missing = required - set(payload.files)
            if missing:
                raise ValueError(
                    f"FLASH semantic state is incomplete; missing {sorted(missing)}"
                )
            embeddings = np.asarray(payload["base_embeddings"], dtype=np.float32)
            counts = np.asarray(payload["token_counts"], dtype=np.int64)
            document_counts = (
                np.asarray(payload["document_counts"], dtype=np.int64)
                if "document_counts" in payload.files
                else counts.copy()
            )
            edge_index = np.asarray(payload["native_edge_index"], dtype=np.int64)
            node_count = int(np.asarray(payload["original_node_count"]).item())
            incident = (
                np.asarray(payload["incident_counts"], dtype=np.int64)
                if "incident_counts" in payload.files
                else np.zeros(node_count, dtype=np.int64)
            )
            distinct = (
                np.asarray(payload["distinct_action_counts"], dtype=np.int64)
                if "distinct_action_counts" in payload.files
                else np.zeros(node_count, dtype=np.int64)
            )
            dataset = (
                str(np.asarray(payload["dataset"]).item())
                if "dataset" in payload.files
                else ""
            )
            model_labels = (
                np.asarray(payload["model_labels"], dtype=np.int64)
                if "model_labels" in payload.files
                else None
            )
            if model_labels is not None and model_labels.size == 0:
                model_labels = None
            edge_type_names = {}
            if {
                "magic_edge_type_ids",
                "magic_edge_type_names",
            }.issubset(payload.files):
                edge_type_names = {
                    int(key): str(value)
                    for key, value in zip(
                        np.asarray(payload["magic_edge_type_ids"]).tolist(),
                        np.asarray(payload["magic_edge_type_names"]).tolist(),
                    )
                }
            node_type_names = {}
            if {"node_type_ids", "node_type_names"}.issubset(payload.files):
                node_type_names = {
                    int(key): str(value)
                    for key, value in zip(
                        np.asarray(payload["node_type_ids"]).tolist(),
                        np.asarray(payload["node_type_names"]).tolist(),
                    )
                }
            test_index = (
                int(np.asarray(payload["test_index"]).item())
                if "test_index" in payload.files
                else 0
            )
            global_node_offset = (
                int(np.asarray(payload["global_node_offset"]).item())
                if "global_node_offset" in payload.files
                else 0
            )
        if embeddings.shape != (node_count, 30):
            raise ValueError(
                "FLASH base_embeddings must have shape "
                f"({node_count}, 30), got {embeddings.shape}"
            )
        if counts.shape != (node_count,):
            raise ValueError("FLASH token_counts length does not match node count")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("FLASH native_edge_index must have shape (2, E)")
        if model_labels is not None and model_labels.shape != (node_count,):
            raise ValueError("FLASH model_labels length does not match node count")
        return cls(
            original_node_count=node_count,
            base_embeddings=embeddings.copy(),
            token_counts=counts.copy(),
            document_counts=document_counts.copy(),
            native_edge_index=edge_index,
            incident_counts=incident.copy(),
            distinct_action_counts=distinct.copy(),
            model_labels=(model_labels.copy() if model_labels is not None else None),
            magic_edge_type_names=edge_type_names,
            node_type_names=node_type_names,
            test_index=test_index,
            global_node_offset=global_node_offset,
            dataset=dataset,
            source=str(path.resolve()),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FlashSemanticState":
        path = Path(path)
        if path.suffix.lower() == ".npz":
            return cls.from_npz(path)
        if path.suffix.lower() == ".json":
            return cls.from_json(path)
        raise ValueError("FLASH semantic state must be a .npz or legacy .json file")

    def to_npz(self, path: str | Path) -> Path:
        if self.base_embeddings is None or self.token_counts is None:
            raise ValueError("cannot save native FLASH state without embeddings")
        if self.native_edge_index is None:
            raise ValueError("cannot save native FLASH state without edge_index")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        edge_type_items = sorted(self.magic_edge_type_names.items())
        node_type_items = sorted(self.node_type_names.items())
        np.savez_compressed(
            path,
            version=np.asarray(2, dtype=np.int64),
            dataset=np.asarray(self.dataset),
            test_index=np.asarray(self.test_index, dtype=np.int64),
            global_node_offset=np.asarray(
                self.global_node_offset, dtype=np.int64
            ),
            original_node_count=np.asarray(self.original_node_count, dtype=np.int64),
            base_embeddings=np.asarray(self.base_embeddings, dtype=np.float32),
            token_counts=np.asarray(self.token_counts, dtype=np.int64),
            document_counts=np.asarray(
                self.document_counts
                if self.document_counts is not None
                else self.token_counts,
                dtype=np.int64,
            ),
            native_edge_index=np.asarray(self.native_edge_index, dtype=np.int64),
            incident_counts=np.asarray(
                self.incident_counts
                if self.incident_counts is not None
                else np.zeros(self.original_node_count),
                dtype=np.int64,
            ),
            distinct_action_counts=np.asarray(
                self.distinct_action_counts
                if self.distinct_action_counts is not None
                else np.zeros(self.original_node_count),
                dtype=np.int64,
            ),
            model_labels=np.asarray(
                self.model_labels
                if self.model_labels is not None
                else np.empty(0, dtype=np.int64),
                dtype=np.int64,
            ),
            magic_edge_type_ids=np.asarray(
                [key for key, _ in edge_type_items], dtype=np.int64
            ),
            magic_edge_type_names=np.asarray(
                [value for _, value in edge_type_items], dtype=np.str_
            ),
            node_type_ids=np.asarray(
                [key for key, _ in node_type_items], dtype=np.int64
            ),
            node_type_names=np.asarray(
                [value for _, value in node_type_items], dtype=np.str_
            ),
        )
        return path

    def validate_for_graph(self, graph, dataset: str) -> None:
        node_count = int(graph.number_of_nodes())
        if self.original_node_count != node_count:
            raise ValueError(
                "FLASH state/graph node mismatch: "
                f"state={self.original_node_count}, graph={node_count}"
            )
        if self.dataset and self.dataset != dataset:
            raise ValueError(
                f"FLASH state dataset={self.dataset!r}, requested={dataset!r}"
            )
        if self.base_embeddings is None and not self.node_phrases:
            raise ValueError("FLASH state contains neither embeddings nor phrases")
        if self.model_labels is not None and self.model_labels.size not in {
            0,
            node_count,
        }:
            raise ValueError("FLASH model_labels length does not match graph")

    def labels_for_graph(self, graph) -> np.ndarray:
        """Return checkpoint-native labels for the attached graph."""
        if self.model_labels is not None and self.model_labels.size:
            return np.asarray(self.model_labels, dtype=np.int64)
        return graph.ndata["type"].view(-1).cpu().numpy().astype(np.int64)

    def event_name_for_edge_type(self, edge_type: int) -> str:
        return self.magic_edge_type_names.get(
            int(edge_type), f"EDGE_TYPE_{int(edge_type)}"
        )

    def embeddings(self, word2vec_model) -> np.ndarray:
        """Return the current 30-D node features.

        Native ``.npz`` states already contain the notebook-equivalent
        positional Word2Vec means.  Legacy JSON states are converted without
        positional terms and are therefore intended only for compatibility
        checks, not paper experiments.
        """
        if self.base_embeddings is not None:
            return self.base_embeddings
        rows = np.zeros((self.original_node_count, 30), dtype=np.float32)
        for node, tokens in self.node_phrases.items():
            vectors = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
            if vectors:
                rows[int(node)] = np.mean(vectors, axis=0, dtype=np.float32)
        return rows

    def edge_index(self) -> Optional[np.ndarray]:
        if self.native_edge_index is None:
            return None
        if not self._added_edges:
            return self.native_edge_index
        added = np.asarray(self._added_edges, dtype=np.int64).T
        return np.concatenate((self.native_edge_index, added), axis=1)

    def add_edge(
        self,
        src: int,
        dst: int,
        action: str,
        *,
        word_vector: Optional[np.ndarray] = None,
        positional_vector: Optional[np.ndarray] = None,
        positional_vectors: Optional[Mapping[int, np.ndarray]] = None,
    ) -> None:
        src, dst, action = int(src), int(dst), str(action)
        self._added_edges.append((src, dst))
        self.edge_actions.append((src, dst, action))
        endpoints = (src,) if src == dst else (src, dst)
        for node in endpoints:
            self._added_actions[node][action] += 1
            if self.incident_counts is not None:
                self.incident_counts[node] += 1
            if self.distinct_action_counts is not None:
                self.distinct_action_counts[node] = max(
                    int(self.distinct_action_counts[node]),
                    len(self._added_actions[node]),
                )
            if self.document_counts is not None:
                self.document_counts[node] += 1
            if (
                self.base_embeddings is not None
                and self.token_counts is not None
                and word_vector is not None
            ):
                count = int(self.token_counts[node])
                value = np.asarray(word_vector, dtype=np.float32)
                node_position = (
                    positional_vectors.get(node)
                    if positional_vectors is not None
                    else positional_vector
                )
                if node_position is not None:
                    value = value + np.asarray(node_position, dtype=np.float32)
                self.base_embeddings[node] = (
                    self.base_embeddings[node] * count + value
                ) / float(count + 1)
                self.token_counts[node] = count + 1
            else:
                self.node_phrases[node].append(action)

    def features(self, node: int, vocabulary: set[str] | None = None) -> List[float]:
        node = int(node)
        if self.token_counts is not None:
            token_count = int(self.token_counts[node])
            incident = (
                int(self.incident_counts[node])
                if self.incident_counts is not None
                else 0
            )
            distinct = (
                int(self.distinct_action_counts[node])
                if self.distinct_action_counts is not None
                else len(self._added_actions[node])
            )
            return [
                min(token_count / 32.0, 1.0),
                min(len(self._added_actions[node]) / 9.0, 1.0),
                1.0 if token_count else 0.0,
                min(incident / 16.0, 1.0),
                min(distinct / 8.0, 1.0),
            ]

        phrase = self.node_phrases.get(node, [])
        counts = Counter(phrase)
        known = sum(token in vocabulary for token in phrase) if vocabulary else len(phrase)
        incident_actions = [
            action for src, dst, action in self.edge_actions if node in (src, dst)
        ]
        return [
            min(len(phrase) / 32.0, 1.0),
            min(len(counts) / 16.0, 1.0),
            known / max(1, len(phrase)),
            min(len(incident_actions) / 16.0, 1.0),
            min(len(set(incident_actions)) / 8.0, 1.0),
        ]
