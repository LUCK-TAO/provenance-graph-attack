"""Native MAGIC evaluation helpers for partitioned entity-level datasets.

MAGIC serializes TRACE as five DGL graphs, but its original evaluator first
concatenates their node embeddings.  It then keeps every malicious node from
the early partitions and every node from the final partition.  The metadata
node IDs therefore live in the concatenated global index space, not in the
local index space of every ``test{i}.pkl`` graph.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


MAGIC_RECALL_FLOORS = {
    "cadets": 0.9976,
    "theia": 0.99996,
    "trace": 0.99979,
}


def graph_offsets(graph_sizes: Sequence[int]) -> List[int]:
    offsets = [0]
    for size in graph_sizes:
        offsets.append(offsets[-1] + int(size))
    return offsets


def localize_global_nodes(
    global_nodes: Iterable[int], graph_sizes: Sequence[int], test_index: int
) -> List[int]:
    """Map concatenated MAGIC metadata IDs into one graph's local ID space."""
    offsets = graph_offsets(graph_sizes)
    test_index = int(test_index)
    if not 0 <= test_index < len(graph_sizes):
        raise IndexError(f"test_index {test_index} outside graph layout")
    start, end = offsets[test_index], offsets[test_index + 1]
    return sorted(
        int(node) - start
        for node in global_nodes
        if start <= int(node) < end
    )


def build_trace_evaluation_layout(
    global_malicious_nodes: Iterable[int], graph_sizes: Sequence[int]
) -> Dict[str, object]:
    """Reproduce the sample selection performed by MAGIC ``eval.py``.

    Benign nodes from every partition except the final one are excluded.  All
    malicious nodes remain, and all nodes in the final partition remain.
    """
    graph_sizes = [int(size) for size in graph_sizes]
    if not graph_sizes:
        raise ValueError("TRACE evaluation requires at least one test graph")
    offsets = graph_offsets(graph_sizes)
    final_start = offsets[-2]
    final_end = offsets[-1]
    malicious = sorted({
        int(node) for node in global_malicious_nodes
        if 0 <= int(node) < final_end
    })
    early_malicious = [node for node in malicious if node < final_start]
    final_malicious_local = [
        node - final_start for node in malicious if node >= final_start
    ]
    labels = np.zeros(len(early_malicious) + graph_sizes[-1], dtype=np.int8)
    labels[:len(early_malicious)] = 1
    if final_malicious_local:
        labels[
            len(early_malicious) + np.asarray(final_malicious_local, dtype=np.int64)
        ] = 1
    return {
        "graph_sizes": graph_sizes,
        "offsets": offsets,
        "early_malicious_global": early_malicious,
        "final_malicious_local": final_malicious_local,
        "labels": labels,
    }


def trace_local_to_evaluation_positions(
    layout: Dict[str, object], test_index: int
) -> Dict[int, int]:
    """Map local IDs in one TRACE graph to MAGIC's evaluation-vector slots."""
    graph_sizes = [int(value) for value in layout["graph_sizes"]]
    offsets = [int(value) for value in layout["offsets"]]
    early = [int(value) for value in layout["early_malicious_global"]]
    test_index = int(test_index)
    if not 0 <= test_index < len(graph_sizes):
        raise IndexError(f"test_index {test_index} outside graph layout")
    if test_index == len(graph_sizes) - 1:
        prefix = len(early)
        return {node: prefix + node for node in range(graph_sizes[test_index])}
    early_positions = {node: pos for pos, node in enumerate(early)}
    start, end = offsets[test_index], offsets[test_index + 1]
    return {
        global_node - start: early_positions[global_node]
        for global_node in early
        if start <= global_node < end
    }


def select_magic_threshold(
    dataset: str, labels: Sequence[int], scores: Sequence[float]
) -> float:
    """Select the operating threshold using MAGIC's native recall rule."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        raise ValueError("MAGIC threshold selection received no score thresholds")
    best_idx = -1
    recall_floor = MAGIC_RECALL_FLOORS.get(str(dataset).lower())
    if recall_floor is not None:
        for index, value in enumerate(recall):
            if value < recall_floor:
                best_idx = index - 1
                break
    if best_idx < 0 or best_idx >= len(thresholds):
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        best_idx = min(int(np.argmax(f1)), len(thresholds) - 1)
    return float(thresholds[best_idx])


def classification_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= float(threshold)
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    auc = (
        float(roc_auc_score(labels, scores))
        if len(np.unique(labels)) == 2 else float("nan")
    )
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
