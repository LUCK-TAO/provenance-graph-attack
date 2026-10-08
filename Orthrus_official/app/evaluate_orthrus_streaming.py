"""Low-memory equivalent of Orthrus' native max-loss node evaluation.

The released evaluator stores one Python float in a list for every incident
edge of every node.  THEIA has millions of events, so that representation can
exhaust Docker Desktop memory.  For ``threshold_method=max_val_loss`` only the
running maximum and its time-window index are needed.  This script preserves
the official ground truth lookup, top-20 KMeans alert rule, sklearn metrics and
result file schema while aggregating CSV files in chunks.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
from sklearn.cluster import KMeans


SRC = Path("/home/src")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import get_runtime_required_args, get_yml_cfg  # noqa: E402
from provnet_utils import classifier_evaluation, listdir_sorted, log  # noqa: E402


def native_cfg():
    args = get_runtime_required_args(args=[
        "THEIA_E3",
        "--from_weights",
        "--detection.gnn_training.encoder.graph_attention.dropout=0.1",
        "--seed=2",
    ])
    return get_yml_cfg(args)


def official_ground_truth(cfg) -> tuple[set[int], dict[int, str]]:
    rows = []
    for relative in cfg.dataset.ground_truth_relative_path:
        with open(Path(cfg._ground_truth_dir) / relative, newline="") as stream:
            rows.extend(csv.reader(stream))
    labels_by_uuid = {row[0]: row[1] for row in rows}
    uuids = list(labels_by_uuid)

    connection = psycopg2.connect(
        host=cfg.database.host,
        user=cfg.database.user,
        password=cfg.database.password,
        port=cfg.database.port,
        database=cfg.dataset.database,
    )
    mapping = {}
    try:
        with connection.cursor() as cursor:
            for table in ("file_node_table", "netflow_node_table", "subject_node_table"):
                cursor.execute(
                    f"SELECT index_id, node_uuid FROM {table} WHERE node_uuid = ANY(%s)",
                    (uuids,),
                )
                mapping.update({str(uuid): int(node) for node, uuid in cursor.fetchall()})
    finally:
        connection.close()
    missing = sorted(set(uuids) - set(mapping))
    if missing:
        raise RuntimeError(f"{len(missing)} Orthrus ground-truth UUIDs missing from database")
    ids = {mapping[uuid] for uuid in uuids}
    labels = {mapping[uuid]: labels_by_uuid[uuid] for uuid in uuids}
    return ids, labels


def validation_max(path: Path) -> float:
    maximum = -np.inf
    for name in listdir_sorted(str(path)):
        for chunk in pd.read_csv(path / name, usecols=["loss"], chunksize=500_000):
            maximum = max(maximum, float(chunk["loss"].max()))
    return maximum


def aggregate_test(path: Path, max_nodes: int):
    scores = np.full(max_nodes, -np.inf, dtype=np.float64)
    max_windows = np.full(max_nodes, -1, dtype=np.int32)
    first_seen = np.full(max_nodes, -1, dtype=np.int64)
    next_order = 0

    for window, name in enumerate(listdir_sorted(str(path))):
        for chunk in pd.read_csv(
            path / name,
            usecols=["loss", "srcnode", "dstnode"],
            chunksize=500_000,
        ):
            loss = chunk["loss"].to_numpy(dtype=np.float64, copy=False)
            src = chunk["srcnode"].to_numpy(dtype=np.int64, copy=False)
            dst = chunk["dstnode"].to_numpy(dtype=np.int64, copy=False)

            # Preserve the native evaluator's first-seen order (src then dst
            # for each event), which is also the stable tie order for top-K.
            encountered = np.empty(src.size * 2, dtype=np.int64)
            encountered[0::2] = src
            encountered[1::2] = dst
            unique, indices = np.unique(encountered, return_index=True)
            unique = unique[np.argsort(indices)]
            unseen = unique[first_seen[unique] < 0]
            if unseen.size:
                first_seen[unseen] = np.arange(
                    next_order, next_order + unseen.size, dtype=np.int64
                )
                next_order += int(unseen.size)

            nodes = np.concatenate([src, dst])
            values = np.concatenate([loss, loss])
            grouped = pd.DataFrame({"node": nodes, "loss": values}).groupby(
                "node", sort=False
            )["loss"].max()
            group_nodes = grouped.index.to_numpy(dtype=np.int64, copy=False)
            group_scores = grouped.to_numpy(dtype=np.float64, copy=False)
            improved = group_scores > scores[group_nodes]
            improved_nodes = group_nodes[improved]
            scores[improved_nodes] = group_scores[improved]
            max_windows[improved_nodes] = window
        if window % 25 == 0:
            log(f"Streaming node maxima: {window}/{len(listdir_sorted(str(path)))}")
    seen = np.flatnonzero(first_seen >= 0)
    seen = seen[np.argsort(first_seen[seen])]
    return seen, scores, max_windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-nodes", type=int, default=1_258_362)
    args = parser.parse_args()
    cfg = native_cfg()

    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)

    losses_root = Path(cfg.detection.gnn_testing._edge_losses_dir)
    val_path = losses_root / "val" / "model_epoch_1"
    test_path = losses_root / "test" / "model_epoch_1"
    threshold = validation_max(val_path)
    log(f"Native max-validation threshold: {threshold:.12g}")

    nodes, all_scores, max_windows = aggregate_test(test_path, args.max_nodes)
    ground_truth, ground_truth_paths = official_ground_truth(cfg)
    scores = all_scores[nodes]

    # Exact native ano rule: stable score sort, top 20, 2-cluster KMeans,
    # mark the cluster with the higher centroid as anomalous.
    ranked = np.argsort(scores, kind="stable")
    top = ranked[-int(cfg.detection.evaluation.node_evaluation.kmeans_top_K):]
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(
        scores[top].reshape(-1, 1)
    )
    high_cluster = int(np.argmax(kmeans.cluster_centers_.ravel()))
    predicted = np.zeros(nodes.size, dtype=np.int8)
    predicted[top[kmeans.labels_ == high_cluster]] = 1
    truth = np.fromiter(
        (int(int(node) in ground_truth) for node in nodes),
        dtype=np.int8,
        count=nodes.size,
    )

    results = {
        int(node): {
            "score": float(score),
            "tw_with_max_loss": int(window),
            "y_true": int(y_true),
            "y_hat": int(y_hat),
        }
        for node, score, window, y_true, y_hat in zip(
            nodes, scores, max_windows[nodes], truth, predicted
        )
    }
    stats = classifier_evaluation(truth.tolist(), predicted.tolist(), scores.tolist())

    tw_labels_path = (
        Path(cfg.graph_construction.build_graphs._tw_labels)
        / "tw_to_malicious_nodes.pkl"
    )
    malicious_windows = set()
    if tw_labels_path.exists():
        malicious_windows = set(int(value) for value in torch.load(tw_labels_path))
    false_positive_positions = np.flatnonzero((truth == 0) & (predicted == 1))
    stats["fp_in_malicious_tw_ratio"] = (
        float(np.mean([
            int(max_windows[nodes[position]]) in malicious_windows
            for position in false_positive_positions
        ]))
        if false_positive_positions.size
        else float("nan")
    )
    stats["native_threshold"] = float(threshold)
    stats["kmeans_centers"] = kmeans.cluster_centers_.ravel().tolist()

    full_predicted = (scores > threshold).astype(np.int8)
    full_results = {
        int(node): {
            "score": float(score),
            "tw_with_max_loss": int(window),
            "y_true": int(y_true),
            "y_hat": int(y_hat),
        }
        for node, score, window, y_true, y_hat in zip(
            nodes, scores, max_windows[nodes], truth, full_predicted
        )
    }
    full_stats = classifier_evaluation(
        truth.tolist(), full_predicted.tolist(), scores.tolist()
    )
    full_stats["native_threshold"] = float(threshold)

    output = Path(cfg.detection.evaluation.node_evaluation._precision_recall_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(results, output / "result_model_epoch_1.pth")
    torch.save(stats, output / "stats_model_epoch_1.pth")
    torch.save(full_results, output / "result_model_epoch_1_full.pth")
    torch.save(full_stats, output / "stats_model_epoch_1_full.pth")

    tp_rows = []
    for node, score, window, y_true, y_hat in zip(
        nodes, scores, max_windows[nodes], truth, predicted
    ):
        if y_true == 1 and y_hat == 1:
            tp_rows.append({
                "node_id": int(node),
                "time_window": int(window),
                "score": float(score),
                "path": ground_truth_paths.get(int(node), ""),
            })
    reproduction = Path("/home/reproduction_results")
    reproduction.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tp_rows).to_csv(
        reproduction / "theia_e3_native_ano_tp_nodes.csv", index=False
    )
    torch.save(stats, reproduction / "theia_e3_native_ano_stats.pth")
    full_tp_rows = [
        {
            "node_id": int(node),
            "time_window": int(window),
            "score": float(score),
            "path": ground_truth_paths.get(int(node), ""),
        }
        for node, score, window, y_true, y_hat in zip(
            nodes, scores, max_windows[nodes], truth, full_predicted
        )
        if y_true == 1 and y_hat == 1
    ]
    pd.DataFrame(full_tp_rows).to_csv(
        reproduction / "theia_e3_native_full_tp_nodes.csv", index=False
    )
    torch.save(full_stats, reproduction / "theia_e3_native_full_stats.pth")

    print(f"result={output / 'result_model_epoch_1.pth'}")
    print(f"stats={stats}")
    print(f"native_TP_nodes={tp_rows}")
    print(f"full_stats={full_stats}")
    print(f"full_native_TP_nodes={full_tp_rows}")


if __name__ == "__main__":
    main()
