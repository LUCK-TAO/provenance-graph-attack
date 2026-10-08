"""Build the unified-attack graph view from reproduced native THEIA_E3 results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent


def one_dir(pattern: str, expected_files: int) -> Path:
    candidates = [
        path for path in ROOT.glob(pattern)
        if path.is_dir() and len([item for item in path.iterdir() if item.is_file()]) == expected_files
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one {pattern!r} directory, found {candidates}")
    return candidates[0]


def one_file(pattern: str) -> Path:
    candidates = [path for path in ROOT.glob(pattern) if path.is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one {pattern!r} file, found {candidates}")
    return candidates[0]


def build(output: Path) -> None:
    embed_dir = one_dir(
        "artifacts/edge_featurization/THEIA_E3/embed_edges/*/edge_embeds/test", 203
    )
    loss_dir = one_dir(
        "artifacts/detection/gnn_testing/*/THEIA_E3/edge_losses/test/model_epoch_1", 203
    )
    result_path = one_file(
        "artifacts/detection/evaluation/*/THEIA_E3/precision_recall_dir/"
        "result_model_epoch_1_full.pth"
    )
    stats_path = one_file(
        "artifacts/detection/evaluation/*/THEIA_E3/precision_recall_dir/"
        "stats_model_epoch_1_full.pth"
    )
    target_csv = ROOT / "reproduction_results" / "theia_e3_native_full_tp_nodes.csv"
    targets = pd.read_csv(target_csv).sort_values(["time_window", "node_id"])
    target_windows = targets["time_window"].to_numpy(dtype=np.int64)
    selected_windows = np.unique(target_windows)

    native_result = torch.load(result_path, map_location="cpu")
    native_stats = torch.load(stats_path, map_location="cpu")
    embedded_files = sorted(path for path in embed_dir.iterdir() if path.is_file())
    loss_files = sorted(path for path in loss_dir.iterdir() if path.is_file())
    selected = [torch.load(embedded_files[int(index)], map_location="cpu") for index in selected_windows]

    src_global = torch.cat([data.src for data in selected]).numpy().astype(np.int64)
    dst_global = torch.cat([data.dst for data in selected]).numpy().astype(np.int64)
    raw_messages = torch.cat([data.msg for data in selected]).numpy().astype(np.float32)
    edge_times = torch.cat([data.t for data in selected]).numpy().astype(np.int64)
    edge_window_ids = np.concatenate([
        np.full(data.src.numel(), int(window), dtype=np.int64)
        for window, data in zip(selected_windows, selected)
    ])
    window_edge_counts = np.asarray([data.src.numel() for data in selected], dtype=np.int64)

    edge_types = raw_messages[:, 131:141].argmax(axis=1).astype(np.int64)
    src_types = raw_messages[:, :3].argmax(axis=1).astype(np.int64)
    dst_types = raw_messages[:, 141:144].argmax(axis=1).astype(np.int64)
    global_node_ids = np.unique(np.concatenate([src_global, dst_global]))
    global_to_local = {int(node): index for index, node in enumerate(global_node_ids)}
    local_src = np.fromiter(
        (global_to_local[int(node)] for node in src_global), dtype=np.int64, count=src_global.size
    )
    local_dst = np.fromiter(
        (global_to_local[int(node)] for node in dst_global), dtype=np.int64, count=dst_global.size
    )
    node_types = np.full(global_node_ids.size, -1, dtype=np.int64)
    node_types[local_src] = src_types
    node_types[local_dst] = dst_types
    if np.any(node_types < 0):
        raise RuntimeError("some local nodes have no observed native node type")

    result_ids = np.asarray(sorted(int(node) for node in native_result), dtype=np.int64)
    result_scores = np.asarray([native_result[int(node)]["score"] for node in result_ids], dtype=np.float64)
    result_truth = np.asarray([native_result[int(node)]["y_true"] for node in result_ids], dtype=np.int8)
    result_hat = np.asarray([native_result[int(node)]["y_hat"] for node in result_ids], dtype=np.int8)
    result_tw = np.asarray(
        [native_result[int(node)]["tw_with_max_loss"] for node in result_ids], dtype=np.int64
    )
    lookup = {int(node): index for index, node in enumerate(result_ids)}
    missing_local = [int(node) for node in global_node_ids if int(node) not in lookup]
    if missing_local:
        raise RuntimeError(f"local nodes missing from native result: {missing_local[:10]}")
    local_baseline_scores = np.asarray(
        [result_scores[lookup[int(node)]] for node in global_node_ids], dtype=np.float64
    )

    target_global_ids = targets["node_id"].to_numpy(dtype=np.int64)
    missing_targets = [int(node) for node in target_global_ids if int(node) not in global_to_local]
    if missing_targets:
        raise RuntimeError(f"native TP nodes absent from their scored windows: {missing_targets}")
    target_local_ids = np.asarray(
        [global_to_local[int(node)] for node in target_global_ids], dtype=np.int64
    )

    # Preserve the maximum clean score from every window that will not be
    # recomputed after an edit. Intervening windows are replayed by the worker.
    outside_max = np.zeros(global_node_ids.size, dtype=np.float64)
    selected_set = set(int(value) for value in selected_windows)
    for index, path in enumerate(loss_files):
        if index in selected_set:
            continue
        frame = pd.read_csv(path, usecols=["srcnode", "dstnode", "loss"])
        loss = frame["loss"].to_numpy(dtype=np.float64)
        for column in ("srcnode", "dstnode"):
            nodes = frame[column].to_numpy(dtype=np.int64)
            positions = np.searchsorted(global_node_ids, nodes)
            valid = (
                (positions >= 0)
                & (positions < global_node_ids.size)
                & (global_node_ids[np.minimum(positions, global_node_ids.size - 1)] == nodes)
            )
            np.maximum.at(outside_max, positions[valid], loss[valid])
        if index % 25 == 0:
            print(f"outside-window score scan: {index}/{len(loss_files)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        version=np.asarray([1], dtype=np.int64),
        dataset=np.asarray(["theia"], dtype="U16"),
        detector_dataset=np.asarray(["THEIA_E3"], dtype="U16"),
        alert_mode=np.asarray(["full"], dtype="U16"),
        native_threshold=np.asarray([native_stats["native_threshold"]], dtype=np.float64),
        source_result=np.asarray([str(result_path.relative_to(ROOT))], dtype="U256"),
        selected_windows=selected_windows,
        selected_window_files=np.asarray(
            [embedded_files[int(index)].name for index in selected_windows], dtype="U256"
        ),
        window_edge_counts=window_edge_counts,
        global_node_ids=global_node_ids,
        node_types=node_types,
        local_src=local_src,
        local_dst=local_dst,
        edge_types=edge_types,
        edge_times=edge_times,
        edge_window_ids=edge_window_ids,
        baseline_result_ids=result_ids,
        baseline_result_scores=result_scores,
        baseline_result_truth=result_truth,
        baseline_result_hat=result_hat,
        baseline_result_tw=result_tw,
        local_baseline_scores=local_baseline_scores,
        local_outside_max_scores=outside_max,
        target_global_ids=target_global_ids,
        target_local_ids=target_local_ids,
        target_windows=target_windows,
        node_type_names=np.asarray(["subject", "file", "netflow"], dtype="U32"),
        edge_type_names=np.asarray([
            "EVENT_CONNECT", "EVENT_EXECUTE", "EVENT_OPEN", "EVENT_READ",
            "EVENT_RECVFROM", "EVENT_RECVMSG", "EVENT_SENDMSG", "EVENT_SENDTO",
            "EVENT_WRITE", "EVENT_CLONE",
        ], dtype="U32"),
    )
    print(f"saved {output}")
    print(f"selected_windows={selected_windows.tolist()}")
    print(f"local nodes={global_node_ids.size}, local edges={local_src.size}")
    print(f"initial full-mode TP local IDs={target_local_ids.tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "attack_state" / "theia_e3_native_full_tp.npz",
    )
    build(parser.parse_args().output.resolve())
