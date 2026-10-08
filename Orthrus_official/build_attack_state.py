"""Build the editable attack-framework view of native Orthrus CADETS_E3.

The detector continues to use the official TemporalData artifacts and weights.
This file only builds a compact, locally re-indexed DGL view for candidate
selection and records the clean native scores needed by the black-box adapter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent


def _single_dir(pattern: str, *, expected_files: int | None = None) -> Path:
    candidates = [path for path in ROOT.glob(pattern) if path.is_dir()]
    if expected_files is not None:
        candidates = [
            path for path in candidates
            if len([item for item in path.iterdir() if item.is_file()]) == expected_files
        ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one directory for {pattern!r}, found {candidates}")
    return candidates[0]


def _native_result(mode: str) -> tuple[Path, dict, float]:
    candidates = sorted(ROOT.glob(
        "artifacts/detection/evaluation/*/CADETS_E3/"
        "precision_recall_dir/result_model_epoch_1.pth"
    ))
    expected = {"ano": (8, 0), "full": (32, 97)}[mode]
    for path in candidates:
        result = torch.load(path, map_location="cpu")
        predictions = list(result.values())
        tp = sum(int(row["y_true"] == 1 and row["y_hat"] == 1) for row in predictions)
        fp = sum(int(row["y_true"] == 0 and row["y_hat"] == 1) for row in predictions)
        if (tp, fp) == expected:
            stats = torch.load(path.with_name("stats_model_epoch_1.pth"), map_location="cpu")
            threshold = float(stats.get("native_threshold", float("nan")))
            if mode == "full" and not np.isfinite(threshold):
                val_dirs = [
                    candidate for candidate in ROOT.glob(
                        "artifacts/detection/gnn_testing/*/CADETS_E3/"
                        "edge_losses/val/model_epoch_1"
                    )
                    if candidate.is_dir()
                ]
                if len(val_dirs) != 1:
                    raise RuntimeError(
                        f"expected one CADETS validation-loss directory, found {val_dirs}"
                    )
                threshold = max(
                    float(pd.read_csv(item, usecols=["loss"])["loss"].max())
                    for item in val_dirs[0].iterdir()
                    if item.is_file()
                )
            return path, result, threshold
    raise RuntimeError(
        f"the reproduced native {mode} result (TP={expected[0]}, FP={expected[1]}) "
        "was not found"
    )


def build(output: Path, mode: str) -> None:
    embed_dir = _single_dir(
        "artifacts/edge_featurization/CADETS_E3/embed_edges/*/edge_embeds/test",
        expected_files=333,
    )
    loss_dir = _single_dir(
        "artifacts/detection/gnn_testing/*/CADETS_E3/edge_losses/test/model_epoch_1",
        expected_files=333,
    )
    result_path, native_result, native_threshold = _native_result(mode)

    if mode == "ano":
        target_csv = ROOT / "reproduction_results" / "cadets_e3_native_ano_tp_nodes.csv"
        target_frame = pd.read_csv(target_csv)
    else:
        target_rows = [
            {
                "node_id": int(node),
                "anomaly_score": float(row["score"]),
                "time_window": int(row["tw_with_max_loss"]),
                "y_true": int(row["y_true"]),
                "y_hat": int(row["y_hat"]),
            }
            for node, row in native_result.items()
            if int(row["y_true"]) == 1 and int(row["y_hat"]) == 1
        ]
        target_frame = pd.DataFrame(target_rows).sort_values("node_id")
        target_csv = ROOT / "reproduction_results" / "cadets_e3_native_full_tp_nodes.csv"
        target_frame.to_csv(target_csv, index=False)

    selected_windows = tuple(sorted(
        set(target_frame["time_window"].to_numpy(dtype=np.int64).tolist())
    ))
    replay_windows = set(range(min(selected_windows), max(selected_windows) + 1))

    embedded_files = sorted(path for path in embed_dir.iterdir() if path.is_file())
    loss_files = sorted(path for path in loss_dir.iterdir() if path.is_file())
    selected = [
        torch.load(embedded_files[index], map_location="cpu")
        for index in selected_windows
    ]

    src_global = torch.cat([data.src for data in selected]).numpy().astype(np.int64)
    dst_global = torch.cat([data.dst for data in selected]).numpy().astype(np.int64)
    raw_messages = torch.cat([data.msg for data in selected]).numpy().astype(np.float32)
    edge_times = torch.cat([data.t for data in selected]).numpy().astype(np.int64)
    window_ids = np.concatenate([
        np.full(data.src.numel(), window, dtype=np.int64)
        for window, data in zip(selected_windows, selected)
    ])
    window_edge_counts = np.asarray([data.src.numel() for data in selected], dtype=np.int64)

    # Raw Orthrus message layout: src type (3), src embedding (128), edge
    # type (10), dst type (3), dst embedding (128).
    edge_type = raw_messages[:, 131:141].argmax(axis=1).astype(np.int64)
    src_type = raw_messages[:, :3].argmax(axis=1).astype(np.int64)
    dst_type = raw_messages[:, 141:144].argmax(axis=1).astype(np.int64)

    global_node_ids = np.unique(np.concatenate([src_global, dst_global]))
    global_to_local = {int(node): index for index, node in enumerate(global_node_ids.tolist())}
    local_src = np.fromiter(
        (global_to_local[int(node)] for node in src_global), dtype=np.int64, count=len(src_global)
    )
    local_dst = np.fromiter(
        (global_to_local[int(node)] for node in dst_global), dtype=np.int64, count=len(dst_global)
    )
    node_types = np.full(len(global_node_ids), -1, dtype=np.int64)
    node_types[local_src] = src_type
    node_types[local_dst] = dst_type
    if np.any(node_types < 0):
        raise RuntimeError("some local nodes have no observed Orthrus node type")

    result_ids = np.asarray(sorted(int(node) for node in native_result), dtype=np.int64)
    result_scores = np.asarray(
        [float(native_result[int(node)]["score"]) for node in result_ids], dtype=np.float64
    )
    result_truth = np.asarray(
        [int(native_result[int(node)]["y_true"]) for node in result_ids], dtype=np.int8
    )
    result_hat = np.asarray(
        [int(native_result[int(node)]["y_hat"]) for node in result_ids], dtype=np.int8
    )
    result_tw = np.asarray(
        [int(native_result[int(node)]["tw_with_max_loss"]) for node in result_ids],
        dtype=np.int64,
    )
    result_lookup = {int(node): index for index, node in enumerate(result_ids.tolist())}
    local_baseline_scores = np.asarray(
        [result_scores[result_lookup[int(node)]] for node in global_node_ids], dtype=np.float64
    )

    target_global_ids = target_frame["node_id"].to_numpy(dtype=np.int64)
    missing_targets = [int(node) for node in target_global_ids if int(node) not in global_to_local]
    if missing_targets:
        raise RuntimeError(
            f"native TP nodes are absent from their selected windows: {missing_targets}"
        )
    target_local_ids = np.asarray(
        [global_to_local[int(node)] for node in target_global_ids], dtype=np.int64
    )
    target_windows = target_frame["time_window"].to_numpy(dtype=np.int64)

    # The full replay interval is re-scored on every query. Preserve each local
    # node's maximum loss outside that interval so the adapter can rebuild the
    # native max-edge-loss node score without rerunning all 333 windows.
    outside_max = np.zeros(len(global_node_ids), dtype=np.float64)
    local_nodes = set(int(node) for node in global_node_ids)
    for index, path in enumerate(loss_files):
        # Every query replays the complete temporal interval from the earliest
        # target window to the latest. Scores in that interval must therefore
        # be recomputed instead of retained as immutable outside-window maxima.
        if index in replay_windows:
            continue
        frame = pd.read_csv(path, usecols=["srcnode", "dstnode", "loss"])
        src = frame["srcnode"].to_numpy(dtype=np.int64)
        dst = frame["dstnode"].to_numpy(dtype=np.int64)
        loss = frame["loss"].to_numpy(dtype=np.float64)
        mask = np.isin(src, global_node_ids) | np.isin(dst, global_node_ids)
        for s, d, value in zip(src[mask], dst[mask], loss[mask]):
            if int(s) in local_nodes:
                local = global_to_local[int(s)]
                outside_max[local] = max(outside_max[local], float(value))
            if int(d) in local_nodes:
                local = global_to_local[int(d)]
                outside_max[local] = max(outside_max[local], float(value))
        if index % 50 == 0:
            print(f"outside-window score scan: {index}/{len(loss_files)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        version=np.asarray([1], dtype=np.int64),
        dataset=np.asarray(["cadets"], dtype="U16"),
        alert_mode=np.asarray([mode], dtype="U16"),
        native_threshold=np.asarray([native_threshold], dtype=np.float64),
        source_result=np.asarray([str(result_path.relative_to(ROOT))], dtype="U256"),
        selected_windows=np.asarray(selected_windows, dtype=np.int64),
        selected_window_files=np.asarray(
            [embedded_files[index].name for index in selected_windows], dtype="U256"
        ),
        window_edge_counts=window_edge_counts,
        global_node_ids=global_node_ids,
        node_types=node_types,
        local_src=local_src,
        local_dst=local_dst,
        edge_types=edge_type,
        edge_times=edge_times,
        edge_window_ids=window_ids,
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
    print(f"local nodes={len(global_node_ids)}, local edges={len(local_src)}")
    print(f"initial TP local IDs={target_local_ids.tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--mode", choices=("ano", "full"), default="ano")
    arguments = parser.parse_args()
    output = arguments.output or (
        ROOT / "artifacts" / "attack_state"
        / ("cadets_e3_native_full_tp.npz" if arguments.mode == "full" else "cadets_e3_tw228_229.npz")
    )
    build(output.resolve(), arguments.mode)
