"""Preflight and optional native-alignment check for the ProvGuard adapter.

Examples:
  python ProvGuard_main/verify_provguard_setup.py --dataset cadets
  python ProvGuard_main/verify_provguard_setup.py --dataset cadets --run-query
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Detector.ProvGuard import PROVGUARD_ROOT, ProvGuardDetector
from MAGIC_main.utils.loaddata import load_entity_level_dataset, load_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ProvGuard attack prerequisites")
    parser.add_argument("--dataset", default="cadets", choices=("cadets", "theia", "trace"))
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--run-query",
        action="store_true",
        help="Run the expensive clean detector query and create the native cache.",
    )
    return parser.parse_args()


def checkpoint_candidates(dataset: str, explicit: Path | None) -> list[Path]:
    values = []
    if explicit is not None:
        values.append(explicit)
    values.extend(
        [
            PROVGUARD_ROOT / "data" / "res_models" / f"graphcl_model_{dataset}.pth",
            PROVGUARD_ROOT / "checkpoints" / f"checkpoint-{dataset}.pt",
            PROVGUARD_ROOT / "checkpoints" / f"checkpoint-{dataset}.pth",
        ]
    )
    return [value if value.is_absolute() else ROOT / value for value in values]


def main() -> int:
    args = parse_args()
    metadata = load_metadata(args.dataset)
    graph = load_entity_level_dataset(args.dataset, "test", args.test_index)
    params_path = (
        PROVGUARD_ROOT
        / "data"
        / "eval_result"
        / f"{args.dataset}_trained_params.pkl"
    )
    candidates = checkpoint_candidates(args.dataset, args.checkpoint)
    checkpoint = next((path for path in candidates if path.is_file()), None)

    report = {
        "dataset": args.dataset,
        "test_index": args.test_index,
        "graph": {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
            "node_feature_dim": int(graph.ndata["attr"].shape[1]),
            "edge_feature_dim": int(graph.edata["attr"].shape[1]),
        },
        "metadata": {
            "node_feature_dim": int(metadata["node_feature_dim"]),
            "edge_feature_dim": int(metadata["edge_feature_dim"]),
            "n_train": int(metadata["n_train"]),
            "n_test": int(metadata["n_test"]),
        },
        "checkpoint": {
            "found": checkpoint is not None,
            "path": str(checkpoint) if checkpoint else None,
            "searched": [str(path) for path in candidates],
        },
        "scoring_parameters": {"found": params_path.is_file(), "path": str(params_path)},
    }

    failures = []
    graph_dims = (
        report["graph"]["node_feature_dim"],
        report["graph"]["edge_feature_dim"],
    )
    metadata_dims = (
        report["metadata"]["node_feature_dim"],
        report["metadata"]["edge_feature_dim"],
    )
    if graph_dims != metadata_dims:
        failures.append("graph feature dimensions differ from metadata")
    if checkpoint is None:
        failures.append("pretrained ProvGuard encoder checkpoint is missing")
    if not params_path.is_file():
        failures.append("native KMeans scoring parameters are missing")
    else:
        with params_path.open("rb") as handle:
            params = pickle.load(handle)
        required = {
            "cluster_centers",
            "global_mean_distance",
            "x_train_mean",
            "x_train_std",
        }
        missing = sorted(required - set(params))
        report["scoring_parameters"].update(
            {
                "missing_keys": missing,
                "cluster_centers_shape": list(np.asarray(params.get("cluster_centers", [])).shape),
                "x_train_mean_shape": list(np.asarray(params.get("x_train_mean", [])).shape),
                "x_train_std_shape": list(np.asarray(params.get("x_train_std", [])).shape),
                "global_mean_distance": float(params.get("global_mean_distance", float("nan"))),
            }
        )
        if missing:
            failures.append(f"scoring parameters miss keys: {missing}")

    if checkpoint is not None:
        detector = ProvGuardDetector(
            dataset_name=args.dataset,
            device=args.device,
            top_k=1,
            checkpoint_path=str(checkpoint),
            threshold_source="native",
        )
        detector.set_evaluation_context(args.test_index)
        report["checkpoint"]["strict_state_dict_load"] = True
        if args.run_query:
            baseline = detector.initialize_tp_baseline(graph)
            report["native_alignment"] = {
                "threshold": float(baseline["threshold"]),
                "alerts": len(detector.initial_alert_nodes),
                "true_tp": int(detector.true_tp_count),
                "false_positive": int(detector.false_positive_count),
                "metrics": detector.initial_metrics,
                "cache_hit": bool(detector.last_baseline_cache_hit),
            }

    report["status"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
