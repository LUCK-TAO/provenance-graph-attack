"""Verify that the TRACE attack adapter reproduces MAGIC's native evaluator.

The fast ``cache`` stage only replays MAGIC's existing distance cache.  The
``query`` stage additionally evaluates the clean final TRACE graph once and
checks whether the attack adapter produces the same detector decisions.  It
never modifies a graph and never regenerates the five-graph score cache.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np


MAGIC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MAGIC_ROOT.parent
for entry in (str(REPO_ROOT), str(MAGIC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from native_evaluation import (  # noqa: E402
    build_trace_evaluation_layout,
    classification_metrics,
    select_magic_threshold,
)


EXPECTED_NATIVE = {
    "evaluation_size": 684107,
    "malicious_count": 68086,
    "threshold": 3598.5849737372982,
    "auc": 0.9998153809050484,
    "f1": 0.995736028228789,
    "precision": 0.9917104937282383,
    "recall": 0.9997943776987928,
    "tn": 615452,
    "fn": 14,
    "tp": 68072,
    "fp": 569,
}


def _configure_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _load_graph_sizes() -> list[int]:
    """Read graph sizes only; this performs no model inference."""
    from utils.loaddata import load_entity_level_dataset, load_metadata

    metadata = load_metadata("trace")
    sizes = []
    for index in range(int(metadata["n_test"])):
        graph = load_entity_level_dataset("trace", "test", index)
        sizes.append(int(graph.number_of_nodes()))
        del graph
    return sizes


def _load_original_baseline() -> dict:
    cache_path = MAGIC_ROOT / "eval_result" / "distance_save_trace.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing original MAGIC cache: {cache_path}")
    with cache_path.open("rb") as handle:
        saved = pickle.load(handle)
    if not isinstance(saved, (list, tuple)) or len(saved) != 2:
        raise ValueError("distance_save_trace.pkl is not the original MAGIC format")

    mean_distance = float(saved[0])
    distances = np.asarray(saved[1], dtype=np.float64)
    scores = distances / mean_distance

    from utils.loaddata import load_metadata

    metadata = load_metadata("trace")
    malicious = [int(value) for value in metadata["malicious"][0]]
    graph_sizes = _load_graph_sizes()
    layout = build_trace_evaluation_layout(malicious, graph_sizes)
    labels = np.asarray(layout["labels"], dtype=np.int8)
    if len(scores) != len(labels):
        raise RuntimeError(
            "Original TRACE cache/layout mismatch: "
            f"scores={len(scores)}, labels={len(labels)}"
        )

    threshold = select_magic_threshold("trace", labels, scores)
    metrics = classification_metrics(labels, scores, threshold)
    return {
        "cache_path": str(cache_path),
        "mean_distance": mean_distance,
        "distances": distances,
        "scores": scores,
        "labels": labels,
        "layout": layout,
        "graph_sizes": graph_sizes,
        "threshold": threshold,
        "metrics": metrics,
    }


def _difference(actual: float, expected: float) -> float:
    return abs(float(actual) - float(expected))


def verify_cache(metric_tolerance: float) -> tuple[dict, dict, bool]:
    baseline = _load_original_baseline()
    metrics = baseline["metrics"]
    checks = {
        "evaluation_size": len(baseline["labels"]),
        "malicious_count": int(np.sum(baseline["labels"])),
        "graph_sizes": baseline["graph_sizes"],
        "threshold": float(baseline["threshold"]),
        **{key: metrics[key] for key in (
            "auc", "f1", "precision", "recall", "tn", "fn", "tp", "fp"
        )},
    }
    exact_keys = ("evaluation_size", "malicious_count", "tn", "fn", "tp", "fp")
    float_keys = ("threshold", "auc", "f1", "precision", "recall")
    exact_ok = all(checks[key] == EXPECTED_NATIVE[key] for key in exact_keys)
    float_ok = all(
        _difference(checks[key], EXPECTED_NATIVE[key]) <= metric_tolerance
        for key in float_keys
    )
    passed = bool(exact_ok and float_ok)
    report = {
        "stage": "cache",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "expected": EXPECTED_NATIVE,
        "metric_tolerance": metric_tolerance,
    }
    return report, baseline, passed


def verify_clean_query(
    baseline: dict,
    device: str,
    metric_tolerance: float,
    adapter_mode: str,
    persist_on_pass: bool,
) -> tuple[dict, bool]:
    # Importing the detector is intentionally deferred so the cache-only stage
    # remains fast and does not initialize CUDA or load the large KNN object.
    from DL_detector import DLDetector
    from utils.loaddata import load_entity_level_dataset

    detector = DLDetector(
        dataset_name="trace",
        device=device,
        top_k=1,
        threshold_source="fixed",
        threshold=1.0,
        native_magic_compat=(adapter_mode == "native-reconstructed"),
    )
    if adapter_mode == "native-rebuilt":
        detector.rebuild_native_magic_knn_from_training()
    graph = load_entity_level_dataset("trace", "test", 4)
    result = detector.query(graph, return_tp=False)
    current_scores = np.fromiter(
        (result["all_scores"][index] for index in range(graph.number_of_nodes())),
        dtype=np.float64,
        count=graph.number_of_nodes(),
    )

    # DLDetector currently emits log1p(distance/current_mean) for TRACE.
    # Recover the underlying distance and project it into MAGIC's original
    # distance/original_mean score domain before comparing decisions.
    current_distances = np.expm1(current_scores) * float(detector.mean_distance)
    native_scale_scores = current_distances / float(baseline["mean_distance"])

    prefix = len(baseline["layout"]["early_malicious_global"])
    expected_test4 = np.asarray(baseline["scores"][prefix:], dtype=np.float64)
    if len(native_scale_scores) != len(expected_test4):
        raise RuntimeError(
            "Clean test4 score length mismatch: "
            f"query={len(native_scale_scores)}, original={len(expected_test4)}"
        )

    threshold = float(baseline["threshold"])
    original_alerts = expected_test4 >= threshold
    adapter_alerts = native_scale_scores >= threshold
    mismatch_count = int(np.count_nonzero(original_alerts != adapter_alerts))

    merged_scores = np.asarray(baseline["scores"], dtype=np.float64).copy()
    merged_scores[prefix:] = native_scale_scores
    merged_metrics = classification_metrics(
        baseline["labels"], merged_scores, threshold
    )
    original_metrics = baseline["metrics"]
    confusion_keys = ("tn", "fn", "tp", "fp")
    metric_keys = ("auc", "f1", "precision", "recall")
    confusion_ok = all(
        int(merged_metrics[key]) == int(original_metrics[key])
        for key in confusion_keys
    )
    metrics_ok = all(
        _difference(merged_metrics[key], original_metrics[key]) <= metric_tolerance
        for key in metric_keys
    )
    passed = bool(mismatch_count == 0 and confusion_ok and metrics_ok)

    persisted_path = None
    if persist_on_pass:
        if adapter_mode not in {"native-reconstructed", "native-rebuilt"}:
            raise ValueError(
                "--persist-on-pass requires a native reconstructed/rebuilt adapter"
            )
        if passed:
            persisted_path = detector.persist_native_magic_knn_baseline()

    absolute_error = np.abs(native_scale_scores - expected_test4)
    if len(expected_test4) > 1:
        correlation = float(np.corrcoef(expected_test4, native_scale_scores)[0, 1])
    else:
        correlation = float("nan")
    equivalent_log_threshold = math.log1p(
        threshold * float(baseline["mean_distance"]) / float(detector.mean_distance)
    )
    report = {
        "stage": "query",
        "status": "PASS" if passed else "FAIL",
        "adapter_mode": adapter_mode,
        "test_index": 4,
        "test4_nodes": int(graph.number_of_nodes()),
        "test4_edges": int(graph.number_of_edges()),
        "original_raw_threshold": threshold,
        "adapter_equivalent_log_threshold": equivalent_log_threshold,
        "original_mean_distance": float(baseline["mean_distance"]),
        "adapter_mean_distance": float(detector.mean_distance),
        "alert_mismatch_count": mismatch_count,
        "alert_mismatch_rate": mismatch_count / len(expected_test4),
        "score_mae_in_native_domain": float(np.mean(absolute_error)),
        "score_max_abs_error_in_native_domain": float(np.max(absolute_error)),
        "score_pearson_correlation": correlation,
        "original_metrics": original_metrics,
        "merged_clean_query_metrics": merged_metrics,
        "metric_tolerance": metric_tolerance,
        "native_knn_cache_persisted": bool(persisted_path),
        "persisted_path": persisted_path,
    }
    return report, passed


def _print_report(report: dict) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))


def main() -> int:
    _configure_console()
    parser = argparse.ArgumentParser(
        description="Automatically verify TRACE alignment with native MAGIC."
    )
    parser.add_argument(
        "--stage", choices=("cache", "query", "all"), default="cache",
        help="cache is fast; query/all additionally query clean test4 once.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--adapter-mode",
        choices=("native-reconstructed", "native-rebuilt", "current"),
        default="native-reconstructed",
        help=(
            "native-reconstructed restores MAGIC's original TRACE KNN "
            "standardization from _dl; native-rebuilt recomputes the KNN from "
            "the four training graphs; current diagnoses the existing _dl cache"
        ),
    )
    parser.add_argument(
        "--persist-on-pass",
        action="store_true",
        help=(
            "Atomically overwrite distance_save_trace_dl.pkl only when the "
            "native reconstructed adapter passes every alignment check."
        ),
    )
    parser.add_argument("--metric-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--report",
        default=str(MAGIC_ROOT / "eval_result" / "trace_native_alignment_report.json"),
    )
    args = parser.parse_args()
    if args.metric_tolerance < 0:
        raise ValueError("--metric-tolerance must be non-negative")

    cache_report, baseline, cache_passed = verify_cache(args.metric_tolerance)
    reports = {"cache": cache_report}
    _print_report(cache_report)

    query_passed = True
    if args.stage in {"query", "all"}:
        if not cache_passed:
            print("Cache replay failed; clean-query verification was not started.")
            query_passed = False
        else:
            query_report, query_passed = verify_clean_query(
                baseline, args.device, args.metric_tolerance,
                args.adapter_mode, args.persist_on_pass
            )
            reports["query"] = query_report
            _print_report(query_report)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2, allow_nan=True)
    print(f"Alignment report saved to: {report_path}")

    passed = cache_passed and query_passed
    print(f"TRACE native alignment: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
