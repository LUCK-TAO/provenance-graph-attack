"""Offline audit of TRACE target-policy TP/FP composition.

This utility reuses the clean native MAGIC evaluation cache.  Ground-truth
metadata is read only after target selection and is never passed to the
ranking function, so the audit does not leak labels into an attack policy.
No detector query or graph modification is performed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

from attack_targeting import rank_alerts
from native_evaluation import localize_global_nodes
from utils.loaddata import load_metadata


POLICIES = (
    "threshold_margin",
    "hybrid_margin_random",
    "random_alert",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit clean TRACE alert targets against metadata offline."
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-index", type=int, default=4)
    parser.add_argument("--margin-fraction", type=float, default=0.5)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).resolve().parent
        / "eval_result"
        / "native_evaluation_trace_dl.pkl",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if not 0.0 <= args.margin_fraction <= 1.0:
        raise ValueError("--margin-fraction must be within [0, 1]")

    with args.cache.open("rb") as handle:
        cache = pickle.load(handle)
    graph_sizes = [int(value) for value in cache["graph_sizes"]]
    if not 0 <= args.test_index < len(graph_sizes):
        raise IndexError("--test-index is outside the cached TRACE graph layout")

    metadata = load_metadata("trace")
    global_malicious, _ = metadata["malicious"]
    local_malicious = set(
        localize_global_nodes(global_malicious, graph_sizes, args.test_index)
    )

    # MAGIC's TRACE evaluation vector stores early-partition malicious scores
    # first, followed by every node score from the final graph.
    if args.test_index != len(graph_sizes) - 1:
        raise ValueError(
            "This zero-query audit currently supports the final TRACE graph only"
        )
    scores = cache["evaluation_scores"]
    final_size = graph_sizes[-1]
    final_scores = scores[-final_size:]
    all_scores = {
        node_id: float(final_scores[node_id]) for node_id in range(final_size)
    }
    threshold = float(cache["threshold"])

    report = {
        "dataset": "trace",
        "test_index": args.test_index,
        "seed": args.seed,
        "top_k": args.top_k,
        "threshold": threshold,
        "margin_fraction": args.margin_fraction,
        "ground_truth_usage": "offline_audit_only",
        "detector_queries": 0,
        "policies": {},
    }
    for policy in POLICIES:
        targets = rank_alerts(
            all_scores,
            threshold,
            strategy=policy,
            # Match DirectedTopKAttackAgent.target_rng exactly.  Each policy is
            # audited as an independent run starting from the same seed.
            rng=random.Random(args.seed + 2001),
            limit=args.top_k,
            margin_fraction=args.margin_fraction,
        )[: args.top_k]
        tp_count = sum(node_id in local_malicious for node_id in targets)
        rows = [
            {
                "node": int(node_id),
                "score": float(all_scores[node_id]),
                "margin": float(all_scores[node_id] - threshold),
                "metadata_tp": bool(node_id in local_malicious),
            }
            for node_id in targets
        ]
        report["policies"][policy] = {
            "target_count": len(targets),
            "metadata_tp": int(tp_count),
            "initial_fp": int(len(targets) - tp_count),
            "tp_ratio": float(tp_count / len(targets)) if targets else None,
            "targets": rows,
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Audit report saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
