"""Controlled single-action sensitivity audit for the native STGAN adapter.

Each target/action probe starts from the same clean graph and semantic state.
The probe applies one macro action (five legal edge insertions by default),
performs exactly one native detector query, and records the target score change.
This is a diagnostic and does not train or update an attack policy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAGIC_ROOT = PROJECT_ROOT / "MAGIC_main"
for entry in (str(PROJECT_ROOT), str(MAGIC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from Detector.STGAN import STGANDetector
from Detector.STGAN_State import STGANSemanticState
from action_schema import get_action_schema
from candidate_selector import CandidateSelector
from greedy_policy import GreedyActionPolicy
from utils.loaddata import load_entity_level_dataset


LOGGER = logging.getLogger("stgan.single_action_sensitivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe every legal STGAN action independently from the same clean "
            "CADETS state."
        )
    )
    parser.add_argument("--dataset", default="cadets", choices=("cadets",))
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--macro-length", type=int, default=5)
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        required=True,
        help="Initial TP node IDs to audit.",
    )
    parser.add_argument(
        "--semantic-state",
        type=Path,
        default=(
            PROJECT_ROOT
            / "STGAN_master/data/cadets/cadets_stgan_official_attack_test0.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "attack_results/stgan_cadets_single_action_sensitivity",
    )
    return parser.parse_args()


def clean_probe_state(clean: STGANSemanticState) -> STGANSemanticState:
    """Return an isolated mutable view while sharing immutable native edges."""
    return replace(
        clean,
        base_embeddings=clean.base_embeddings.copy(),
        token_counts=clean.token_counts.copy(),
        # These arrays are read-only in the adapter and dominate state size.
        native_edge_index=clean.native_edge_index,
        native_edge_time=clean.native_edge_time,
        _added_edges=[],
        _added_times=[],
    )


def write_outputs(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = payload.get("probes", [])
    fieldnames = [
        "target",
        "action_id",
        "action_name",
        "edge_name",
        "initial_score",
        "final_score",
        "score_drop",
        "relative_margin_drop",
        "threshold",
        "escaped",
        "primitive_operations",
        "query_seconds",
        "endpoints",
    ]
    with (run_dir / "results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = {key: row.get(key) for key in fieldnames}
            item["endpoints"] = json.dumps(
                row.get("endpoints", []), ensure_ascii=False
            )
            writer.writerow(item)

    best_rows = []
    for target in sorted({int(row["target"]) for row in rows}):
        candidates = [row for row in rows if int(row["target"]) == target]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda row: (
                float(row["score_drop"]),
                -int(row["action_id"]),
            ),
        )
        best_rows.append(
            {
                "target": target,
                "initial_score": best["initial_score"],
                "best_action_id": best["action_id"],
                "best_action_name": best["action_name"],
                "best_final_score": best["final_score"],
                "best_score_drop": best["score_drop"],
                "best_relative_margin_drop": best["relative_margin_drop"],
                "any_single_action_escape": any(
                    bool(row["escaped"]) for row in candidates
                ),
                "legal_actions_tested": len(candidates),
            }
        )
    summary_fields = [
        "target",
        "initial_score",
        "best_action_id",
        "best_action_name",
        "best_final_score",
        "best_score_drop",
        "best_relative_margin_drop",
        "any_single_action_escape",
        "legal_actions_tested",
    ]
    with (run_dir / "target_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(best_rows)


def main() -> None:
    args = parse_args()
    if args.macro_length <= 0:
        raise ValueError("--macro-length must be positive")
    if not args.semantic_state.is_file():
        raise FileNotFoundError(args.semantic_state)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / (
        f"stgan_{args.dataset}_single_action_m{args.macro_length}_"
        f"seed{args.seed}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )
    LOGGER.info("run_dir=%s", run_dir)

    graph = load_entity_level_dataset(args.dataset, "test", args.test_index)
    clean_state = STGANSemanticState.load(args.semantic_state)
    detector = STGANDetector(
        dataset_name=args.dataset,
        device=args.device,
        top_k=max(len(args.targets), 1),
        threshold_source="native",
    )
    detector.set_evaluation_context(args.test_index)
    detector.attach_semantic_state(clean_state)

    baseline_started = time.perf_counter()
    baseline = detector.initialize_tp_baseline(graph)
    baseline_seconds = time.perf_counter() - baseline_started
    scores = {int(key): float(value) for key, value in baseline["all_scores"].items()}
    native_tp = set(int(node) for node in detector.initial_true_tp_nodes)
    targets = list(dict.fromkeys(int(node) for node in args.targets))
    invalid = [node for node in targets if node not in native_tp]
    if invalid:
        raise ValueError(f"requested targets are not clean native TPs: {invalid}")

    # CADETS formal attacks use the audited static CADETS action vocabulary.
    # Do not substitute the graph-observed fallback used by THEIA/TRACE.
    schema = get_action_schema(args.dataset, allow_observed_fallback=False)
    if schema is None:
        raise RuntimeError(f"no audited static action schema for {args.dataset}")
    selector = CandidateSelector(
        max_hop=2,
        local_ratio=0.5,
        action_schema=schema,
        dataset_name=args.dataset,
        selection_mode="semantic_mixed",
        seed=args.seed,
    )
    selector.build_type_index(graph)
    threshold = float(detector.best_threshold)

    payload = {
        "configuration": {
            "dataset": args.dataset,
            "test_index": args.test_index,
            "device": str(detector.device),
            "seed": args.seed,
            "macro_length": args.macro_length,
            "candidate_policy": "semantic_mixed",
            "threshold": threshold,
            "semantic_state": str(args.semantic_state.resolve()),
            "targets": targets,
            "clean_native_tp_count": len(native_tp),
            "baseline_cache_hit": bool(detector.last_baseline_cache_hit),
            "baseline_seconds": baseline_seconds,
            "probe_isolation": "clean graph and clean semantic state per target-action",
        },
        "probes": [],
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "completed_at": None,
    }
    write_outputs(run_dir, payload)

    total_planned = 0
    legal_by_target = {}
    for target in targets:
        pools = selector.legal_candidate_pools(
            graph,
            target,
            top_k=50,
            anomaly_scores=scores,
            pending_edges=set(),
        )
        legal = sorted(
            action_id
            for action_id, candidates in pools.items()
            if len(candidates) >= args.macro_length
        )
        legal_by_target[target] = legal
        total_planned += len(legal)
        LOGGER.info(
            "target=%d initial_score=%.9f legal_actions=%s",
            target,
            scores[target],
            legal,
        )

    completed = 0
    for target in targets:
        initial_score = float(scores[target])
        for action_id in legal_by_target[target]:
            action = schema.get(action_id)
            detector.attach_semantic_state(clean_probe_state(clean_state))
            pending_edges = set()
            endpoints = []

            for _ in range(args.macro_length):
                candidates = selector.select_candidates(
                    graph,
                    target,
                    action_id,
                    top_k=50,
                    anomaly_scores=scores,
                    pending_edges=pending_edges,
                )
                if not candidates:
                    raise RuntimeError(
                        f"candidate pool exhausted: target={target}, action={action_id}"
                    )
                candidate_index = GreedyActionPolicy.select_candidate_index(
                    candidates, scores, graph
                )
                candidate = int(candidates[candidate_index])
                resolved = selector.resolve_operation(
                    graph, target, action_id, candidate
                )
                src, dst = int(resolved["src"]), int(resolved["dst"])
                pending_edges.add((src, dst))
                endpoints.append(
                    {
                        "src": src,
                        "dst": dst,
                        "candidate": candidate,
                        "src_type": int(resolved["src_type"]),
                        "dst_type": int(resolved["dst_type"]),
                    }
                )
                detector.record_attack_modification(
                    kind="edge",
                    src=src,
                    dst=dst,
                    action_type=action_id,
                    edge_type=int(resolved["edge_type"]),
                    graph=graph,
                )

            query_started = time.perf_counter()
            result = detector.query(graph, node_indices=[target])
            query_seconds = time.perf_counter() - query_started
            final_score = float(result["all_scores"][target])
            score_drop = initial_score - final_score
            margin = max(abs(initial_score - threshold), 1e-12)
            escaped = target not in set(result["native_alerts"])
            completed += 1
            row = {
                "target": target,
                "action_id": int(action_id),
                "action_name": action.name,
                "edge_name": action.edge_name,
                "initial_score": initial_score,
                "final_score": final_score,
                "score_drop": score_drop,
                "relative_margin_drop": score_drop / margin,
                "threshold": threshold,
                "escaped": bool(escaped),
                "primitive_operations": args.macro_length,
                "query_seconds": query_seconds,
                "endpoints": endpoints,
            }
            payload["probes"].append(row)
            write_outputs(run_dir, payload)
            LOGGER.info(
                "probe=%d/%d target=%d action=%s score=%.9f->%.9f "
                "drop=%+.9f relative=%+.6f escaped=%s query=%.2fs",
                completed,
                total_planned,
                target,
                action.name,
                initial_score,
                final_score,
                score_drop,
                score_drop / margin,
                escaped,
                query_seconds,
            )
            gc.collect()

    detector.attach_semantic_state(clean_state)
    payload["completed_at"] = datetime.now().isoformat(timespec="seconds")
    payload["total_probes"] = len(payload["probes"])
    payload["total_query_seconds"] = sum(
        float(row["query_seconds"]) for row in payload["probes"]
    )
    write_outputs(run_dir, payload)
    LOGGER.info("completed probes=%d output=%s", len(payload["probes"]), run_dir)
    print(f"RESULT_DIR={run_dir.resolve()}")


if __name__ == "__main__":
    main()
