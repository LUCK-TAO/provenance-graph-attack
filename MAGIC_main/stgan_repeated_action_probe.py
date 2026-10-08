"""Cumulative repeated-action upper-bound probe for the native STGAN adapter."""

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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAGIC_ROOT = PROJECT_ROOT / "MAGIC_main"
for entry in (str(PROJECT_ROOT), str(MAGIC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from Detector.STGAN import STGANDetector
from Detector.STGAN_State import STGANSemanticState
from action_schema import get_action_schema
from attack_directed_topk import ModificationLog
from candidate_selector import CandidateSelector
from greedy_policy import GreedyActionPolicy
from utils.loaddata import load_entity_level_dataset


LOGGER = logging.getLogger("stgan.repeated_action_probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cadets", choices=("cadets",))
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", type=int, default=760)
    parser.add_argument("--action-id", type=int, default=2)
    parser.add_argument("--macro-length", type=int, default=5)
    parser.add_argument("--macro-count", type=int, default=20)
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
        default=PROJECT_ROOT / "attack_results/stgan_cadets_repeated_action_probe",
    )
    return parser.parse_args()


def save(run_dir: Path, payload: dict) -> None:
    (run_dir / "trajectory.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields = [
        "macro_index",
        "cumulative_operations",
        "score_before",
        "score_after",
        "batch_score_drop",
        "cumulative_score_drop",
        "relative_margin_closed",
        "escaped",
        "query_seconds",
        "endpoints",
    ]
    with (run_dir / "trajectory.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload.get("trajectory", []):
            item = {name: row.get(name) for name in fields}
            item["endpoints"] = json.dumps(
                row.get("endpoints", []), ensure_ascii=False
            )
            writer.writerow(item)


def main() -> None:
    args = parse_args()
    if args.macro_length <= 0 or args.macro_count <= 0:
        raise ValueError("macro length and count must be positive")
    if not args.semantic_state.is_file():
        raise FileNotFoundError(args.semantic_state)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / (
        f"stgan_{args.dataset}_target{args.target}_action{args.action_id}_"
        f"m{args.macro_length}x{args.macro_count}_seed{args.seed}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "run.log", encoding="utf-8"),
        ],
    )
    LOGGER.info("run_dir=%s", run_dir)

    current_graph = load_entity_level_dataset(
        args.dataset, "test", args.test_index
    )
    clean_state = STGANSemanticState.load(args.semantic_state)
    state = replace(
        clean_state,
        base_embeddings=clean_state.base_embeddings.copy(),
        token_counts=clean_state.token_counts.copy(),
        native_edge_index=clean_state.native_edge_index,
        native_edge_time=clean_state.native_edge_time,
        _added_edges=[],
        _added_times=[],
    )
    detector = STGANDetector(
        dataset_name=args.dataset,
        device=args.device,
        top_k=1,
        threshold_source="native",
    )
    detector.set_evaluation_context(args.test_index)
    detector.attach_semantic_state(state)
    baseline_started = time.perf_counter()
    baseline = detector.initialize_tp_baseline(current_graph)
    baseline_seconds = time.perf_counter() - baseline_started
    scores = {
        int(node): float(score)
        for node, score in baseline["all_scores"].items()
    }
    if args.target not in detector.initial_true_tp_nodes:
        raise ValueError(
            f"target {args.target} is not a clean native true positive"
        )

    schema = get_action_schema(args.dataset, allow_observed_fallback=False)
    if schema is None:
        raise RuntimeError(f"no audited static schema for {args.dataset}")
    action = schema.get(args.action_id)
    selector = CandidateSelector(
        max_hop=2,
        local_ratio=0.5,
        action_schema=schema,
        dataset_name=args.dataset,
        selection_mode="semantic_mixed",
        seed=args.seed,
    )
    selector.build_type_index(current_graph)

    threshold = float(detector.best_threshold)
    initial_score = float(scores[args.target])
    initial_margin = max(abs(initial_score - threshold), 1e-12)
    payload = {
        "configuration": {
            "dataset": args.dataset,
            "test_index": args.test_index,
            "device": str(detector.device),
            "seed": args.seed,
            "target": args.target,
            "action_id": args.action_id,
            "action_name": action.name,
            "edge_name": action.edge_name,
            "macro_length": args.macro_length,
            "macro_count": args.macro_count,
            "maximum_operations": args.macro_length * args.macro_count,
            "candidate_policy": "semantic_mixed",
            "threshold": threshold,
            "initial_score": initial_score,
            "semantic_state": str(args.semantic_state.resolve()),
            "baseline_cache_hit": bool(detector.last_baseline_cache_hit),
            "baseline_seconds": baseline_seconds,
            "isolation": "clean graph; no edits from other targets",
        },
        "trajectory": [],
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "completed_at": None,
    }
    save(run_dir, payload)

    previous_score = initial_score
    for macro_index in range(1, args.macro_count + 1):
        mod_log = ModificationLog(current_graph)
        endpoints = []
        for _ in range(args.macro_length):
            pools = selector.legal_candidate_pools(
                current_graph,
                args.target,
                top_k=50,
                anomaly_scores=scores,
                pending_edges=mod_log.pending_edge_pairs(),
            )
            candidates = pools.get(args.action_id, [])
            if not candidates:
                raise RuntimeError(
                    f"WRITE candidate pool exhausted at macro {macro_index}"
                )
            candidate_index = GreedyActionPolicy.select_candidate_index(
                candidates, scores, current_graph
            )
            candidate = int(candidates[candidate_index])
            resolved = selector.resolve_operation(
                current_graph, args.target, args.action_id, candidate
            )
            src, dst = int(resolved["src"]), int(resolved["dst"])
            mod_log.add_edge(
                src,
                dst,
                args.action_id,
                edge_type=int(resolved["edge_type"]),
            )
            detector.record_attack_modification(
                kind="edge",
                src=src,
                dst=dst,
                action_type=args.action_id,
                edge_type=int(resolved["edge_type"]),
                graph=current_graph,
            )
            endpoints.append(
                {
                    "src": src,
                    "dst": dst,
                    "candidate": candidate,
                    "src_type": int(resolved["src_type"]),
                    "dst_type": int(resolved["dst_type"]),
                }
            )

        modified_graph = mod_log.apply_modifications()
        query_started = time.perf_counter()
        result = detector.query(modified_graph, node_indices=[args.target])
        query_seconds = time.perf_counter() - query_started
        current_score = float(result["all_scores"][args.target])
        escaped = args.target not in set(result["native_alerts"])
        scores = {
            int(node): float(score)
            for node, score in result["all_scores"].items()
        }
        row = {
            "macro_index": macro_index,
            "cumulative_operations": macro_index * args.macro_length,
            "score_before": previous_score,
            "score_after": current_score,
            "batch_score_drop": previous_score - current_score,
            "cumulative_score_drop": initial_score - current_score,
            "relative_margin_closed": (
                (initial_score - current_score) / initial_margin
            ),
            "escaped": bool(escaped),
            "query_seconds": query_seconds,
            "endpoints": endpoints,
        }
        payload["trajectory"].append(row)
        save(run_dir, payload)
        LOGGER.info(
            "macro=%d/%d operations=%d score=%.9f->%.9f "
            "batch_drop=%+.9f cumulative_drop=%+.9f margin_closed=%.6f%% "
            "escaped=%s query=%.2fs",
            macro_index,
            args.macro_count,
            row["cumulative_operations"],
            previous_score,
            current_score,
            row["batch_score_drop"],
            row["cumulative_score_drop"],
            100.0 * row["relative_margin_closed"],
            escaped,
            query_seconds,
        )
        current_graph = modified_graph
        previous_score = current_score
        gc.collect()

    payload["completed_at"] = datetime.now().isoformat(timespec="seconds")
    payload["result"] = {
        "final_score": previous_score,
        "threshold": threshold,
        "total_score_drop": initial_score - previous_score,
        "relative_margin_closed": (
            (initial_score - previous_score) / initial_margin
        ),
        "escaped": bool(payload["trajectory"][-1]["escaped"]),
        "total_operations": args.macro_length * args.macro_count,
        "detector_queries": args.macro_count,
        "total_query_seconds": sum(
            float(row["query_seconds"]) for row in payload["trajectory"]
        ),
    }
    save(run_dir, payload)
    LOGGER.info("completed result=%s output=%s", payload["result"], run_dir)
    print(f"RESULT_DIR={run_dir.resolve()}")


if __name__ == "__main__":
    main()
