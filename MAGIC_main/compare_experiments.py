"""Aggregate structured attack runs into a CSV table for baseline analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "run",
    "dataset",
    "detector",
    "seed",
    "action_policy",
    "ppo_selection_mode",
    "target_policy",
    "candidate_policy",
    "reward_mode",
    "experiment_variant",
    "top_k",
    "max_rounds",
    "checkpoint_updates_start",
    "checkpoint_updates_end",
    "policy_updates_start",
    "policy_updates_end",
    "mean_normalized_entropy",
    "initial_normalized_entropy",
    "final_normalized_entropy",
    "mean_max_action_probability",
    "final_max_action_probability",
    "mean_top1_top2_probability_gap",
    "final_top1_top2_probability_gap",
    "mean_kl_to_uniform",
    "final_kl_to_uniform",
    "indirect_reward_nonzero_rate",
    "initial_tp",
    "final_tp",
    "signed_tp_reduction",
    "direct_tp_asr",
    "evasion_amplification",
    "graph_modification_rate",
    "average_attack_budget",
    "attacked_unique_nodes",
    "attacked_initial_tp",
    "direct_tp_escaped",
    "attacked_initial_fp",
    "direct_fp_escaped",
    "direct_fp_evasion_rate",
    "direct_alert_evasion_rate",
    "indirect_tp_escaped",
    "indirect_tp_evasion_rate",
    "tp_reduction_per_query",
    "tp_reduction_per_action",
    "edits_per_reduced_tp",
    "initial_fp",
    "final_fp",
    "auc_before",
    "auc_after",
    "total_actions",
    "attack_queries",
    "total_detector_queries",
    "reused_query_count",
    "elapsed_seconds",
    "delta_nodes",
    "delta_edges",
    "stop_reason",
)


def _nested(mapping, *keys, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def summary_to_row(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = payload.get("arguments", {})
    result = payload.get("result", {})
    return {
        "run": path.parent.name,
        "dataset": args.get("dataset"),
        "detector": args.get("detector"),
        "seed": result.get("seed", args.get("seed")),
        "action_policy": result.get("action_policy", args.get("action_policy")),
        "ppo_selection_mode": result.get(
            "ppo_selection_mode", args.get("ppo_selection_mode")
        ),
        "target_policy": result.get("target_policy", args.get("target_policy")),
        "candidate_policy": result.get(
            "candidate_policy", args.get("candidate_policy")
        ),
        "reward_mode": result.get("reward_mode", args.get("reward_mode")),
        "experiment_variant": result.get(
            "experiment_variant", args.get("experiment_variant")
        ),
        "top_k": args.get("top_k"),
        "max_rounds": args.get("max_rounds"),
        "checkpoint_updates_start": result.get("checkpoint_updates_start"),
        "checkpoint_updates_end": result.get("checkpoint_updates_end"),
        "policy_updates_start": result.get("policy_updates_start"),
        "policy_updates_end": result.get("policy_updates_end"),
        "mean_normalized_entropy": _nested(
            result, "policy_health", "mean_normalized_entropy"
        ),
        "initial_normalized_entropy": _nested(
            result, "policy_health", "initial_normalized_entropy"
        ),
        "final_normalized_entropy": _nested(
            result, "policy_health", "final_normalized_entropy"
        ),
        "mean_max_action_probability": _nested(
            result, "policy_health", "mean_max_action_probability"
        ),
        "final_max_action_probability": _nested(
            result, "policy_health", "final_max_action_probability"
        ),
        "mean_top1_top2_probability_gap": _nested(
            result, "policy_health", "mean_top1_top2_probability_gap"
        ),
        "final_top1_top2_probability_gap": _nested(
            result, "policy_health", "final_top1_top2_probability_gap"
        ),
        "mean_kl_to_uniform": _nested(
            result, "policy_health", "mean_kl_to_uniform"
        ),
        "final_kl_to_uniform": _nested(
            result, "policy_health", "final_kl_to_uniform"
        ),
        "indirect_reward_nonzero_rate": _nested(
            result, "policy_health", "indirect_reward_nonzero_rate"
        ),
        "initial_tp": result.get("initial_native_tp"),
        "final_tp": result.get("final_native_tp"),
        "signed_tp_reduction": result.get("tp_reduction_count"),
        "direct_tp_asr": result.get("direct_tp_asr"),
        "evasion_amplification": result.get("evasion_amplification"),
        "graph_modification_rate": result.get("graph_modification_rate"),
        "average_attack_budget": result.get("average_attack_budget"),
        "attacked_unique_nodes": result.get("attacked_unique_nodes"),
        "attacked_initial_tp": result.get("attacked_initial_tp_count"),
        "direct_tp_escaped": result.get("direct_native_tp_escaped"),
        "attacked_initial_fp": result.get("attacked_initial_fp_count"),
        "direct_fp_escaped": result.get("direct_initial_fp_escaped"),
        "direct_fp_evasion_rate": result.get("direct_fp_evasion_rate"),
        "direct_alert_evasion_rate": result.get("target_evasion_rate"),
        "indirect_tp_escaped": result.get("indirect_native_tp_escaped"),
        "indirect_tp_evasion_rate": result.get("indirect_tp_evasion_rate"),
        "tp_reduction_per_query": result.get("tp_reduction_per_query"),
        "tp_reduction_per_action": result.get("tp_reduction_per_action"),
        "edits_per_reduced_tp": result.get("edits_per_reduced_tp"),
        "initial_fp": _nested(result, "initial_metrics", "fp"),
        "final_fp": _nested(result, "final_metrics", "fp"),
        "auc_before": _nested(result, "initial_metrics", "auc"),
        "auc_after": _nested(result, "final_metrics", "auc"),
        "total_actions": result.get("total_actions"),
        "attack_queries": result.get("attack_queries_excluding_baseline"),
        "total_detector_queries": result.get("total_detector_queries"),
        "reused_query_count": result.get("reused_query_count"),
        "elapsed_seconds": result.get("attack_elapsed_seconds"),
        "delta_nodes": result.get("delta_nodes"),
        "delta_edges": result.get("delta_edges"),
        "stop_reason": result.get("stop_reason"),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate attack experiment summaries")
    parser.add_argument("--results-dir", type=Path, default=Path("attack_results"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summaries = sorted(args.results_dir.glob("*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No summary.json files found under {args.results_dir}")
    rows = [summary_to_row(path) for path in summaries]
    output = args.output or args.results_dir / "comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {output.resolve()}")


if __name__ == "__main__":
    main()
