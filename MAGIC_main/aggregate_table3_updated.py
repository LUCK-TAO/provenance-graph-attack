"""Aggregate the final Table 3 action-policy experiments.

The script intentionally reuses the Random + threshold-margin runs from Table 2
and combines them with the Table 3 baselines.  Metrics are first computed for
each seed and then summarized with the arithmetic mean and sample standard
deviation (ddof=1) across seeds 42, 43, and 44.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class StrategySpec:
    key: str
    label: str
    run_template: str
    color: str


STRATEGIES = (
    StrategySpec("no_attack", "No-Attack", "t3_no_attack_margin_k10_r5_seed{seed}", "#8A8F98"),
    StrategySpec("random", "Random", "t2_random_threshold_margin_k10_r5_seed{seed}", "#5B8FF9"),
    StrategySpec("greedy", "Greedy", "t3_greedy_margin_k10_r5_seed{seed}", "#F6BD16"),
    StrategySpec("linucb", "LinUCB", "t3_linucb_margin_k10_r5_seed{seed}", "#5AD8A6"),
    StrategySpec("double_dqn", "Double DQN", "t3_double_dqn_margin_k10_r5_seed{seed}", "#5D7092"),
    StrategySpec(
        "benign_mimicry",
        "Adapted Benign-Mimicry",
        "t3_benign_mimicry_margin_k10_r5_seed{seed}",
        "#F6903D",
    ),
    StrategySpec("ppo_online", "PPO", "t3_ppo_online_margin_k10_r5_seed{seed}", "#6F5EF9"),
)


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(keys))
        current = current[key]
    return current


def load_run(runs_dir: Path, spec: StrategySpec, seed: int) -> dict[str, Any]:
    run_name = spec.run_template.format(seed=seed)
    summary_path = runs_dir / run_name / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing Table 3 result: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    arguments = summary.get("arguments", {})
    result = nested(summary, "result")
    actual_seed = int(arguments.get("seed", re.search(r"seed(\d+)$", run_name).group(1)))
    if actual_seed != seed:
        raise ValueError(f"Seed mismatch in {summary_path}: expected {seed}, found {actual_seed}")

    expected = {
        "dataset": "theia",
        "detector": "magic",
        "target_policy": "threshold_margin",
        "top_k": 10,
        "max_rounds": 5,
    }
    for key, expected_value in expected.items():
        actual = arguments.get(key)
        if actual != expected_value:
            raise ValueError(
                f"Configuration mismatch in {summary_path}: {key}={actual!r}, "
                f"expected {expected_value!r}"
            )

    queries = int(result["total_detector_queries"])
    elapsed_seconds = float(result["attack_elapsed_seconds"])
    f1_before = float(nested(result, "initial_metrics", "f1"))
    f1_after = float(nested(result, "final_metrics", "f1"))
    aab = finite_or_none(result.get("average_attack_budget"))

    dasr_value = finite_or_none(result.get("direct_tp_asr"))
    if dasr_value is None:
        if spec.key != "no_attack":
            raise ValueError(f"Undefined DASR for attacking policy in {summary_path}")
        dasr_value = 0.0

    return {
        "strategy_key": spec.key,
        "strategy": spec.label,
        "seed": seed,
        "run_name": run_name,
        "dasr": dasr_value,
        "aab": aab,
        "aqt_seconds_per_query": elapsed_seconds / queries if queries > 0 else None,
        "f1_before": f1_before,
        "f1_after": f1_after,
        "f1_drop": f1_before - f1_after,
        "total_detector_queries": queries,
        "runtime_seconds": elapsed_seconds,
        "runtime_minutes": elapsed_seconds / 60.0,
        "direct_evasion_count": int(result.get("direct_native_tp_escaped", 0)),
        "attacked_initial_tp_count": int(result.get("attacked_initial_tp_count", 0)),
        "source_summary": str(summary_path.resolve()),
    }


def sample_stats(values: Iterable[float | None]) -> tuple[float | None, float | None, int]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None, None, 0
    return mean(clean), stdev(clean) if len(clean) > 1 else 0.0, len(clean)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "dasr",
        "aab",
        "aqt_seconds_per_query",
        "f1_before",
        "f1_after",
        "f1_drop",
        "total_detector_queries",
        "runtime_seconds",
        "runtime_minutes",
    )
    output: list[dict[str, Any]] = []
    for spec in STRATEGIES:
        group = [row for row in rows if row["strategy_key"] == spec.key]
        if [row["seed"] for row in group] != list(SEEDS):
            raise ValueError(f"Incomplete or unordered seeds for {spec.label}: {group}")
        record: dict[str, Any] = {
            "strategy_key": spec.key,
            "strategy": spec.label,
            "n_seeds": len(group),
        }
        for metric in metrics:
            avg, sd, n = sample_stats(row[metric] for row in group)
            record[f"{metric}_mean"] = avg
            record[f"{metric}_std"] = sd
            record[f"{metric}_n"] = n
        output.append(record)
    return output


def fmt(value: float | None, digits: int) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], aggregated: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_fields = [
        "strategy",
        "seed",
        "dasr",
        "aab",
        "aqt_seconds_per_query",
        "f1_before",
        "f1_after",
        "f1_drop",
        "total_detector_queries",
        "runtime_seconds",
        "runtime_minutes",
        "direct_evasion_count",
        "attacked_initial_tp_count",
        "run_name",
        "source_summary",
    ]
    write_csv(output_dir / "table03_action_policy_by_seed.csv", rows, per_seed_fields)

    aggregate_fields = ["strategy", "n_seeds"]
    for metric in (
        "dasr",
        "aab",
        "aqt_seconds_per_query",
        "f1_before",
        "f1_after",
        "f1_drop",
        "total_detector_queries",
        "runtime_seconds",
        "runtime_minutes",
    ):
        aggregate_fields.extend((f"{metric}_mean", f"{metric}_std", f"{metric}_n"))
    write_csv(output_dir / "table03_action_policy_statistics.csv", aggregated, aggregate_fields)

    paper_rows = []
    for record in aggregated:
        dasr = f"{100.0 * record['dasr_mean']:.2f} +/- {100.0 * record['dasr_std']:.2f}%"
        aab = (
            "N/A"
            if record["aab_mean"] is None
            else f"{record['aab_mean']:.2f} +/- {record['aab_std']:.2f}"
        )
        aqt = f"{record['aqt_seconds_per_query_mean']:.2f} +/- {record['aqt_seconds_per_query_std']:.2f}"
        f1 = (
            f"{record['f1_before_mean']:.4f} +/- {record['f1_before_std']:.4f} "
            f"(down {record['f1_drop_mean']:.4f} +/- {record['f1_drop_std']:.4f})"
        )
        paper_rows.append(
            {
                "Strategy": record["strategy"],
                "DASR (mean +/- std)": dasr,
                "AAB (mean +/- std)": aab,
                "AQT, s/query (mean +/- std)": aqt,
                "F1 before (drop), mean +/- std": f1,
            }
        )
    write_csv(
        output_dir / "table03_action_policy_paper.csv",
        paper_rows,
        list(paper_rows[0].keys()),
    )

    with (output_dir / "table03_action_policy_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "definition": {
                    "seeds": list(SEEDS),
                    "sample_standard_deviation": True,
                    "random_source": "Table 2 Random + threshold_margin runs",
                    "dasr": "direct_tp_asr from each run",
                    "aab": "average_attack_budget from each run; undefined for No-Attack",
                    "aqt": "attack_elapsed_seconds / total_detector_queries",
                    "f1_drop": "initial_metrics.f1 - final_metrics.f1",
                },
                "per_seed": rows,
                "aggregate": aggregated,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=20, labelsize=8.5)


def compact_tick_labels(labels: list[str]) -> list[str]:
    return [
        "Adapted\nBenign-Mimicry"
        if label == "Adapted Benign-Mimicry"
        else label
        for label in labels
    ]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=320 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(output_dir: Path, aggregated: list[dict[str, Any]]) -> None:
    # Keep No-Attack in the exported tables for reproducibility, but omit it
    # from the action-policy figure because it has no action-selection policy.
    aggregated = [row for row in aggregated if row["strategy"] != "No-Attack"]
    labels = [row["strategy"] for row in aggregated]
    tick_labels = compact_tick_labels(labels)
    color_by_label = {spec.label: spec.color for spec in STRATEGIES}
    colors = [color_by_label[label] for label in labels]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)

    panels = (
        (axes[0, 0], "dasr", 100.0, "Attack success rate", "ASR (%)"),
        (axes[0, 1], "aab", 1.0, "Average attack budget", "AAB (edits / evaded TP)"),
        (axes[1, 0], "aqt_seconds_per_query", 1.0, "Average query time", "AQT (seconds / query)"),
        (axes[1, 1], "runtime_minutes", 1.0, "Total attack runtime", "Total runtime (minutes)"),
    )
    for ax, metric, scale, title, ylabel in panels:
        means = np.array(
            [np.nan if row[f"{metric}_mean"] is None else row[f"{metric}_mean"] * scale for row in aggregated],
            dtype=float,
        )
        errors = np.array(
            [0.0 if row[f"{metric}_std"] is None else row[f"{metric}_std"] * scale for row in aggregated],
            dtype=float,
        )
        bars = ax.bar(x, means, yerr=errors, capsize=3, color=colors, edgecolor="#FFFFFF", linewidth=0.7)
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, tick_labels)
        style_axis(ax)
        if metric == "aab" and np.isnan(means[0]):
            ax.text(0, max(np.nanmax(means) * 0.04, 1.0), "N/A", ha="center", va="bottom", fontsize=8.5)
        for bar, value in zip(bars, means):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                )

    fig.suptitle("Action-policy comparison on MAGIC–THEIA (R=5, K=10)", fontsize=14, fontweight="bold")
    save_figure(fig, output_dir, "fig_table03_metric_comparison")


def plot_time_cost(output_dir: Path, rows: list[dict[str, Any]], aggregated: list[dict[str, Any]]) -> None:
    aggregated = [row for row in aggregated if row["strategy"] != "No-Attack"]
    rows = [row for row in rows if row["strategy"] != "No-Attack"]
    labels = [row["strategy"] for row in aggregated]
    tick_labels = compact_tick_labels(labels)
    color_by_label = {spec.label: spec.color for spec in STRATEGIES}
    colors = [color_by_label[label] for label in labels]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.7), constrained_layout=True)

    runtime_mean = np.array([row["runtime_minutes_mean"] for row in aggregated])
    runtime_std = np.array([row["runtime_minutes_std"] for row in aggregated])
    bars = axes[0].bar(x, runtime_mean, yerr=runtime_std, capsize=3, color=colors)
    axes[0].set_title("Total attack runtime", fontsize=12, fontweight="semibold")
    axes[0].set_ylabel("Minutes (mean +/- sample std)")
    axes[0].set_xticks(x, tick_labels)
    style_axis(axes[0])
    for bar, value in zip(bars, runtime_mean):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}", ha="center", va="bottom", fontsize=7.4)

    for spec in STRATEGIES:
        if spec.label == "No-Attack":
            continue
        group = [row for row in rows if row["strategy_key"] == spec.key]
        axes[1].plot(
            [row["seed"] for row in group],
            [row["runtime_minutes"] for row in group],
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=spec.color,
            label=spec.label,
        )
    axes[1].set_title("Runtime variation across seeds", fontsize=12, fontweight="semibold")
    axes[1].set_ylabel("Minutes")
    axes[1].set_xlabel("Seed")
    axes[1].set_xticks(SEEDS)
    axes[1].grid(color="#D9DEE7", linewidth=0.7, alpha=0.8)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=7.8, ncol=2, loc="upper center")

    fig.suptitle("Time-cost comparison by action policy", fontsize=14, fontweight="bold")
    save_figure(fig, output_dir, "fig_table03_time_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("attack_results/paper_suite_theia_final_v1/runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("attack_results/paper_suite_theia_final_v1/table3_updated"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [load_run(args.runs_dir, spec, seed) for spec in STRATEGIES for seed in SEEDS]
    aggregated = aggregate(rows)
    write_outputs(args.output_dir, rows, aggregated)
    plot_metrics(args.output_dir, aggregated)
    plot_time_cost(args.output_dir, rows, aggregated)
    print(f"Validated {len(rows)} runs and wrote Table 3 outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
