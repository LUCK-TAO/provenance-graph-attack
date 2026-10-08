"""Generate paper-ready tables and figures from structured attack runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

try:
    from .native_evaluation import localize_global_nodes
except ImportError:
    from native_evaluation import localize_global_nodes


METRIC_DEFINITIONS = [
    ("DirectTP-ASR", "|(S∩T0)\\T1|/|S∩T0|", "Higher", "Near 1 with non-zero denominator", "Final evasion among directly attacked initial TPs"),
    ("EvasionAmplification", "|T0\\T1|/|S|", "Higher", "Above 1 indicates indirect TP evasion", "All evaded initial TPs per actually attacked unique node"),
    ("GraphModifyRate", "(deltaE+deltaV)/(E0+V0)", "Lower", "As small as possible at equal effectiveness", "Relative graph perturbation"),
    ("AverageAttackBudget", "(deltaE+deltaV)/|T0\\T1|", "Lower", "As small as possible; N/A with zero success", "Average edits per successfully evaded initial TP"),
]

PRIMARY_METRICS = [
    ("direct_tp_asr", "direct_tp_asr"),
    ("evasion_amplification", "evasion_amplification"),
    ("graph_modify_rate", "graph_modification_rate"),
    ("average_attack_budget", "average_attack_budget"),
]

# Current paper-facing metrics for the LinUCB component ablation.  Keep the
# query count alongside AQT: AQT measures amortized wall-clock cost per query,
# whereas total_detector_queries is the actual black-box query budget consumed.
LINUCB_ABLATION_METRICS = [
    ("direct_tp_asr", "direct_tp_asr"),
    ("average_attack_budget", "average_attack_budget"),
    ("average_query_time_seconds", "average_query_time_seconds"),
    ("f1_drop", "f1_drop"),
    ("total_detector_queries", "total_detector_queries"),
]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _strategy(payload):
    result = payload.get("result", {})
    args = payload.get("arguments", {})
    action = result.get("action_policy", args.get("action_policy", "unknown"))
    target = result.get("target_policy", args.get("target_policy", "unknown"))
    labels = {
        "no_attack": "No attack",
        "random": "Random",
        "greedy": "Greedy",
        "linucb": "LinUCB",
        "double_dqn": "Double DQN",
        "benign_mimicry": "Benign-Mimicry",
        "ppo_online": "Online PPO",
        "ppo_frozen": "Frozen PPO",
    }
    targets = {
        "threshold_margin": "Margin",
        "highest_score": "Highest",
        "random_alert": "Random target",
    }
    return f"{labels.get(action, action)} + {targets.get(target, target)}"


def _load_runs(results_dir):
    runs = []
    for path in sorted(Path(results_dir).glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_summary_path"] = path
        payload["_run_dir"] = path.parent
        payload["_strategy"] = _strategy(payload)
        result = payload.setdefault("result", {})
        if (
            not _finite(result.get("attack_elapsed_minutes"))
            and _finite(result.get("attack_elapsed_seconds"))
        ):
            result["attack_elapsed_minutes"] = (
                float(result["attack_elapsed_seconds"]) / 60.0
            )
        total_queries = result.get("total_detector_queries")
        elapsed_seconds = result.get("attack_elapsed_seconds")
        if _finite(total_queries) and float(total_queries) > 0 and _finite(elapsed_seconds):
            result["average_query_time_seconds"] = (
                float(elapsed_seconds) / float(total_queries)
            )
        else:
            result["average_query_time_seconds"] = None
        initial_f1 = result.get("initial_metrics", {}).get("f1")
        final_f1 = result.get("final_metrics", {}).get("f1")
        result["f1_drop"] = (
            float(initial_f1) - float(final_f1)
            if _finite(initial_f1) and _finite(final_f1)
            else None
        )
        runs.append(payload)
    return runs


def _result(run, key, default=None):
    value = run.get("result", {})
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _metric_stats(runs, key):
    values = [float(_result(run, key)) for run in runs if _finite(_result(run, key))]
    if not values:
        return None, None, 0
    return mean(values), stdev(values) if len(values) > 1 else 0.0, len(values)


def _group_runs(runs, key_fn=lambda run: run["_strategy"]):
    groups = defaultdict(list)
    for run in runs:
        groups[key_fn(run)].append(run)
    return dict(sorted(groups.items()))


def _argument(run, key, default=None):
    return run.get("arguments", {}).get(key, default)


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(groups, metrics, group_columns=("strategy",)):
    rows = []
    for group_key, strategy_runs in groups.items():
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_columns, values))
        row["runs"] = len(strategy_runs)
        for column, key in metrics:
            avg, sd, n = _metric_stats(strategy_runs, key)
            row[f"{column}_mean"] = avg
            row[f"{column}_std"] = sd
            row[f"{column}_n"] = n
        rows.append(row)
    return rows


def _attach_cost_columns(rows, groups, group_columns):
    """Attach exactly two paper-facing cost columns as mean +/- std."""
    summaries = {}
    for group_key, group_runs in groups.items():
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        query_mean, query_std, _ = _metric_stats(
            group_runs, "total_detector_queries"
        )
        runtime_mean, runtime_std, _ = _metric_stats(
            group_runs, "attack_elapsed_minutes"
        )
        summaries[values] = {
            "detector_queries": (
                None if query_mean is None
                else f"{query_mean:.3f} +/- {query_std:.3f}"
            ),
            "runtime_minutes": (
                None if runtime_mean is None
                else f"{runtime_mean:.3f} +/- {runtime_std:.3f}"
            ),
        }
    for row in rows:
        key = tuple(row[column] for column in group_columns)
        row.update(summaries.get(key, {}))


def _mean_and_error(row, key):
    value = row.get(key)
    if _finite(value):
        std_key = key.replace("_mean", "_std")
        error = row.get(std_key)
        return float(value), float(error) if _finite(error) else 0.0
    if isinstance(value, str) and "+/-" in value:
        average, error = value.split("+/-", 1)
        return float(average.strip()), float(error.strip())
    return np.nan, 0.0


def _primary_metric_panels(groups, output_base):
    plt = _mpl()
    labels = list(groups)
    if not labels:
        return
    specifications = [
        ("DirectTP-ASR", "direct_tp_asr", 100.0, "%"),
        ("EvasionAmplification", "evasion_amplification", 1.0, "ratio"),
        ("GraphModifyRate", "graph_modification_rate", 100.0, "%"),
        ("AverageAttackBudget", "average_attack_budget", 1.0, "edits / TP"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for ax, (title, key, factor, unit) in zip(axes.flat, specifications):
        averages, errors = [], []
        for label in labels:
            avg, sd, _ = _metric_stats(groups[label], key)
            averages.append(np.nan if avg is None else avg * factor)
            errors.append(0.0 if sd is None else sd * factor)
        x = np.arange(len(labels))
        ax.bar(x, averages, yerr=errors, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, output_base)


def _table2_cost_by_seed(runs, output_base):
    """Plot target-policy runtime costs without hiding per-seed variation."""
    if not runs:
        return
    plt = _mpl()
    groups = defaultdict(list)
    dataset_detector_pairs = {
        (
            _argument(run, "dataset", "unknown"),
            _argument(run, "detector", "unknown"),
        )
        for run in runs
    }
    include_context = len(dataset_detector_pairs) > 1
    target_labels = {
        "threshold_margin": "Margin",
        "highest_score": "Highest score",
        "random_alert": "Random target",
    }
    for run in runs:
        target = _result(
            run, "target_policy", _argument(run, "target_policy", "unknown")
        )
        if include_context:
            label = (
                f"{_argument(run, 'dataset', 'unknown')}/"
                f"{_argument(run, 'detector', 'unknown')}/"
                f"{target_labels.get(target, target)}"
            )
        else:
            label = target_labels.get(target, target)
        groups[label].append(run)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for label, group_runs in sorted(groups.items()):
        points = []
        for run in group_runs:
            seed = _argument(run, "seed", _result(run, "seed"))
            elapsed_seconds = _result(run, "attack_elapsed_seconds")
            detector_queries = _result(run, "total_detector_queries")
            if (
                seed is None
                or not _finite(elapsed_seconds)
                or not _finite(detector_queries)
                or float(detector_queries) <= 0
            ):
                continue
            points.append(
                (
                    int(seed),
                    float(elapsed_seconds) / 60.0,
                    float(elapsed_seconds) / float(detector_queries),
                )
            )
        points.sort(key=lambda item: item[0])
        if not points:
            continue
        seeds = [item[0] for item in points]
        axes[0].plot(
            seeds, [item[1] for item in points],
            marker="o", linewidth=1.8, label=label,
        )
        axes[1].plot(
            seeds, [item[2] for item in points],
            marker="o", linewidth=1.8, label=label,
        )

    axes[0].set_title("Total attack runtime by seed")
    axes[0].set_ylabel("minutes")
    axes[1].set_title("Runtime per detector query by seed")
    axes[1].set_ylabel("seconds / query")
    for ax in axes:
        ax.set_xlabel("seed")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    _save_figure(fig, output_base)


def _budget_metric_panels(rows, output_base, x_key="top_k"):
    if not rows:
        return
    plt = _mpl()
    specifications = [
        ("DirectTP-ASR", "direct_tp_asr_mean", 100.0, "%"),
        ("EvasionAmplification", "evasion_amplification_mean", 1.0, "ratio"),
        ("GraphModifyRate", "graph_modify_rate_mean", 100.0, "%"),
        ("AverageAttackBudget", "average_attack_budget_mean", 1.0, "edits / TP"),
        ("DetectorQueries", "detector_queries", 1.0, "queries"),
        ("RuntimeMinutes", "runtime_minutes", 1.0, "minutes"),
    ]
    series = defaultdict(list)
    for row in rows:
        label = str(row.get("action_policy", "strategy"))
        series[label].append(row)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    for ax, (title, key, factor, unit) in zip(axes.flat, specifications):
        for label, items in sorted(series.items()):
            items = sorted(items, key=lambda row: float(row[x_key]))
            xs = [float(row[x_key]) for row in items]
            values = [_mean_and_error(row, key) for row in items]
            ys = [average * factor for average, _ in values]
            errors = [error * factor for _, error in values]
            ax.errorbar(xs, ys, yerr=errors, marker="o", capsize=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("Top-K nodes per round")
        ax.set_ylabel(unit)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    _save_figure(fig, output_base)


def _detector_metric_panels(groups, output_base):
    if not groups:
        return
    plt = _mpl()
    metrics = [
        ("AUC", "auc"), ("F1", "f1"),
        ("Precision", "precision"), ("Recall", "recall"),
    ]
    labels = list(groups)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (title, key) in zip(axes.flat, metrics):
        before = []
        after = []
        for label in labels:
            initial, _, _ = _metric_stats(groups[label], f"initial_metrics.{key}")
            final, _, _ = _metric_stats(groups[label], f"final_metrics.{key}")
            before.append(np.nan if initial is None else initial)
            after.append(np.nan if final is None else final)
        ax.bar(x - 0.18, before, 0.36, label="Before")
        ax.bar(x + 0.18, after, 0.36, label="After")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    _save_figure(fig, output_base)


def _generate_ppo_learning_curve(runs, output_dir):
    """Compare reward trajectories of all online adaptive action policies."""
    trajectories = defaultdict(list)
    adaptive_policies = {"linucb", "double_dqn", "ppo_online"}
    for run in runs:
        policy = _result(
            run, "action_policy", _argument(run, "action_policy")
        )
        if policy not in adaptive_policies:
            continue
        path = run["_run_dir"] / "trajectory.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rewards = [
            float(item.get("total_reward", 0.0))
            for item in payload.get("queries", [])
            if item.get("kind") == "attack_feedback"
        ]
        if rewards:
            trajectories[policy].append(rewards)
    if not trajectories:
        return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = {
        "linucb": "LinUCB",
        "double_dqn": "Double DQN",
        "ppo_online": "Online PPO",
    }
    for policy, series_group in sorted(trajectories.items()):
        length = max(len(values) for values in series_group)
        means, deviations = [], []
        for index in range(length):
            values = [
                series[index] for series in series_group
                if index < len(series)
            ]
            means.append(mean(values))
            deviations.append(
                stdev(values) if len(values) > 1 else 0.0
            )
        x = np.arange(1, length + 1)
        y = np.asarray(means)
        error = np.asarray(deviations)
        line = ax.plot(x, y, label=labels.get(policy, policy))[0]
        ax.fill_between(
            x, y - error, y + error,
            color=line.get_color(), alpha=0.15,
        )
    ax.axhline(0, color="0.35", lw=1)
    ax.set_xlabel("Detector feedback query")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir / "fig09_adaptive_learning_curves")


def _generate_reward_component_figure(runs, output_dir):
    components = defaultdict(lambda: defaultdict(list))
    for run in runs:
        variant = _result(
            run, "experiment_variant", _argument(run, "experiment_variant", "auto")
        )
        path = run["_run_dir"] / "trajectory.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("queries", []):
            if item.get("kind") != "attack_feedback":
                continue
            components[variant]["score"].append(
                float(item.get("direct_contribution", 0.0))
            )
            components[variant]["crossing"].append(
                float(item.get("crossing_contribution", 0.0))
            )
            components[variant]["indirect"].append(
                float(item.get("indirect_contribution", 0.0))
            )
            components[variant]["penalty"].append(
                -float(item.get("perturbation_penalty", 0.0))
            )
    if not components:
        return
    labels = sorted(components)
    x = np.arange(len(labels))
    width = 0.19
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 4.8))
    entries = [
        ("Target score", "score"),
        ("Target crossing", "crossing"),
        ("Indirect effect", "indirect"),
        ("Edit penalty", "penalty"),
    ]
    for offset, (display, key) in enumerate(entries):
        values = [
            mean(components[label][key]) if components[label][key] else 0.0
            for label in labels
        ]
        centered_offset = offset - (len(entries) - 1) / 2
        ax.bar(x + centered_offset * width, values, width, label=display)
    ax.axhline(0, color="0.35", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Mean reward contribution per feedback batch")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir / "fig10_reward_components")


def _save_figure(fig, output_base):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="This figure includes Axes that are not compatible"
        )
        fig.tight_layout()
    fig.savefig(str(output_base) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(output_base) + ".pdf", bbox_inches="tight")
    _mpl().close(fig)


def _bar_figure(groups, metrics, ylabel, output_base, percent=False):
    plt = _mpl()
    labels = list(groups)
    if not labels:
        return
    x = np.arange(len(labels))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(max(7.2, len(labels) * 1.25), 4.6))
    for index, (display, key) in enumerate(metrics):
        values, errors = [], []
        for label in labels:
            avg, sd, _ = _metric_stats(groups[label], key)
            factor = 100.0 if percent else 1.0
            values.append(np.nan if avg is None else avg * factor)
            errors.append(0.0 if sd is None else sd * factor)
        positions = x - 0.4 + width / 2 + index * width
        ax.bar(positions, values, width, yerr=errors, capsize=3, label=display)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, output_base)


def _generate_workflow(output_dir):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.axis("off")
    nodes = [
        (0.02, 0.64, "Input graph"),
        (0.18, 0.64, "MAGIC query\nfixed threshold"),
        (0.37, 0.64, "Target selector\nmargin / highest / random"),
        (0.59, 0.64, "Action policy\nheuristic / LinUCB / DDQN / PPO"),
        (0.80, 0.64, "Legal directed\ngraph edits"),
        (0.59, 0.17, "Score-drop + crossing\n- edit-cost reward"),
        (0.80, 0.17, "One macro action\n= five same-type edges"),
        (0.95, 0.40, "Final fixed-threshold\nevaluation"),
    ]
    boxes = {}
    for x, y, text in nodes:
        box = ax.text(
            x, y, text, transform=ax.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="0.96", edgecolor="0.25"),
            fontsize=9,
        )
        boxes[text] = (x, y)
    arrows = [
        (nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3]),
        (nodes[3], nodes[4]), (nodes[4], nodes[6]), (nodes[6], nodes[5]),
        (nodes[6], nodes[7]), (nodes[5], nodes[3]),
    ]
    for source, target in arrows:
        ax.annotate(
            "", xy=(target[0], target[1]), xytext=(source[0], source[1]),
            xycoords=ax.transAxes, textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", color="0.3", lw=1.4),
        )
    _save_figure(fig, output_dir / "fig01_attack_workflow")


def _generate_protocol(output_dir):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    boxes = [
        (0.08, 0.70, "Clean graph", "0.90"),
        (0.28, 0.70, "Online PPO training", "0.90"),
        (0.49, 0.70, "Saved checkpoint", "0.90"),
        (0.70, 0.70, "Frozen PPO attack", "0.90"),
        (0.28, 0.22, "Random attack\nsame seed and budget", "0.96"),
        (0.70, 0.22, "Fresh clean graph", "0.96"),
        (0.91, 0.46, "Paired comparison\nmean ± std", "0.90"),
    ]
    for x, y, label, color in boxes:
        ax.text(x, y, label, transform=ax.transAxes, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="0.25"),
                fontsize=9)
    for a, b in [((0.08, .70), (.28, .70)), ((.28, .70), (.49, .70)),
                 ((.49, .70), (.70, .70)), ((.70, .22), (.70, .70)),
                 ((.08, .70), (.28, .22)), ((.28, .22), (.91, .46)),
                 ((.70, .70), (.91, .46))]:
        ax.annotate("", xy=b, xytext=a, xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", color="0.3", lw=1.4))
    _save_figure(fig, output_dir / "fig02_train_evaluation_protocol")


def _generate_action_distribution(runs, output_dir):
    distributions = defaultdict(Counter)
    for run in runs:
        path = run["_run_dir"] / "modifications.json"
        if not path.exists():
            continue
        for record in json.loads(path.read_text(encoding="utf-8")):
            distributions[run["_strategy"]][record.get("action_name", "unknown")] += 1
    if not distributions:
        return []
    actions = sorted({action for counts in distributions.values() for action in counts})
    strategies = sorted(distributions)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 1.3), 5))
    bottoms = np.zeros(len(strategies))
    rows = []
    for action in actions:
        values = []
        for strategy in strategies:
            total = sum(distributions[strategy].values())
            proportion = distributions[strategy][action] / total if total else 0.0
            values.append(proportion * 100)
            rows.append({
                "strategy": strategy, "action": action,
                "count": distributions[strategy][action],
                "proportion": proportion,
            })
        positions = np.arange(len(strategies))
        ax.bar(positions, values, bottom=bottoms, label=action)
        bottoms += np.asarray(values)
    ax.set_ylabel("Action proportion (%)")
    ax.set_xticks(np.arange(len(strategies)))
    ax.set_xticklabels(strategies, rotation=22, ha="right")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    _save_figure(fig, output_dir / "fig08_action_distribution")
    return rows


def generate_single_run_outputs(run_dir, project_root=None):
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    result = payload.get("result", {})
    output_dir = run_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    plt = _mpl()
    generated = []

    initial = result.get("initial_metrics", {})
    final = result.get("final_metrics", {})
    if initial and final:
        matrices = [
            np.asarray([[initial.get("tn", 0), initial.get("fp", 0)],
                        [initial.get("fn", 0), initial.get("tp", 0)]], dtype=float),
            np.asarray([[final.get("tn", 0), final.get("fp", 0)],
                        [final.get("fn", 0), final.get("tp", 0)]], dtype=float),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
        vmax = max(np.log1p(matrix).max() for matrix in matrices)
        for ax, matrix, title in zip(axes, matrices, ["Before attack", "After attack"]):
            image = ax.imshow(np.log1p(matrix), vmin=0, vmax=vmax, cmap="Blues")
            for row in range(2):
                for col in range(2):
                    ax.text(col, row, f"{int(matrix[row, col]):,}", ha="center", va="center")
            ax.set_xticks([0, 1], ["Pred benign", "Pred malicious"])
            ax.set_yticks([0, 1], ["True benign", "True malicious"])
            ax.set_title(title)
        fig.colorbar(image, ax=axes, label="log(1 + count)", shrink=0.8)
        _save_figure(fig, output_dir / "fig_confusion_matrices")
        generated.append("fig_confusion_matrices")

    trajectory_path = run_dir / "trajectory.json"
    if trajectory_path.exists():
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        feedback = [q for q in trajectory.get("queries", [])
                    if q.get("kind") == "attack_feedback"]
        if feedback:
            x = np.arange(1, len(feedback) + 1)
            rewards = [q.get("total_reward", 0.0) for q in feedback]
            drops = [q.get("score_before", 0.0) - q.get("score_after", 0.0)
                     for q in feedback]
            fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
            axes[0].plot(x, rewards, lw=1.2)
            axes[0].axhline(0, color="0.35", lw=1)
            axes[0].set_ylabel("Batch reward")
            axes[1].plot(x, drops, lw=1.2)
            axes[1].axhline(0, color="0.35", lw=1)
            axes[1].set_ylabel("Target score drop")
            axes[1].set_xlabel("Detector feedback query")
            for ax in axes:
                ax.grid(alpha=0.25)
            _save_figure(fig, output_dir / "fig_reward_trajectory")
            generated.append("fig_reward_trajectory")

    scores_path = run_dir / "scores.npz"
    dataset = payload.get("arguments", {}).get("dataset")
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    metadata_path = root / "MAGIC_main" / "data" / str(dataset) / "metadata.json"
    if scores_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        test_index = int(payload.get("arguments", {}).get("test_index", 0))
        global_malicious = [
            int(node) for node in metadata["malicious"][0]
        ]
        if str(dataset).lower() == "trace":
            cache_path = (
                root / "MAGIC_main" / "eval_result"
                / "native_evaluation_trace_dl.pkl"
            )
            if not cache_path.exists():
                raise FileNotFoundError(
                    "TRACE per-run plots require native evaluation cache: "
                    f"{cache_path}"
                )
            with cache_path.open("rb") as handle:
                native_cache = pickle.load(handle)
            malicious = set(localize_global_nodes(
                global_malicious,
                native_cache["graph_sizes"],
                test_index,
            ))
        else:
            malicious = set(global_malicious)
        with np.load(scores_path) as scores:
            initial_ids = scores["initial_node_ids"]
            initial_scores = scores["initial_scores"]
            final_map = dict(zip(scores["final_node_ids"].tolist(), scores["final_scores"].tolist()))
        ids = initial_ids
        before = initial_scores
        after = np.asarray([final_map.get(int(node), np.nan) for node in ids])
        finite = np.isfinite(before) & np.isfinite(after)
        ids, before, after = ids[finite], before[finite], after[finite]
        malicious_mask = np.asarray([int(node) in malicious for node in ids])
        benign_indices = np.flatnonzero(~malicious_mask)
        if len(benign_indices) > 100000:
            benign_indices = np.random.default_rng(0).choice(
                benign_indices, 100000, replace=False
            )
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        datasets = [
            (before[malicious_mask], "Malicious before"),
            (after[malicious_mask], "Malicious after"),
            (before[benign_indices], "Benign before"),
            (after[benign_indices], "Benign after"),
        ]
        combined = np.concatenate([values for values, _ in datasets if len(values)])
        if len(combined):
            low, high = np.quantile(combined, [0.001, 0.999])
            bins = np.linspace(low, high if high > low else low + 1, 80)
            for values, label in datasets:
                visible = values[(values >= low) & (values <= high)]
                if len(visible):
                    ax.hist(
                        visible, bins=bins, density=True, histtype="step",
                        lw=1.4, label=label,
                    )
            threshold = result.get("threshold")
            if _finite(threshold):
                ax.axvline(threshold, color="0.25", ls="--", lw=1.2, label="Fixed threshold")
            ax.set_xlabel("MAGIC anomaly score")
            ax.set_ylabel("Density")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=0.2)
            _save_figure(fig, output_dir / "fig_score_distributions")
            generated.append("fig_score_distributions")
        else:
            plt.close(fig)
    return generated


def generate_aggregate_outputs(
    results_dir, output_dir=None, main_top_k=10, main_rounds=5
):
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir / "paper_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(results_dir)
    if not runs:
        raise FileNotFoundError(f"No summary.json files found under {results_dir}")
    def policy_value(run, name, default="unknown"):
        return _result(run, name, _argument(run, name, default))

    def is_budget(run, top_k, rounds):
        return (
            int(_argument(run, "top_k", -1)) == int(top_k)
            and int(_argument(run, "max_rounds", -1)) == int(rounds)
        )

    def variant(run):
        value = policy_value(run, "experiment_variant", "auto")
        if value not in {None, "auto", "unspecified"}:
            return value
        action = policy_value(run, "action_policy")
        if action == "random":
            return "no_ppo"
        if action == "no_attack":
            return "no_attack"
        if action == "greedy":
            return "greedy"
        if action == "linucb":
            return "linucb_baseline"
        if action == "double_dqn":
            return "double_dqn_baseline"
        return "full_model"

    def figure_label(run, field):
        return (
            f"{_argument(run, 'dataset', 'unknown')}/"
            f"{_argument(run, 'detector', 'unknown')} | {field(run)}"
        )

    def write_table(filename, rows):
        if rows:
            _write_csv(output_dir / filename, rows, list(rows[0]))

    definitions = [dict(zip(
        ["metric", "formula", "ideal_direction", "ideal_value", "interpretation"], row
    )) for row in METRIC_DEFINITIONS]
    _write_csv(output_dir / "table00_metric_definitions.csv", definitions, list(definitions[0]))

    main_metrics = PRIMARY_METRICS

    # Table 1: intentionally no Strategy column.  It is the full frozen model
    # at the declared main budget, grouped only by dataset and detector.
    table1_runs = [
        run for run in runs
        if policy_value(run, "action_policy") == "ppo_frozen"
        and policy_value(run, "target_policy") == "threshold_margin"
        and variant(run) == "full_model"
        and is_budget(run, main_top_k, main_rounds)
    ]
    table1_groups = _group_runs(
        table1_runs,
        lambda run: (
            _argument(run, "dataset", "unknown"),
            _argument(run, "detector", "unknown"),
        ),
    )
    table1_rows = _aggregate_rows(
        table1_groups,
        main_metrics + [
            ("auc_before", "initial_metrics.auc"),
            ("auc_after", "final_metrics.auc"),
            ("f1_before", "initial_metrics.f1"),
            ("f1_after", "final_metrics.f1"),
            ("precision_before", "initial_metrics.precision"),
            ("precision_after", "final_metrics.precision"),
            ("recall_before", "initial_metrics.recall"),
            ("recall_after", "final_metrics.recall"),
        ],
        ("dataset", "detector"),
    )
    write_table("table01_cross_detector_dataset.csv", table1_rows)

    # Table 2: isolate target selection with one action policy (random).
    table2_runs = [
        run for run in runs
        if policy_value(run, "action_policy") == "random"
        and is_budget(run, main_top_k, main_rounds)
    ]
    table2_groups = _group_runs(
        table2_runs,
        lambda run: (
            _argument(run, "dataset", "unknown"),
            _argument(run, "detector", "unknown"),
            policy_value(run, "target_policy"),
        ),
    )
    table2_rows = _aggregate_rows(
        table2_groups,
        main_metrics + [("actual_unique_targets", "attacked_unique_nodes")],
        ("dataset", "detector", "target_policy"),
    )
    _attach_cost_columns(
        table2_rows, table2_groups,
        ("dataset", "detector", "target_policy"),
    )
    write_table("table02_target_policy.csv", table2_rows)

    # Table 3: action policies under margin targeting.
    table3_runs = [
        run for run in runs
        if policy_value(run, "target_policy") == "threshold_margin"
        and policy_value(run, "action_policy") in {
            "no_attack", "random", "greedy", "linucb", "double_dqn",
            "benign_mimicry", "ppo_online", "ppo_frozen"
        }
        and variant(run) in {
            "no_attack", "no_ppo", "greedy", "linucb_baseline",
            "double_dqn_baseline", "benign_mimicry_baseline", "full_model",
        }
        and is_budget(run, main_top_k, main_rounds)
    ]
    table3_groups = _group_runs(
        table3_runs,
        lambda run: (
            _argument(run, "dataset", "unknown"),
            _argument(run, "detector", "unknown"),
            policy_value(run, "action_policy"),
        ),
    )
    table3_rows = _aggregate_rows(
        table3_groups, main_metrics,
        ("dataset", "detector", "action_policy"),
    )
    write_table("table03_action_policy.csv", table3_rows)

    # Keep a run-level table so partially completed multi-seed experiments are
    # never mistaken for a balanced mean +/- std comparison.  In particular,
    # this lets us compare every policy on seed 42 before committing to the
    # remaining seeds.
    table3_by_seed_rows = []
    for run in sorted(
        table3_runs,
        key=lambda item: (
            str(_argument(item, "dataset", "unknown")),
            str(_argument(item, "detector", "unknown")),
            int(_argument(item, "seed", -1)),
            str(policy_value(item, "action_policy")),
        ),
    ):
        row = {
            "dataset": _argument(run, "dataset", "unknown"),
            "detector": _argument(run, "detector", "unknown"),
            "seed": _argument(run, "seed", _result(run, "seed")),
            "action_policy": policy_value(run, "action_policy"),
            "run_name": _argument(run, "run_name", run["_run_dir"].name),
        }
        for column, key in main_metrics:
            row[column] = _result(run, key)
        row["detector_queries"] = _result(run, "total_detector_queries")
        row["runtime_minutes"] = _result(run, "attack_elapsed_minutes")
        row["actual_unique_targets"] = _result(run, "attacked_unique_nodes")
        table3_by_seed_rows.append(row)
    write_table("table03_action_policy_by_seed.csv", table3_by_seed_rows)

    for seed in sorted({
        int(row["seed"])
        for row in table3_by_seed_rows
        if row.get("seed") is not None
    }):
        write_table(
            f"table03_action_policy_seed{seed}.csv",
            [row for row in table3_by_seed_rows if int(row["seed"]) == seed],
        )

    # Tables 4-1/4-2 follow the user's two fixed-round Top-K sweeps.
    table41_runs = [
        run for run in runs
        if policy_value(run, "target_policy") == "threshold_margin"
        and policy_value(run, "action_policy") in {
            "random", "greedy", "linucb", "double_dqn",
            "ppo_online", "ppo_frozen",
        }
        and int(_argument(run, "max_rounds", -1)) == 5
        and int(_argument(run, "top_k", -1)) in {1, 5, 10, 20}
        and variant(run) in {
            "no_ppo", "greedy", "linucb_baseline",
            "double_dqn_baseline", "full_model",
        }
    ]
    table41_groups = _group_runs(
        table41_runs,
        lambda run: (
            _argument(run, "dataset", "unknown"),
            _argument(run, "detector", "unknown"),
            policy_value(run, "action_policy"),
            int(_argument(run, "top_k", -1)),
        ),
    )
    table41_rows = _aggregate_rows(
        table41_groups,
        main_metrics + [("actual_unique_targets", "attacked_unique_nodes")],
        ("dataset", "detector", "action_policy", "top_k"),
    )
    _attach_cost_columns(
        table41_rows, table41_groups,
        ("dataset", "detector", "action_policy", "top_k"),
    )
    for row in table41_rows:
        row["rounds"] = 5
        row["planned_target_budget"] = 5 * int(row["top_k"])
    write_table("table04_1_budget_r5.csv", table41_rows)

    table42_runs = [
        run for run in runs
        if policy_value(run, "target_policy") == "threshold_margin"
        and policy_value(run, "action_policy") in {
            "random", "greedy", "linucb", "double_dqn",
            "ppo_online", "ppo_frozen",
        }
        and int(_argument(run, "max_rounds", -1)) == 10
        and int(_argument(run, "top_k", -1)) in {1, 3, 5, 10}
        and variant(run) in {
            "no_ppo", "greedy", "linucb_baseline",
            "double_dqn_baseline", "full_model",
        }
    ]
    table42_groups = _group_runs(
        table42_runs,
        lambda run: (
            _argument(run, "dataset", "unknown"),
            _argument(run, "detector", "unknown"),
            policy_value(run, "action_policy"),
            int(_argument(run, "top_k", -1)),
        ),
    )
    table42_rows = _aggregate_rows(
        table42_groups,
        main_metrics + [("actual_unique_targets", "attacked_unique_nodes")],
        ("dataset", "detector", "action_policy", "top_k"),
    )
    _attach_cost_columns(
        table42_rows, table42_groups,
        ("dataset", "detector", "action_policy", "top_k"),
    )
    for row in table42_rows:
        row["rounds"] = 10
        row["planned_target_budget"] = 10 * int(row["top_k"])
    write_table("table04_2_budget_r10.csv", table42_rows)

    # Table 5: LinUCB component ablation.  The Full row is reused from Table 3;
    # the other rows remove exactly one component under the same R=5, K=10
    # Margin protocol.
    table5_variants = {
        "linucb_baseline",
        "linucb_no_context",
        "linucb_no_exploration",
        "linucb_no_candidate_selector",
        "linucb_no_reward_feedback",
        "linucb_no_macro_action",
    }
    table5_runs = []
    for run in runs:
        current_variant = variant(run)
        action = policy_value(run, "action_policy")
        if (
            current_variant in table5_variants
            and action == "linucb"
            and policy_value(run, "target_policy") == "threshold_margin"
            and is_budget(run, main_top_k, main_rounds)
        ):
            table5_runs.append(run)
    table5_groups = _group_runs(
        table5_runs,
        lambda run: (
            _argument(run, "dataset", "unknown"),
            _argument(run, "detector", "unknown"),
            variant(run),
        ),
    )
    table5_rows = _aggregate_rows(
        table5_groups, LINUCB_ABLATION_METRICS,
        ("dataset", "detector", "variant"),
    )
    write_table("table05_ablation.csv", table5_rows)

    # Run-level values make it possible to audit every seed behind the
    # aggregate mean +/- standard deviation without bloating the paper table.
    table5_by_seed_rows = []
    for run in sorted(
        table5_runs,
        key=lambda item: (
            str(_argument(item, "dataset", "unknown")),
            str(_argument(item, "detector", "unknown")),
            str(variant(item)),
            int(_argument(item, "seed", -1)),
        ),
    ):
        result = run.get("result", {})
        table5_by_seed_rows.append({
            "dataset": _argument(run, "dataset", "unknown"),
            "detector": _argument(run, "detector", "unknown"),
            "variant": variant(run),
            "seed": int(_argument(run, "seed", -1)),
            "direct_tp_asr": result.get("direct_tp_asr"),
            "average_attack_budget": result.get("average_attack_budget"),
            "average_query_time_seconds": result.get(
                "average_query_time_seconds"
            ),
            "f1_drop": result.get("f1_drop"),
            "total_detector_queries": result.get("total_detector_queries"),
        })
    if table5_by_seed_rows:
        _write_csv(
            output_dir / "table05_ablation_by_seed.csv",
            table5_by_seed_rows,
            list(table5_by_seed_rows[0]),
        )

    action_rows = _generate_action_distribution(runs, output_dir)
    if action_rows:
        _write_csv(output_dir / "appendix_action_distribution.csv", action_rows, list(action_rows[0]))

    _generate_workflow(output_dir)
    _generate_protocol(output_dir)
    table1_figure_groups = _group_runs(
        table1_runs,
        lambda run: f"{_argument(run, 'dataset')}/{_argument(run, 'detector')}",
    )
    _primary_metric_panels(
        table1_figure_groups, output_dir / "fig03_table01_attack_metrics"
    )
    _detector_metric_panels(
        table1_figure_groups, output_dir / "fig04_table01_detector_metrics"
    )
    table2_figure_groups = _group_runs(
        table2_runs,
        lambda run: figure_label(run, lambda item: policy_value(item, "target_policy")),
    )
    _primary_metric_panels(
        table2_figure_groups, output_dir / "fig05_table02_target_policy"
    )
    _table2_cost_by_seed(
        table2_runs, output_dir / "fig05b_table02_cost_by_seed"
    )
    table3_figure_groups = _group_runs(
        table3_runs,
        lambda run: figure_label(run, lambda item: policy_value(item, "action_policy")),
    )
    _primary_metric_panels(
        table3_figure_groups, output_dir / "fig06_table03_action_policy"
    )
    for seed in sorted({
        int(_argument(run, "seed"))
        for run in table3_runs
        if _argument(run, "seed") is not None
    }):
        seed_runs = [
            run for run in table3_runs
            if int(_argument(run, "seed")) == seed
        ]
        seed_groups = _group_runs(
            seed_runs,
            lambda run: figure_label(
                run, lambda item: policy_value(item, "action_policy")
            ),
        )
        _primary_metric_panels(
            seed_groups,
            output_dir / f"fig06_table03_action_policy_seed{seed}",
        )
    _budget_metric_panels(
        table41_rows, output_dir / "fig07_table04_1_budget_r5"
    )
    _budget_metric_panels(
        table42_rows, output_dir / "fig08_table04_2_budget_r10"
    )
    table5_figure_groups = _group_runs(
        table5_runs,
        lambda run: figure_label(run, variant),
    )
    _primary_metric_panels(
        table5_figure_groups, output_dir / "fig09_table05_ablation"
    )
    _generate_ppo_learning_curve(runs, output_dir)
    _generate_reward_component_figure(table5_runs, output_dir)

    for run in runs:
        generate_single_run_outputs(run["_run_dir"])
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables and figures")
    parser.add_argument("--results-dir", type=Path, default=Path("attack_results"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--main-top-k", type=int, default=10)
    parser.add_argument("--main-rounds", type=int, default=5)
    args = parser.parse_args()
    output = generate_aggregate_outputs(
        args.results_dir, args.output_dir,
        main_top_k=args.main_top_k, main_rounds=args.main_rounds,
    )
    print(f"Paper tables and figures written to {output.resolve()}")


if __name__ == "__main__":
    main()
