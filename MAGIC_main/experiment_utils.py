"""Reproducible experiment setup and structured artifact persistence."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import dgl
import numpy as np
import torch


def configure_reproducibility(seed: int, deterministic: bool = False) -> Dict[str, Any]:
    """Seed every RNG used by the attack and optionally request deterministic kernels."""
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    try:
        dgl.random.seed(seed)
    except AttributeError:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "deterministic_requested": bool(deterministic),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "dgl_version": getattr(dgl, "__version__", "unknown"),
    }


def create_run_directory(
    output_dir: Path,
    *,
    detector: str,
    dataset: str,
    action_policy: str,
    target_policy: str,
    seed: int,
    run_name: Optional[str] = None,
) -> Path:
    if run_name:
        safe_name = run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_name = (
            f"{detector}_{dataset}_{action_policy}_{target_policy}_"
            f"seed{seed}_{timestamp}"
        )
    run_dir = Path(output_dir).resolve() / safe_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def attach_run_log(run_dir: Path) -> Path:
    """Mirror all root logger output to a UTF-8 file for the current run."""
    path = Path(run_dir) / "run.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(handler)
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Namespace):
        return _json_safe(vars(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _score_arrays(scores: Dict[int, float]):
    node_ids = np.fromiter(sorted(scores), dtype=np.int64)
    values = np.asarray([scores[int(node_id)] for node_id in node_ids], dtype=np.float64)
    return node_ids, values


def save_experiment_artifacts(
    run_dir: Path,
    *,
    args: Namespace,
    reproducibility: Dict[str, Any],
    result: Dict[str, Any],
    agent: Any,
    save_final_graph: bool = False,
) -> Dict[str, str]:
    """Save enough state to audit metrics and replay graph modifications."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    modifications_path = run_dir / "modifications.json"
    trajectory_path = run_dir / "trajectory.json"
    scores_path = run_dir / "scores.npz"

    summary = {
        "arguments": args,
        "reproducibility": reproducibility,
        "result": result,
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    modifications_path.write_text(
        json.dumps(_json_safe(agent.modification_records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trajectory_path.write_text(
        json.dumps(_json_safe({
            "rounds": agent.round_records,
            "queries": agent.query_records,
        }), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    initial_ids, initial_values = _score_arrays(agent.initial_all_scores)
    final_ids, final_values = _score_arrays(agent.final_all_scores)
    np.savez_compressed(
        scores_path,
        initial_node_ids=initial_ids,
        initial_scores=initial_values,
        final_node_ids=final_ids,
        final_scores=final_values,
    )

    paths = {
        "run_dir": str(run_dir),
        "summary": str(summary_path),
        "modifications": str(modifications_path),
        "trajectory": str(trajectory_path),
        "scores": str(scores_path),
    }
    if save_final_graph:
        graph_path = run_dir / "final_graph.bin"
        dgl.save_graphs(str(graph_path), [agent.final_graph])
        paths["final_graph"] = str(graph_path)
    return paths
