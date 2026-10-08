"""Reproducible native FLASH pipeline for DARPA E3 CADETS.

This is the executable Python replacement for ``Cadets.ipynb``.  It deliberately
separates two stages:

``prepare``
    Stream the original CDM JSON, reproduce FLASH's actor->object multigraph and
    positional Word2Vec node features, and map them to MAGIC's stable DGL node
    indices.  The result is a compact ``.npz`` detector-input sidecar.

``evaluate``
    Load the official Word2Vec/GraphSAGE weights, apply the native persistent
    misclassification ensemble rule, and report node-level AUC/F1/Precision/
    Recall against metadata.  Metadata is used only after inference for offline
    evaluation; it never enters the score or alert rule.

The notebook is retained as provenance, but this module has no downloads, shell
commands, notebook magics, or import-time execution.
"""

from __future__ import annotations

import argparse
from array import array
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Detector.Flash import FlashDetector
from Detector.Flash_State import FlashSemanticState
from MAGIC_main.utils.loaddata import load_entity_level_dataset


LOGGER = logging.getLogger("flash.cadets.native")
FLASH_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = FLASH_ROOT / "ta1-cadets-e3-official-2.json"
DEFAULT_UUID_MAP = ROOT / "MAGIC_main" / "data" / "cadets" / "uuid_to_index_test0.json"
DEFAULT_STATE = FLASH_ROOT / "data_files" / "cadets_flash_native_test0.npz"
DEFAULT_W2V = FLASH_ROOT / "trained_weights" / "cadets" / "word2vec_cadets_E3.model"
DEFAULT_FLASH_GT = FLASH_ROOT / "data_files" / "cadets.json"

try:
    import orjson as _orjson

    _JSON_LOADS = _orjson.loads
except ImportError:
    _JSON_LOADS = json.loads


def _nested_uuid(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    for key, item in value.items():
        if str(key).endswith(".UUID"):
            return str(item) if item is not None else None
    return None


def _optional_string(value: Any) -> str:
    if isinstance(value, dict):
        if "string" in value:
            value = value["string"]
        elif len(value) == 1:
            value = next(iter(value.values()))
    return "" if value is None else str(value)


def _event_from_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    datum = record.get("datum")
    if not isinstance(datum, dict):
        return None
    for key, value in datum.items():
        if str(key).endswith(".Event") and isinstance(value, dict):
            return value
    return None


def _event_objects(event: Dict[str, Any]) -> Iterable[tuple[str, str]]:
    first = _nested_uuid(event.get("predicateObject"))
    if first and first != "null":
        yield first, _optional_string(event.get("predicateObjectPath"))
    second = _nested_uuid(event.get("predicateObject2"))
    if second and second != "null":
        yield second, _optional_string(event.get("predicateObject2Path"))


def _event_exec(event: Dict[str, Any]) -> str:
    properties = event.get("properties")
    if not isinstance(properties, dict):
        return ""
    mapping = properties.get("map", properties)
    if not isinstance(mapping, dict):
        return ""
    return _optional_string(mapping.get("exec"))


def positional_encoding(max_len: int, dim: int = 30) -> np.ndarray:
    position = np.arange(max_len, dtype=np.float32)[:, None]
    scale = np.exp(
        np.arange(0, dim, 2, dtype=np.float32)
        * (-np.log(10000.0) / dim)
    )
    result = np.zeros((max_len, dim), dtype=np.float32)
    result[:, 0::2] = np.sin(position * scale)
    result[:, 1::2] = np.cos(position * scale)
    return result


def prepare_native_state(
    *,
    raw_json: Path = DEFAULT_RAW,
    uuid_map_path: Path = DEFAULT_UUID_MAP,
    word2vec_path: Path = DEFAULT_W2V,
    output_path: Path = DEFAULT_STATE,
    force: bool = False,
    progress_every: int = 1_000_000,
) -> Path:
    """Build the clean FLASH detector input without loading the 4 GB file."""
    if output_path.exists() and not force:
        raise FileExistsError(
            f"state already exists: {output_path}; pass --force to replace it"
        )
    for required in (raw_json, uuid_map_path, word2vec_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    import scipy.linalg
    if not hasattr(scipy.linalg, "triu"):
        scipy.linalg.triu = np.triu
    from gensim.models import Word2Vec

    LOGGER.info("Loading UUID map: %s", uuid_map_path)
    node_to_index = json.loads(uuid_map_path.read_text(encoding="utf-8"))
    node_to_index = {str(key): int(value) for key, value in node_to_index.items()}
    node_count = len(node_to_index)
    if sorted(node_to_index.values()) != list(range(node_count)):
        raise ValueError("UUID map values must be contiguous DGL node indices")

    LOGGER.info("Loading FLASH Word2Vec model: %s", word2vec_path)
    word2vec = Word2Vec.load(str(word2vec_path))
    if int(word2vec.vector_size) != 30:
        raise ValueError(f"expected 30-D FLASH Word2Vec, got {word2vec.vector_size}")
    vocabulary = word2vec.wv.key_to_index

    word_sums = np.zeros((node_count, 30), dtype=np.float32)
    position_sums = np.zeros((node_count, 30), dtype=np.float32)
    token_counts = np.zeros(node_count, dtype=np.int64)
    document_counts = np.zeros(node_count, dtype=np.int64)
    incident_counts = np.zeros(node_count, dtype=np.int64)
    action_masks = np.zeros(node_count, dtype=np.uint64)
    action_bits: Dict[str, int] = {}
    edge_sources = array("I")
    edge_destinations = array("I")
    pe = positional_encoding(100_000, 30)

    accepted_edges = 0
    malformed = 0
    start = time.perf_counter()
    LOGGER.info("Streaming native CADETS events: %s", raw_json)
    with raw_json.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if b"Event" not in line:
                continue
            try:
                event = _event_from_record(_JSON_LOADS(line))
            except Exception:
                malformed += 1
                continue
            if event is None:
                continue
            actor_uuid = _nested_uuid(event.get("subject"))
            if actor_uuid not in node_to_index:
                continue
            actor = node_to_index[actor_uuid]
            action = _optional_string(event.get("type"))
            executable = _event_exec(event)
            bit = action_bits.setdefault(action, len(action_bits))
            if bit >= 64:
                raise ValueError("CADETS contains more than 64 event types")

            for object_uuid, path in _event_objects(event):
                if object_uuid not in node_to_index:
                    continue
                obj = node_to_index[object_uuid]
                edge_sources.append(actor)
                edge_destinations.append(obj)
                accepted_edges += 1

                mask = np.uint64(1) << np.uint64(bit)
                for node in (actor, obj):
                    incident_counts[node] += 1
                    action_masks[node] |= mask
                    document_tokens = (executable, action) + ((path,) if path else ())
                    document_counts[node] += len(document_tokens)
                    for token in document_tokens:
                        if not token or token not in vocabulary:
                            continue
                        position = int(token_counts[node])
                        word_sums[node] += np.asarray(
                            word2vec.wv[token], dtype=np.float32
                        )
                        if position < len(pe):
                            position_sums[node] += pe[position]
                        token_counts[node] = position + 1

            if progress_every and line_number % progress_every == 0:
                LOGGER.info(
                    "raw_lines=%d, native_edges=%d, elapsed=%.1f min",
                    line_number,
                    accepted_edges,
                    (time.perf_counter() - start) / 60.0,
                )

    embedding_sums = word_sums.copy()
    positional_nodes = document_counts < 100_000
    embedding_sums[positional_nodes] += position_sums[positional_nodes]
    embeddings = np.zeros_like(embedding_sums)
    nonzero = token_counts > 0
    embeddings[nonzero] = embedding_sums[nonzero] / token_counts[nonzero, None]
    distinct_counts = np.fromiter(
        (bin(int(value)).count("1") for value in action_masks),
        dtype=np.int64,
        count=node_count,
    )
    native_edge_index = np.vstack(
        (
            np.frombuffer(edge_sources, dtype=np.uint32).astype(np.int64),
            np.frombuffer(edge_destinations, dtype=np.uint32).astype(np.int64),
        )
    )
    if accepted_edges == 0:
        raise ValueError("no native FLASH edges matched the MAGIC UUID map")

    state = FlashSemanticState(
        original_node_count=node_count,
        base_embeddings=embeddings,
        token_counts=token_counts,
        document_counts=document_counts,
        native_edge_index=native_edge_index,
        incident_counts=incident_counts,
        distinct_action_counts=distinct_counts,
        dataset="cadets",
        source=str(output_path.resolve()),
    )
    state.to_npz(output_path)
    LOGGER.info(
        "Saved native FLASH state: %s (nodes=%d, edges=%d, zero_feature_nodes=%d, "
        "malformed_lines=%d, elapsed=%.1f min)",
        output_path,
        node_count,
        accepted_edges,
        int(np.sum(~nonzero)),
        malformed,
        (time.perf_counter() - start) / 60.0,
    )
    return output_path


def evaluate_native(
    *,
    state_path: Path = DEFAULT_STATE,
    device: str = "cuda:0",
    test_index: int = 0,
    top_k: int = 10,
) -> Dict[str, Any]:
    graph = load_entity_level_dataset("cadets", "test", test_index)
    state = FlashSemanticState.load(state_path)
    detector = FlashDetector(
        dataset_name="cadets",
        device=device,
        top_k=top_k,
        threshold_source="native",
    )
    detector.set_evaluation_context(test_index)
    detector.attach_semantic_state(state)
    baseline = detector.initialize_tp_baseline(graph)
    official_metrics = flash_official_postprocessed_metrics(
        alerts=detector.initial_alert_nodes,
        state=state,
        uuid_map_path=DEFAULT_UUID_MAP,
        ground_truth_path=DEFAULT_FLASH_GT,
    )
    result = {
        "status": "PASS",
        "dataset": "cadets",
        "test_index": test_index,
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges_magic_view": int(graph.number_of_edges()),
        "graph_edges_flash_native_view": int(state.native_edge_index.shape[1]),
        "ensemble_size": len(detector.model_ensemble),
        "score_definition": "min_model(max_other_logit - true_type_logit)",
        "alert_rule": "all native models misclassify the observable node type",
        "threshold": float(detector.best_threshold),
        "threshold_uses_malicious_ground_truth": False,
        "initial_alerts": len(detector.initial_alert_nodes),
        "initial_true_tp": detector.true_tp_count,
        "initial_false_positive": detector.false_positive_count,
        "cache_reused": detector.last_baseline_cache_hit,
        "official_flash_postprocessed": official_metrics,
        **detector.initial_metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _two_hop_mask(
    seeds: set[int], edge_index: np.ndarray, node_count: int
) -> np.ndarray:
    reached = np.zeros(node_count, dtype=bool)
    frontier = np.zeros(node_count, dtype=bool)
    if seeds:
        frontier[np.fromiter(seeds, dtype=np.int64)] = True
    src, dst = edge_index[0], edge_index[1]
    for _ in range(2):
        touching = frontier[src] | frontier[dst]
        next_frontier = np.zeros(node_count, dtype=bool)
        next_frontier[src[touching]] = True
        next_frontier[dst[touching]] = True
        reached |= next_frontier
        frontier = next_frontier
    return reached


def flash_official_postprocessed_metrics(
    *,
    alerts: set[int],
    state: FlashSemanticState,
    uuid_map_path: Path,
    ground_truth_path: Path,
) -> Dict[str, Any]:
    """Reproduce the two-hop tolerance in the official CADETS notebook."""
    uuid_map = json.loads(uuid_map_path.read_text(encoding="utf-8"))
    ground_truth_uuids = set(json.loads(ground_truth_path.read_text(encoding="utf-8")))
    ground_truth = {
        int(uuid_map[uuid]) for uuid in ground_truth_uuids if uuid in uuid_map
    }
    alerts = set(int(node) for node in alerts)
    tp = alerts & ground_truth
    fp = alerts - ground_truth
    fn = ground_truth - alerts
    tn = state.original_node_count - len(alerts | ground_truth)
    gp_neighborhood = _two_hop_mask(
        ground_truth, state.native_edge_index, state.original_node_count
    )
    tp_neighborhood = _two_hop_mask(
        tp, state.native_edge_index, state.original_node_count
    )
    filtered_fp = {node for node in fp if not gp_neighborhood[node]}
    recovered_fn = {node for node in fn if tp_neighborhood[node]}
    tolerant_tp = tp | recovered_fn
    remaining_fn = fn - recovered_fn
    tp_count = len(tolerant_tp)
    fp_count = len(filtered_fp)
    fn_count = len(remaining_fn)
    precision = tp_count / (tp_count + fp_count) if tp_count + fp_count else 0.0
    recall = tp_count / (tp_count + fn_count) if tp_count + fn_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "scope": "official UUID ground truth with two-hop tolerance",
        "auc": None,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tn": int(tn),
        "fn": fn_count,
        "tp": tp_count,
        "fp": fp_count,
        "ground_truth_nodes": len(ground_truth),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and evaluate native FLASH on CADETS"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build the native .npz sidecar")
    prepare.add_argument("--raw-json", type=Path, default=DEFAULT_RAW)
    prepare.add_argument("--uuid-map", type=Path, default=DEFAULT_UUID_MAP)
    prepare.add_argument("--word2vec", type=Path, default=DEFAULT_W2V)
    prepare.add_argument("--output", type=Path, default=DEFAULT_STATE)
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--progress-every", type=int, default=1_000_000)

    evaluate = subparsers.add_parser(
        "evaluate", help="verify initial AUC/F1/Precision/Recall"
    )
    evaluate.add_argument("--state", type=Path, default=DEFAULT_STATE)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--test-index", type=int, default=0)
    evaluate.add_argument("--top-k", type=int, default=10)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_native_state(
            raw_json=args.raw_json,
            uuid_map_path=args.uuid_map,
            word2vec_path=args.word2vec,
            output_path=args.output,
            force=args.force,
            progress_every=args.progress_every,
        )
    elif args.command == "evaluate":
        evaluate_native(
            state_path=args.state,
            device=args.device,
            test_index=args.test_index,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
