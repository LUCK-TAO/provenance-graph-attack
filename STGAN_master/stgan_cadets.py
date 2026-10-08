"""Reproducible native STGAN pipeline for DARPA E3 CADETS.

``prepare`` streams the released CDM input and builds the exact detector-side
state needed by the STGAN evaluator. ``audit`` checks alignment without model
inference. ``evaluate`` runs the released 50-snapshot rule and reports both
raw node metrics and the released evaluator's two-hop-tolerant metrics.
"""

from __future__ import annotations

import argparse
from array import array
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Detector.STGAN import STGANDetector
from Detector.STGAN_State import STGANSemanticState
from Detector.STGAN_WordVectors import STGANWordVectors
from Flash_IDS_main.flash_cadets import (
    _JSON_LOADS,
    _event_exec,
    _event_from_record,
    _event_objects,
    _nested_uuid,
    _optional_string,
    flash_official_postprocessed_metrics,
    positional_encoding,
)
from MAGIC_main.action_schema import CADETS_EDGE_TYPES, CADETS_NODE_TYPES
from MAGIC_main.utils.loaddata import load_entity_level_dataset


LOGGER = logging.getLogger("stgan.cadets.native")
STGAN_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = STGAN_ROOT / "data" / "cadets" / "ta1-cadets-e3-official-2.json"
DEFAULT_UUID_MAP = ROOT / "MAGIC_main" / "data" / "cadets" / "uuid_to_index_test0.json"
DEFAULT_W2V = STGAN_ROOT / "train" / "word2vec_cadets_compat.npz"
DEFAULT_STATE = STGAN_ROOT / "data" / "cadets" / "cadets_stgan_native_test0.npz"
DEFAULT_GT = STGAN_ROOT / "data" / "cadets" / "cadets.json"
DEFAULT_FLASH_STATE = ROOT / "Flash_IDS_main" / "data_files" / "cadets_flash_native_test0.npz"


def _position_vector(position: int, dim: int = 30) -> np.ndarray:
    scale = np.exp(
        np.arange(0, dim, 2, dtype=np.float32) * (-np.log(10000.0) / dim)
    )
    value = np.zeros(dim, dtype=np.float32)
    value[0::2] = np.sin(float(position) * scale)
    value[1::2] = np.cos(float(position) * scale)
    return value


def prepare_native_state(
    *,
    raw_json: Path = DEFAULT_RAW,
    uuid_map_path: Path = DEFAULT_UUID_MAP,
    word2vec_path: Path = DEFAULT_W2V,
    output_path: Path = DEFAULT_STATE,
    flash_state_path: Path = DEFAULT_FLASH_STATE,
    force: bool = False,
    progress_every: int = 1_000_000,
) -> Path:
    if output_path.exists() and not force:
        raise FileExistsError(f"state exists: {output_path}; pass --force to replace")
    for path in (raw_json, uuid_map_path, word2vec_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    node_to_index = {
        str(key): int(value)
        for key, value in json.loads(
            uuid_map_path.read_text(encoding="utf-8")
        ).items()
    }
    node_count = len(node_to_index)
    if sorted(node_to_index.values()) != list(range(node_count)):
        raise ValueError("UUID map indices must be contiguous")
    graph = load_entity_level_dataset("cadets", "test", 0)
    if graph.number_of_nodes() != node_count:
        raise ValueError("UUID map and MAGIC test0 node counts differ")
    model_labels = graph.ndata["type"].view(-1).cpu().numpy().astype(np.int64)
    if model_labels.min() < 0 or model_labels.max() >= 6:
        raise ValueError("CADETS entity labels do not match STGAN's six classes")

    LOGGER.info("Loading STGAN Word2Vec: %s", word2vec_path)
    word2vec = STGANWordVectors(word2vec_path)
    if int(word2vec.vector_size) != 30:
        raise ValueError("released STGAN CADETS Word2Vec must be 30-D")
    vocabulary = word2vec.key_to_index

    word_sums = np.zeros((node_count, 30), dtype=np.float32)
    position_sums = np.zeros((node_count, 30), dtype=np.float32)
    token_counts = np.zeros(node_count, dtype=np.int64)
    edge_sources = array("I")
    edge_destinations = array("I")
    edge_times = array("f")
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
            try:
                timestamp = float(event.get("timestampNanos", 0.0))
            except (TypeError, ValueError):
                timestamp = 0.0

            for object_uuid, path in _event_objects(event):
                if object_uuid not in node_to_index:
                    continue
                obj = node_to_index[object_uuid]
                edge_sources.append(actor)
                edge_destinations.append(obj)
                edge_times.append(timestamp)
                accepted_edges += 1
                tokens = (executable, action) + ((path,) if path else ())
                # The released ``prepare_graph`` invokes add_node_properties
                # once for actor and once for object. Preserve both calls for
                # self-loop events instead of deduplicating the endpoint.
                for node in (actor, obj):
                    for token in tokens:
                        if not token or token not in vocabulary:
                            continue
                        position = int(token_counts[node])
                        word_sums[node] += np.asarray(word2vec[token], dtype=np.float32)
                        position_sums[node] += (
                            pe[position]
                            if position < len(pe)
                            else _position_vector(position)
                        )
                        token_counts[node] = position + 1

            if progress_every and line_number % progress_every == 0:
                LOGGER.info(
                    "raw_lines=%d, native_edges=%d, elapsed=%.1f min",
                    line_number,
                    accepted_edges,
                    (time.perf_counter() - start) / 60.0,
                )

    native_edge_index = np.vstack(
        (
            np.frombuffer(edge_sources, dtype=np.uint32).astype(np.int64),
            np.frombuffer(edge_destinations, dtype=np.uint32).astype(np.int64),
        )
    )
    native_edge_time = np.frombuffer(edge_times, dtype=np.float32).copy()
    if accepted_edges == 0:
        raise ValueError("no native STGAN edges matched the MAGIC UUID map")
    embeddings = np.zeros((node_count, 30), dtype=np.float32)
    nonzero = token_counts > 0
    embeddings[nonzero] = (
        word_sums[nonzero] + position_sums[nonzero]
    ) / token_counts[nonzero, None]

    # The FLASH and STGAN released pipelines use the same CADETS CDM split.
    # Compare against the already-audited FLASH topology when available.
    if flash_state_path.is_file():
        with np.load(flash_state_path, allow_pickle=False) as flash:
            flash_edges = np.asarray(flash["native_edge_index"], dtype=np.int64)
        if flash_edges.shape != native_edge_index.shape:
            raise ValueError(
                "STGAN/FLASH native CADETS edge count differs: "
                f"STGAN={native_edge_index.shape[1]}, FLASH={flash_edges.shape[1]}"
            )
        mismatch = int(np.sum(flash_edges != native_edge_index))
        if mismatch:
            raise ValueError(f"STGAN/FLASH native topology has {mismatch} endpoint mismatches")

    state = STGANSemanticState(
        original_node_count=node_count,
        base_embeddings=embeddings,
        token_counts=token_counts,
        model_labels=model_labels,
        native_edge_index=native_edge_index,
        native_edge_time=native_edge_time,
        magic_edge_type_names=dict(CADETS_EDGE_TYPES),
        node_type_names=dict(CADETS_NODE_TYPES),
        dataset="cadets",
        test_index=0,
        global_node_offset=0,
        source=str(output_path.resolve()),
    )
    state.to_npz(output_path)
    LOGGER.info(
        "Saved STGAN native state: %s (nodes=%d, edges=%d, zero_feature_nodes=%d, "
        "malformed=%d, elapsed=%.1f min)",
        output_path,
        node_count,
        accepted_edges,
        int(np.sum(~nonzero)),
        malformed,
        (time.perf_counter() - start) / 60.0,
    )
    return output_path


def audit_native_state(state_path: Path = DEFAULT_STATE) -> dict:
    state = STGANSemanticState.load(state_path)
    graph = load_entity_level_dataset("cadets", "test", 0)
    state.validate_for_graph(graph, "cadets", 0)
    graph_labels = graph.ndata["type"].view(-1).cpu().numpy().astype(np.int64)
    report = {
        "status": "PASS",
        "dataset": "cadets",
        "test_index": 0,
        "nodes": state.original_node_count,
        "native_edges": int(state.native_edge_index.shape[1]),
        "magic_view_edges": int(graph.number_of_edges()),
        "entity_label_mismatches": int(np.sum(graph_labels != state.model_labels)),
        "zero_feature_nodes": int(np.sum(state.token_counts == 0)),
        "finite_embeddings": bool(np.all(np.isfinite(state.base_embeddings))),
        "finite_edge_times": bool(np.all(np.isfinite(state.native_edge_time))),
        "gat_checkpoints": len(list((STGAN_ROOT / "trained_weights" / "cadets").glob("gat*.pth"))),
        "tgn_checkpoints": len(list((STGAN_ROOT / "trained_weights" / "cadets").glob("tgn*.pth"))),
        "attention_checkpoints": len(list((STGAN_ROOT / "trained_weights" / "cadets").glob("selfattention*.pth"))),
    }
    if report["entity_label_mismatches"] or not report["finite_embeddings"] or not report["finite_edge_times"]:
        report["status"] = "FAIL"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def evaluate_native(
    *, state_path: Path = DEFAULT_STATE, device: str = "cuda:0", top_k: int = 10
) -> dict:
    graph = load_entity_level_dataset("cadets", "test", 0)
    state = STGANSemanticState.load(state_path)
    detector = STGANDetector(
        dataset_name="cadets",
        device=device,
        top_k=top_k,
        threshold_source="native",
    )
    detector.set_evaluation_context(0)
    detector.attach_semantic_state(state)
    detector.initialize_tp_baseline(graph)
    official = flash_official_postprocessed_metrics(
        alerts=detector.initial_alert_nodes,
        state=state,
        uuid_map_path=DEFAULT_UUID_MAP,
        ground_truth_path=DEFAULT_GT,
    )
    report = {
        "status": "PASS",
        "dataset": "cadets",
        "test_index": 0,
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges_magic_view": int(graph.number_of_edges()),
        "graph_edges_stgan_native_view": int(state.native_edge_index.shape[1]),
        "ensemble_size": len(detector.snapshot_paths),
        "threshold": detector.best_threshold,
        "threshold_uses_malicious_ground_truth": False,
        "initial_alerts": len(detector.initial_alert_nodes),
        "initial_true_tp": detector.true_tp_count,
        "initial_false_positive": detector.false_positive_count,
        "cache_reused": detector.last_baseline_cache_hit,
        "raw_direct_metadata_metrics": detector.raw_initial_metrics,
        "released_code_metrics": detector.released_initial_metrics,
        "released_helper_crosscheck": official,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Native STGAN CADETS reproduction")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--raw-json", type=Path, default=DEFAULT_RAW)
    prepare.add_argument("--uuid-map", type=Path, default=DEFAULT_UUID_MAP)
    prepare.add_argument("--word2vec", type=Path, default=DEFAULT_W2V)
    prepare.add_argument("--output", type=Path, default=DEFAULT_STATE)
    prepare.add_argument("--flash-state", type=Path, default=DEFAULT_FLASH_STATE)
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--progress-every", type=int, default=1_000_000)
    audit = commands.add_parser("audit")
    audit.add_argument("--state", type=Path, default=DEFAULT_STATE)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--state", type=Path, default=DEFAULT_STATE)
    evaluate.add_argument("--device", default="cuda:0")
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
            flash_state_path=args.flash_state,
            force=args.force,
            progress_every=args.progress_every,
        )
    elif args.command == "audit":
        audit_native_state(args.state)
    else:
        evaluate_native(state_path=args.state, device=args.device, top_k=args.top_k)


if __name__ == "__main__":
    main()
