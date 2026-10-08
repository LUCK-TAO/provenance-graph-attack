"""Prepare, audit, and evaluate the released STGAN DARPA E3 TRACE split.

The released TRACE evaluator consumes ``ta1-trace-e3-official-1.json.4`` and
the MAGIC-aligned attack graph is ``test4``.  The compressed member is streamed
directly; no multi-gigabyte raw JSON is left expanded on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Detector.STGAN import STGANDetector
from Detector.STGAN_State import STGANSemanticState
from Flash_IDS_main.flash_e3 import _stream_archive_to_sqlite
from MAGIC_main.action_schema import build_observed_action_schema
from MAGIC_main.utils.loaddata import load_entity_level_dataset
from STGAN_master.stgan_theia import (
    _align_to_audited_flash_state,
    _build_state,
    _load_word2vec,
    _write_compat_vectors,
)


LOGGER = logging.getLogger("stgan.trace.native")
STGAN_ROOT = Path(__file__).resolve().parent
DATA_DIR = STGAN_ROOT / "data" / "trace"
DEFAULT_ARCHIVE = DATA_DIR / "ta1-trace-e3-official-1.json.tar.gz"
DEFAULT_MEMBER = "ta1-trace-e3-official-1.json.4"
DEFAULT_STATE = DATA_DIR / "trace_stgan_native_test4.npz"
DEFAULT_UUID_MAP = ROOT / "MAGIC_main" / "data" / "trace" / "uuid_to_index_test4.json"
DEFAULT_GT = DATA_DIR / "trace.json"
DEFAULT_NATIVE_W2V = STGAN_ROOT / "train" / "word2vec_trace.model"
DEFAULT_FALLBACK_W2V = (
    ROOT / "Flash_IDS_main" / "trained_weights" / "trace"
    / "word2vec_trace_E3.model"
)
DEFAULT_COMPAT_W2V = STGAN_ROOT / "train" / "word2vec_trace_compat.npz"
DEFAULT_FLASH_STATE = (
    ROOT / "Flash_IDS_main" / "data_files" / "trace_flash_native_test4.npz"
)

LABELS = {
    "SUBJECT_PROCESS": 0,
    "MemoryObject": 1,
    "FILE_OBJECT_CHAR": 2,
    "FILE_OBJECT_FILE": 3,
    "FILE_OBJECT_DIR": 4,
    "SUBJECT_UNIT": 5,
    "UnnamedPipeObject": 6,
    "FILE_OBJECT_UNIX_SOCKET": 7,
    "SRCSINK_UNKNOWN": 8,
    "FILE_OBJECT_LINK": 9,
    "NetFlowObject": 10,
    "FILE_OBJECT_BLOCK": 11,
}


def _select_word2vec(explicit: Path | None) -> tuple[Path, str]:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit, "explicit"
    if DEFAULT_NATIVE_W2V.is_file():
        return DEFAULT_NATIVE_W2V, "released_stgan"
    if DEFAULT_FALLBACK_W2V.is_file():
        LOGGER.warning(
            "Released STGAN word2vec_trace.model is absent; using the "
            "available pretrained TRACE E3 vectors from FLASH. The audit "
            "report records this compatibility fallback explicitly."
        )
        return DEFAULT_FALLBACK_W2V, "released_trace_e3_compatible_fallback"
    raise FileNotFoundError(
        "TRACE Word2Vec is missing. Expected either "
        f"{DEFAULT_NATIVE_W2V} or {DEFAULT_FALLBACK_W2V}."
    )


def _audit(graph, state: STGANSemanticState, *, word2vec_source: str) -> dict:
    state.validate_for_graph(graph, "trace", 4)
    schema = build_observed_action_schema(
        graph,
        "trace",
        action_dim=9,
        edge_type_names=state.magic_edge_type_names,
        node_type_names=state.node_type_names,
    )
    mapped_gt = 0
    if DEFAULT_UUID_MAP.is_file() and DEFAULT_GT.is_file():
        uuid_map = json.loads(DEFAULT_UUID_MAP.read_text(encoding="utf-8"))
        ground_truth = set(json.loads(DEFAULT_GT.read_text(encoding="utf-8")))
        mapped_gt = sum(1 for value in ground_truth if value in uuid_map)
    weights = STGAN_ROOT / "trained_weights" / "trace"
    return {
        "status": "PASS",
        "dataset": "trace",
        "test_index": 4,
        "nodes": int(state.original_node_count),
        "native_edges": int(state.native_edge_index.shape[1]),
        "magic_view_edges": int(graph.number_of_edges()),
        "zero_feature_nodes": int(np.sum(state.token_counts == 0)),
        "finite_embeddings": bool(np.all(np.isfinite(state.base_embeddings))),
        "finite_edge_times": bool(np.all(np.isfinite(state.native_edge_time))),
        "gat_checkpoints": len(list(weights.glob("gat*.pth"))),
        "tgn_checkpoints": len(list(weights.glob("tgn*.pth"))),
        "attention_checkpoints": len(list(weights.glob("selfattention*.pth"))),
        "mapped_native_ground_truth": int(mapped_gt),
        "semantic_actions": [spec.edge_name for spec in schema.actions],
        "word2vec_source": word2vec_source,
        "uses_positional_encoding": bool(state.use_positional_encoding),
    }


def prepare(
    *,
    archive: Path = DEFAULT_ARCHIVE,
    member: str = DEFAULT_MEMBER,
    output: Path = DEFAULT_STATE,
    uuid_map_output: Path = DEFAULT_UUID_MAP,
    word2vec_path: Path | None = None,
    force: bool = False,
) -> dict:
    if output.exists() and not force:
        raise FileExistsError(f"state exists: {output}; pass --force to replace")
    for path in (archive, DEFAULT_GT, DEFAULT_FLASH_STATE):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_w2v, source_kind = _select_word2vec(word2vec_path)
    database = output.with_suffix(".build.sqlite3")
    if database.exists():
        database.unlink()
    try:
        entity_types, event_rows = _stream_archive_to_sqlite(
            archive, member, database, "cmdLine"
        )
        LOGGER.info("Loading TRACE Word2Vec: %s", source_w2v)
        word2vec = _load_word2vec(source_w2v)
        graph = load_entity_level_dataset("trace", "test", 4)
        built = _build_state(
            database,
            entity_types,
            graph,
            word2vec,
            labels=LABELS,
            use_positional_encoding=True,
        )
        built = _align_to_audited_flash_state(built, DEFAULT_FLASH_STATE)

        output.parent.mkdir(parents=True, exist_ok=True)
        uuid_map_output.parent.mkdir(parents=True, exist_ok=True)
        _write_compat_vectors(word2vec, source_w2v, DEFAULT_COMPAT_W2V)
        uuid_map_output.write_text(
            json.dumps(built["node_map"], ensure_ascii=False), encoding="utf-8"
        )
        state = STGANSemanticState(
            original_node_count=int(graph.number_of_nodes()),
            base_embeddings=built["base_embeddings"],
            token_counts=built["token_counts"],
            model_labels=built["model_labels"],
            native_edge_index=built["native_edge_index"],
            native_edge_time=built["native_edge_time"],
            magic_edge_type_names=built["magic_edge_type_names"],
            node_type_names=built["node_type_names"],
            dataset="trace",
            test_index=4,
            global_node_offset=0,
            use_positional_encoding=True,
            word2vec_source=str(source_w2v.resolve()),
            source=str(output.resolve()),
        )
        state.to_npz(output)
        report = _audit(graph, state, word2vec_source=source_kind)
        report.update({
            "archive": str(archive.resolve()),
            "archive_member": member,
            "raw_target_event_rows": int(event_rows),
            "magic_edges_without_raw_match": built["magic_edges_without_raw_match"],
            "node_permutation_moved": built["node_permutation_moved"],
            "node_alignment_template": str(DEFAULT_FLASH_STATE.resolve()),
            "state": str(output.resolve()),
            "uuid_map": str(uuid_map_output.resolve()),
            "word2vec_model": str(source_w2v.resolve()),
            "word2vec_compat": str(DEFAULT_COMPAT_W2V.resolve()),
        })
        report_path = output.with_name(output.stem + "_audit.json")
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    finally:
        if database.exists():
            database.unlink()


def audit(state_path: Path = DEFAULT_STATE) -> dict:
    graph = load_entity_level_dataset("trace", "test", 4)
    state = STGANSemanticState.load(state_path)
    source_kind = (
        "released_stgan"
        if Path(state.word2vec_source) == DEFAULT_NATIVE_W2V.resolve()
        else "released_trace_e3_compatible_fallback"
    )
    report = _audit(graph, state, word2vec_source=source_kind)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def evaluate(
    state_path: Path = DEFAULT_STATE, *, device: str = "cuda:0", top_k: int = 10
) -> dict:
    graph = load_entity_level_dataset("trace", "test", 4)
    detector = STGANDetector(
        dataset_name="trace",
        device=device,
        top_k=top_k,
        threshold_source="native",
    )
    detector.set_evaluation_context(4)
    detector.attach_semantic_state(STGANSemanticState.load(state_path))
    detector.initialize_tp_baseline(graph)
    report = {
        "status": "PASS",
        "dataset": "trace",
        "test_index": 4,
        "ensemble_size": len(detector.snapshot_paths),
        "threshold": detector.best_threshold,
        "initial_alerts": len(detector.initial_alert_nodes),
        "initial_true_tp": detector.true_tp_count,
        "initial_false_positive": detector.false_positive_count,
        "cache_reused": detector.last_baseline_cache_hit,
        "raw_direct_metrics": detector.raw_initial_metrics,
        "released_code_metrics": detector.released_initial_metrics,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    prepare_parser.add_argument("--member", default=DEFAULT_MEMBER)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_STATE)
    prepare_parser.add_argument("--uuid-map", type=Path, default=DEFAULT_UUID_MAP)
    prepare_parser.add_argument("--word2vec", type=Path)
    prepare_parser.add_argument("--force", action="store_true")
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    evaluate_parser.add_argument("--device", default="cuda:0")
    evaluate_parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.command == "prepare":
        prepare(
            archive=args.archive,
            member=args.member,
            output=args.output,
            uuid_map_output=args.uuid_map,
            word2vec_path=args.word2vec,
            force=args.force,
        )
    elif args.command == "audit":
        audit(args.state)
    else:
        evaluate(args.state, device=args.device, top_k=args.top_k)


if __name__ == "__main__":
    main()
