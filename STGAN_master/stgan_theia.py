"""Prepare, audit, and evaluate the released STGAN DARPA E3 THEIA split.

The original evaluator consumes only ``ta1-theia-e3-official-6r.json.8``.
This module streams that member directly from the existing ``.tar.gz`` and
never leaves the multi-gigabyte raw JSON expanded on disk.  Node IDs are
constructed with the same endpoint ordering as the MAGIC THEIA test graph so
the resulting state can be used by the common black-box attack pipeline.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Detector.STGAN import STGANDetector
from Detector.STGAN_State import STGANSemanticState
from Flash_IDS_main.flash_e3 import (
    _event_cursor,
    _magic_oriented_endpoints,
    _stream_archive_to_sqlite,
)
from MAGIC_main.action_schema import build_observed_action_schema
from MAGIC_main.utils.loaddata import load_entity_level_dataset


LOGGER = logging.getLogger("stgan.theia.native")
STGAN_ROOT = Path(__file__).resolve().parent
DATA_DIR = STGAN_ROOT / "data" / "theia"
DEFAULT_ARCHIVE = DATA_DIR / "ta1-theia-e3-official-6r.json.tar.gz"
DEFAULT_MEMBER = "ta1-theia-e3-official-6r.json.8"
DEFAULT_STATE = DATA_DIR / "theia_stgan_native_test0.npz"
DEFAULT_UUID_MAP = ROOT / "MAGIC_main" / "data" / "theia" / "uuid_to_index_test0.json"
DEFAULT_GT = DATA_DIR / "theia.json"
DEFAULT_NATIVE_W2V = STGAN_ROOT / "train" / "word2vec_theia.model"
DEFAULT_FALLBACK_W2V = (
    ROOT / "Flash_IDS_main" / "trained_weights" / "theia"
    / "word2vec_theia_E3.model"
)
DEFAULT_COMPAT_W2V = STGAN_ROOT / "train" / "word2vec_theia_compat.npz"
DEFAULT_FLASH_STATE = (
    ROOT / "Flash_IDS_main" / "data_files" / "theia_flash_native_test0.npz"
)

LABELS = {
    "SUBJECT_PROCESS": 0,
    "MemoryObject": 1,
    "FILE_OBJECT_BLOCK": 2,
    "NetFlowObject": 3,
    "PRINCIPAL_REMOTE": 4,
    "PRINCIPAL_LOCAL": 5,
}


def _load_word2vec(path: Path):
    import scipy.linalg

    if not hasattr(scipy.linalg, "triu"):
        scipy.linalg.triu = np.triu
    from gensim.models import Word2Vec

    model = Word2Vec.load(str(path))
    if int(model.vector_size) != 30:
        raise ValueError(f"expected 30-D THEIA Word2Vec, got {model.vector_size}")
    return model


def _select_word2vec(explicit: Path | None) -> tuple[Path, str]:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit, "explicit"
    if DEFAULT_NATIVE_W2V.is_file():
        return DEFAULT_NATIVE_W2V, "released_stgan"
    if DEFAULT_FALLBACK_W2V.is_file():
        LOGGER.warning(
            "Released STGAN word2vec_theia.model is absent; using the "
            "available pretrained THEIA E3 vectors from FLASH. The audit "
            "report records this compatibility fallback explicitly."
        )
        return DEFAULT_FALLBACK_W2V, "released_theia_e3_compatible_fallback"
    raise FileNotFoundError(
        "THEIA Word2Vec is missing. Expected either "
        f"{DEFAULT_NATIVE_W2V} or {DEFAULT_FALLBACK_W2V}."
    )


def _write_compat_vectors(model, source: Path, output: Path) -> None:
    keys = np.asarray(model.wv.index_to_key, dtype=np.str_)
    vectors = np.asarray(model.wv.vectors, dtype=np.float32)
    if vectors.shape != (len(keys), 30):
        raise ValueError(f"unexpected THEIA Word2Vec matrix: {vectors.shape}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        version=np.asarray(1, dtype=np.int64),
        source_model=np.asarray(str(source.resolve())),
        keys=keys,
        vectors=vectors,
    )


def _position_vector(position: int, dim: int = 30) -> np.ndarray:
    scale = np.exp(
        np.arange(0, dim, 2, dtype=np.float32)
        * (-np.log(10000.0) / dim)
    )
    value = np.zeros(dim, dtype=np.float32)
    value[0::2] = np.sin(float(position) * scale)
    value[1::2] = np.cos(float(position) * scale)
    return value


def _build_state(
    database: Path,
    entity_types: Dict[str, str],
    graph,
    word2vec,
    *,
    labels: Dict[str, int] | None = None,
    use_positional_encoding: bool = False,
):
    labels = LABELS if labels is None else labels
    node_map: Dict[str, int] = {}
    raw_types: list[str] = []
    first_relation: Dict[tuple[int, int], str] = {}

    for actor, obj, action, _ts, _executable, _path in _event_cursor(database):
        actor_type, object_type = entity_types.get(actor), entity_types.get(obj)
        if actor_type not in labels or object_type not in labels:
            continue
        magic_source, magic_destination = _magic_oriented_endpoints(
            actor, obj, action
        )
        for entity in (magic_source, magic_destination):
            if entity not in node_map:
                node_map[entity] = len(node_map)
                raw_types.append(entity_types[entity])
        first_relation.setdefault(
            (node_map[magic_source], node_map[magic_destination]), action
        )

    node_count = len(node_map)
    if node_count != int(graph.number_of_nodes()):
        raise ValueError(
            "native STGAN/MAGIC node mismatch: "
            f"raw={node_count}, MAGIC={graph.number_of_nodes()}"
        )

    model_labels = np.asarray([labels[name] for name in raw_types], dtype=np.int64)
    word_sums = np.zeros((node_count, 30), dtype=np.float32)
    position_sums = np.zeros((node_count, 30), dtype=np.float32)
    position_table = None
    if use_positional_encoding:
        positions = np.arange(100_000, dtype=np.float32)[:, None]
        scale = np.exp(
            np.arange(0, 30, 2, dtype=np.float32)
            * (-np.log(10000.0) / 30)
        )[None, :]
        position_table = np.empty((100_000, 30), dtype=np.float32)
        position_table[:, 0::2] = np.sin(positions * scale)
        position_table[:, 1::2] = np.cos(positions * scale)
    token_counts = np.zeros(node_count, dtype=np.int64)
    edge_sources, edge_destinations = array("I"), array("I")
    edge_times = array("f")
    vocabulary = word2vec.wv.key_to_index

    for actor, obj, action, timestamp, executable, path in _event_cursor(database):
        if actor not in node_map or obj not in node_map:
            continue
        actor_id, object_id = node_map[actor], node_map[obj]
        edge_sources.append(actor_id)
        edge_destinations.append(object_id)
        edge_times.append(float(timestamp))
        tokens = (executable, action) + ((path,) if path else ())
        # The released THEIA prepare_graph appends the phrase independently to
        # both endpoint roles. Preserve two contributions for self loops.
        for node in (actor_id, object_id):
            for token in tokens:
                if token and token in vocabulary:
                    word_sums[node] += np.asarray(
                        word2vec.wv[token], dtype=np.float32
                    )
                    if use_positional_encoding:
                        position = int(token_counts[node])
                        position_sums[node] += (
                            position_table[position]
                            if position < len(position_table)
                            else _position_vector(position)
                        )
                    token_counts[node] += 1

    embeddings = np.zeros_like(word_sums)
    nonzero = token_counts > 0
    embeddings[nonzero] = (
        word_sums[nonzero] + position_sums[nonzero]
    ) / token_counts[nonzero, None]
    native_edge_index = np.vstack((
        np.frombuffer(edge_sources, dtype=np.uint32).astype(np.int64),
        np.frombuffer(edge_destinations, dtype=np.uint32).astype(np.int64),
    ))
    native_edge_time = np.frombuffer(edge_times, dtype=np.float32).copy()

    magic_node_types = np.asarray(
        graph.ndata["attr"].argmax(dim=1).cpu(), dtype=np.int64
    )
    node_votes: Dict[int, Counter] = defaultdict(Counter)
    for magic_type, raw_type in zip(magic_node_types.tolist(), raw_types):
        node_votes[int(magic_type)][raw_type] += 1
    node_type_names = {
        key: votes.most_common(1)[0][0] for key, votes in node_votes.items()
    }

    src, dst = graph.edges()
    magic_edge_types = graph.edata["attr"].argmax(dim=1).cpu().tolist()
    edge_votes: Dict[int, Counter] = defaultdict(Counter)
    unmatched = 0
    for source, destination, edge_type in zip(
        src.cpu().tolist(), dst.cpu().tolist(), magic_edge_types
    ):
        action = first_relation.get((int(source), int(destination)))
        if action is None:
            unmatched += 1
        else:
            edge_votes[int(edge_type)][action] += 1
    edge_type_names = {
        key: votes.most_common(1)[0][0] for key, votes in edge_votes.items()
    }
    return {
        "node_map": node_map,
        "model_labels": model_labels,
        "base_embeddings": embeddings,
        "token_counts": token_counts,
        "native_edge_index": native_edge_index,
        "native_edge_time": native_edge_time,
        "magic_edge_type_names": edge_type_names,
        "node_type_names": node_type_names,
        "zero_feature_nodes": int(np.sum(~nonzero)),
        "magic_edges_without_raw_match": int(unmatched),
    }


def _align_to_audited_flash_state(built: dict, flash_path: Path) -> dict:
    """Permute raw-stream IDs onto the already audited MAGIC node IDs.

    A small set of equal-timestamp events has an implementation-dependent
    insertion order.  Both native states retain the exact same event rows, so
    their endpoint sequences define a complete, ground-truth-free bijection.
    """
    if not flash_path.is_file():
        raise FileNotFoundError(
            "Exact THEIA node alignment requires the audited FLASH native "
            f"state: {flash_path}"
        )
    with np.load(flash_path, allow_pickle=False) as flash:
        flash_edges = np.asarray(flash["native_edge_index"], dtype=np.int64)
        flash_labels = np.asarray(flash["model_labels"], dtype=np.int64)
        edge_type_names = {
            int(key): str(value)
            for key, value in zip(
                np.asarray(flash["magic_edge_type_ids"]).tolist(),
                np.asarray(flash["magic_edge_type_names"]).tolist(),
            )
        }
        node_type_names = {
            int(key): str(value)
            for key, value in zip(
                np.asarray(flash["node_type_ids"]).tolist(),
                np.asarray(flash["node_type_names"]).tolist(),
            )
        }

    own_edges = np.asarray(built["native_edge_index"], dtype=np.int64)
    if flash_edges.shape != own_edges.shape:
        raise ValueError(
            "STGAN/FLASH THEIA edge count mismatch: "
            f"STGAN={own_edges.shape[1]}, FLASH={flash_edges.shape[1]}"
        )
    node_count = len(built["model_labels"])
    own_flat, flash_flat = own_edges.reshape(-1), flash_edges.reshape(-1)
    mapping_min = np.full(node_count, node_count, dtype=np.int64)
    mapping_max = np.full(node_count, -1, dtype=np.int64)
    np.minimum.at(mapping_min, own_flat, flash_flat)
    np.maximum.at(mapping_max, own_flat, flash_flat)
    conflicts = np.flatnonzero(mapping_min != mapping_max)
    if len(conflicts):
        raise ValueError(
            "event endpoint sequence does not define a consistent THEIA "
            f"node mapping; conflicts={len(conflicts)}"
        )
    mapping = mapping_min
    if mapping.min() != 0 or mapping.max() != node_count - 1:
        raise ValueError("THEIA node alignment is not a dense zero-based range")
    if len(np.unique(mapping)) != node_count:
        raise ValueError("THEIA raw-to-MAGIC node alignment is not bijective")
    remapped_edges = mapping[own_edges]
    if not np.array_equal(remapped_edges, flash_edges):
        raise ValueError("THEIA edge alignment failed after applying the bijection")

    def remap(values):
        values = np.asarray(values)
        result = np.empty_like(values)
        result[mapping] = values
        return result

    remapped_labels = remap(built["model_labels"])
    if not np.array_equal(remapped_labels, flash_labels):
        raise ValueError("THEIA entity labels disagree with the audited native state")
    built["base_embeddings"] = remap(built["base_embeddings"])
    built["token_counts"] = remap(built["token_counts"])
    built["model_labels"] = remapped_labels
    built["native_edge_index"] = flash_edges
    built["magic_edge_type_names"] = edge_type_names
    built["node_type_names"] = node_type_names
    built["node_map"] = {
        node_uuid: int(mapping[raw_index])
        for node_uuid, raw_index in built["node_map"].items()
    }
    built["node_permutation_moved"] = int(np.sum(mapping != np.arange(node_count)))
    return built


def _audit(graph, state: STGANSemanticState, *, word2vec_source: str) -> dict:
    state.validate_for_graph(graph, "theia", 0)
    schema = build_observed_action_schema(
        graph,
        "theia",
        action_dim=9,
        edge_type_names=state.magic_edge_type_names,
        node_type_names=state.node_type_names,
    )
    mapped_gt = 0
    if DEFAULT_UUID_MAP.is_file() and DEFAULT_GT.is_file():
        uuid_map = json.loads(DEFAULT_UUID_MAP.read_text(encoding="utf-8"))
        gt = set(json.loads(DEFAULT_GT.read_text(encoding="utf-8")))
        mapped_gt = sum(1 for value in gt if value in uuid_map)
    return {
        "status": "PASS",
        "dataset": "theia",
        "test_index": 0,
        "nodes": int(state.original_node_count),
        "native_edges": int(state.native_edge_index.shape[1]),
        "magic_view_edges": int(graph.number_of_edges()),
        "zero_feature_nodes": int(np.sum(state.token_counts == 0)),
        "finite_embeddings": bool(np.all(np.isfinite(state.base_embeddings))),
        "finite_edge_times": bool(np.all(np.isfinite(state.native_edge_time))),
        "gat_checkpoints": len(list((STGAN_ROOT / "trained_weights" / "theia").glob("gat*.pth"))),
        "tgn_checkpoints": len(list((STGAN_ROOT / "trained_weights" / "theia").glob("tgn*.pth"))),
        "attention_checkpoints": len(list((STGAN_ROOT / "trained_weights" / "theia").glob("selfattention*.pth"))),
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
    for path in (archive, DEFAULT_GT):
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
        LOGGER.info("Loading THEIA Word2Vec: %s", source_w2v)
        word2vec = _load_word2vec(source_w2v)
        graph = load_entity_level_dataset("theia", "test", 0)
        built = _build_state(database, entity_types, graph, word2vec)
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
            dataset="theia",
            test_index=0,
            global_node_offset=0,
            use_positional_encoding=False,
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
    graph = load_entity_level_dataset("theia", "test", 0)
    state = STGANSemanticState.load(state_path)
    source_kind = (
        "released_stgan"
        if Path(state.word2vec_source) == DEFAULT_NATIVE_W2V.resolve()
        else "released_theia_e3_compatible_fallback"
    )
    report = _audit(graph, state, word2vec_source=source_kind)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def evaluate(
    state_path: Path = DEFAULT_STATE, *, device: str = "cuda:0", top_k: int = 10
) -> dict:
    graph = load_entity_level_dataset("theia", "test", 0)
    detector = STGANDetector(
        dataset_name="theia",
        device=device,
        top_k=top_k,
        threshold_source="native",
    )
    detector.set_evaluation_context(0)
    detector.attach_semantic_state(STGANSemanticState.load(state_path))
    detector.initialize_tp_baseline(graph)
    report = {
        "status": "PASS",
        "dataset": "theia",
        "test_index": 0,
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
