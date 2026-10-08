"""Build and validate native FLASH states for DARPA E3 THEIA and TRACE.

This module replaces the download-heavy notebook execution path.  It keeps
FLASH's actor->object multigraph and Word2Vec features, while indexing every
entity with the matching MAGIC test graph node ID.  Ground-truth metadata is
not used while constructing scores, thresholds, or action schemas.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import sqlite3
import sys
import tarfile
import time
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Detector.Flash import FlashDetector
from Detector.Flash_State import FlashSemanticState
from MAGIC_main.action_schema import build_observed_action_schema
from MAGIC_main.utils.loaddata import load_entity_level_dataset


LOGGER = logging.getLogger("flash.e3.native")
ZERO_UUID = "00000000-0000-0000-0000-000000000000"

CONFIG = {
    "theia": {
        "archive": ROOT / "STGAN_master/data/theia/ta1-theia-e3-official-6r.json.tar.gz",
        "member": "ta1-theia-e3-official-6r.json.8",
        "test_index": 0,
        "exec_key": "cmdLine",
        "state": ROOT / "Flash_IDS_main/data_files/theia_flash_native_test0.npz",
        "w2v": ROOT / "Flash_IDS_main/trained_weights/theia/word2vec_theia_E3.model",
        "labels": {
            "SUBJECT_PROCESS": 0,
            "MemoryObject": 1,
            "FILE_OBJECT_BLOCK": 2,
            "NetFlowObject": 3,
            "PRINCIPAL_REMOTE": 4,
            "PRINCIPAL_LOCAL": 5,
        },
    },
    "trace": {
        "archive": ROOT / "STGAN_master/data/trace/ta1-trace-e3-official-1.json.tar.gz",
        "member": "ta1-trace-e3-official-1.json.4",
        "test_index": 4,
        "exec_key": "exec",
        "state": ROOT / "Flash_IDS_main/data_files/trace_flash_native_test4.npz",
        "w2v": ROOT / "Flash_IDS_main/trained_weights/trace/word2vec_trace_E3.model",
        "labels": {
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
        },
    },
}

try:
    import orjson as _orjson

    def _loads(value: bytes) -> Dict[str, Any]:
        return _orjson.loads(value)
except ImportError:
    def _loads(value: bytes) -> Dict[str, Any]:
        return json.loads(value)


def _datum(record: Dict[str, Any]) -> tuple[str, Optional[Dict[str, Any]]]:
    datum = record.get("datum")
    if not isinstance(datum, dict) or not datum:
        return "", None
    key, value = next(iter(datum.items()))
    return str(key).rsplit(".", 1)[-1], value if isinstance(value, dict) else None


def _uuid(value: Any) -> Optional[str]:
    # Entity definitions store ``uuid`` as a plain string, while Event
    # endpoints use the Avro union wrapper {"...UUID": "..."}.
    if isinstance(value, str):
        return None if value in ("", "null") else value
    if not isinstance(value, dict):
        return None
    for key, item in value.items():
        if str(key).endswith(".UUID"):
            return None if item in (None, "null") else str(item)
    return None


def _text(value: Any) -> str:
    if isinstance(value, dict):
        if "string" in value:
            value = value["string"]
        elif len(value) == 1:
            value = next(iter(value.values()))
    return "" if value is None else str(value)


def _entity_type(kind: str, value: Dict[str, Any]) -> str:
    explicit = _text(value.get("type"))
    if explicit:
        return explicit
    return kind


def _event_rows(value: Dict[str, Any], exec_key: str) -> Iterable[tuple]:
    actor = _uuid(value.get("subject"))
    if not actor:
        return
    action = _text(value.get("type"))
    timestamp = int(value.get("timestampNanos") or 0)
    properties = value.get("properties")
    properties = properties.get("map", properties) if isinstance(properties, dict) else {}
    executable = _text(properties.get(exec_key)) if isinstance(properties, dict) else ""
    objects = (
        (value.get("predicateObject"), value.get("predicateObjectPath")),
        (value.get("predicateObject2"), value.get("predicateObject2Path")),
    )
    for object_value, path_value in objects:
        obj = _uuid(object_value)
        if obj:
            yield actor, obj, action, timestamp, executable, _text(path_value)


def _archive_members(archive: Path) -> set[str]:
    with tarfile.open(archive, "r:gz") as handle:
        return {Path(member.name).name for member in handle if member.isfile()}


def _stream_archive_to_sqlite(
    archive: Path,
    target_member: str,
    database: Path,
    exec_key: str,
) -> tuple[Dict[str, str], int]:
    """Read the compressed archive once and spool only target events."""
    entity_types: Dict[str, str] = {}
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE events (seq INTEGER, actor TEXT, obj TEXT, action TEXT, "
        "ts INTEGER, executable TEXT, path TEXT)"
    )
    batch = []
    sequence = 0
    found = False
    start = time.perf_counter()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if not member.isfile():
                continue
            name = Path(member.name).name
            is_target = name == target_member
            found |= is_target
            stream = handle.extractfile(member)
            if stream is None:
                continue
            LOGGER.info("Scanning %s%s", name, " [FLASH test]" if is_target else "")
            for line_number, line in enumerate(stream, 1):
                try:
                    kind, value = _datum(_loads(line))
                except Exception:
                    continue
                if value is None:
                    continue
                if kind == "Event":
                    if is_target:
                        for row in _event_rows(value, exec_key):
                            batch.append((sequence, *row))
                            sequence += 1
                        if len(batch) >= 50_000:
                            connection.executemany(
                                "INSERT INTO events VALUES (?,?,?,?,?,?,?)", batch
                            )
                            connection.commit()
                            batch.clear()
                elif kind not in {
                    "Host", "TimeMarker", "StartMarker", "EndMarker",
                    "UnitDependency",
                }:
                    entity = _uuid(value.get("uuid"))
                    node_type = _entity_type(kind, value)
                    if entity and entity != ZERO_UUID and node_type != "SUBJECT_UNIT":
                        entity_types[entity] = node_type
                if line_number % 2_000_000 == 0:
                    LOGGER.info(
                        "%s lines=%d, entities=%d, target_events=%d, elapsed=%.1f min",
                        name, line_number, len(entity_types), sequence,
                        (time.perf_counter() - start) / 60.0,
                    )
    if batch:
        connection.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?)", batch)
        connection.commit()
    connection.execute("CREATE INDEX events_time ON events(ts, seq)")
    connection.commit()
    connection.close()
    if not found:
        raise FileNotFoundError(
            f"archive {archive} does not contain required FLASH test member "
            f"{target_member!r}"
        )
    return entity_types, sequence


def _event_cursor(database: Path) -> Iterator[tuple]:
    connection = sqlite3.connect(database)
    cursor = connection.execute(
        "SELECT actor,obj,action,ts,executable,path FROM events "
        "GROUP BY actor,obj,action,ts,executable,path ORDER BY ts,MIN(seq)"
    )
    try:
        yield from cursor
    finally:
        connection.close()


def _magic_oriented_endpoints(
    actor: str, obj: str, action: str
) -> tuple[str, str]:
    """Return the endpoint order used by MAGIC's E3 graph parser.

    MAGIC reverses data-ingress events before it sorts the event stream and
    assigns node IDs.  The ID assignment order matters: assigning IDs from the
    original CDM actor/object order first can produce a graph with the correct
    number of nodes but a different UUID-to-index mapping.  TRACE contains many
    RECV events, so that latent mismatch affects a large fraction of its edges.
    """
    if any(token in action.upper() for token in ("READ", "RECV", "LOAD")):
        return obj, actor
    return actor, obj


def _position_encoding(length: int = 100_000, dim: int = 30) -> np.ndarray:
    position = np.arange(length, dtype=np.float32)[:, None]
    scale = np.exp(
        np.arange(0, dim, 2, dtype=np.float32)
        * (-np.log(10000.0) / dim)
    )
    result = np.zeros((length, dim), dtype=np.float32)
    result[:, 0::2] = np.sin(position * scale)
    result[:, 1::2] = np.cos(position * scale)
    return result


def _global_offset(dataset: str, test_index: int) -> int:
    offset = 0
    for index in range(test_index):
        graph = load_entity_level_dataset(dataset, "test", index)
        offset += int(graph.number_of_nodes())
        del graph
    return offset


def _build_state(
    dataset: str,
    database: Path,
    entity_types: Dict[str, str],
    graph,
    word2vec,
    output: Path,
) -> tuple[FlashSemanticState, Dict[str, Any]]:
    config = CONFIG[dataset]
    node_map: Dict[str, int] = {}
    raw_types: list[str] = []
    first_relation: Dict[tuple[int, int], str] = {}
    accepted = 0

    for actor, obj, action, _ts, _executable, _path in _event_cursor(database):
        actor_type, object_type = entity_types.get(actor), entity_types.get(obj)
        if not actor_type or not object_type:
            continue

        # Reproduce MAGIC's node-ID assignment exactly.  Its parser first
        # reverses READ/RECV/LOAD edges and only then inserts previously unseen
        # source and destination UUIDs into the per-graph node map.
        magic_source, magic_destination = _magic_oriented_endpoints(
            actor, obj, action
        )
        for entity in (magic_source, magic_destination):
            if entity not in node_map:
                node_map[entity] = len(node_map)
                raw_types.append(entity_types[entity])

        magic_pair = (
            node_map[magic_source],
            node_map[magic_destination],
        )
        first_relation.setdefault(magic_pair, action)
        accepted += 1

    node_count = len(node_map)
    if node_count != int(graph.number_of_nodes()):
        raise ValueError(
            "native FLASH/MAGIC node mismatch: "
            f"raw={node_count}, MAGIC={graph.number_of_nodes()}. "
            "Use the exact official test member configured for this dataset."
        )

    label_map = config["labels"]
    model_labels = np.asarray([label_map[name] for name in raw_types], dtype=np.int64)
    class_count = FlashDetector.CLASS_COUNTS[dataset]
    if model_labels.min() < 0 or model_labels.max() > class_count:
        invalid = sorted({name for name in raw_types if label_map[name] > class_count})
        raise ValueError(
            f"raw node types {invalid} are outside FLASH's {class_count} classes"
        )

    magic_node_types = np.asarray(
        graph.ndata["attr"].argmax(dim=1).cpu(), dtype=np.int64
    )
    node_type_votes: Dict[int, Counter] = defaultdict(Counter)
    for magic_type, raw_type in zip(magic_node_types.tolist(), raw_types):
        node_type_votes[int(magic_type)][raw_type] += 1
    node_type_names = {
        key: votes.most_common(1)[0][0] for key, votes in node_type_votes.items()
    }

    word_sums = np.zeros((node_count, 30), dtype=np.float32)
    position_sums = np.zeros((node_count, 30), dtype=np.float32)
    token_counts = np.zeros(node_count, dtype=np.int64)
    document_counts = np.zeros(node_count, dtype=np.int64)
    incident_counts = np.zeros(node_count, dtype=np.int64)
    action_sets: Dict[int, set[str]] = defaultdict(set)
    edge_sources, edge_destinations = array("I"), array("I")
    pe = _position_encoding()
    vocabulary = word2vec.wv.key_to_index

    for actor, obj, action, _ts, executable, path in _event_cursor(database):
        if actor not in node_map or obj not in node_map:
            continue
        actor_id, object_id = node_map[actor], node_map[obj]
        edge_sources.append(actor_id)
        edge_destinations.append(object_id)
        tokens = (executable, action) + ((path,) if path else ())
        for node in {actor_id, object_id}:
            incident_counts[node] += 1
            action_sets[node].add(action)
            document_counts[node] += len(tokens)
            for token in tokens:
                if not token or token not in vocabulary:
                    continue
                position = int(token_counts[node])
                word_sums[node] += np.asarray(word2vec.wv[token], dtype=np.float32)
                if position < len(pe):
                    position_sums[node] += pe[position]
                token_counts[node] = position + 1

    embedding_sums = word_sums.copy()
    positional_nodes = document_counts < len(pe)
    embedding_sums[positional_nodes] += position_sums[positional_nodes]
    embeddings = np.zeros_like(embedding_sums)
    nonzero = token_counts > 0
    embeddings[nonzero] = embedding_sums[nonzero] / token_counts[nonzero, None]
    native_edge_index = np.vstack((
        np.frombuffer(edge_sources, dtype=np.uint32).astype(np.int64),
        np.frombuffer(edge_destinations, dtype=np.uint32).astype(np.int64),
    ))
    distinct_counts = np.asarray(
        [len(action_sets.get(node, ())) for node in range(node_count)],
        dtype=np.int64,
    )

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
    magic_edge_type_names = {
        key: votes.most_common(1)[0][0] for key, votes in edge_votes.items()
    }

    state = FlashSemanticState(
        original_node_count=node_count,
        base_embeddings=embeddings,
        token_counts=token_counts,
        document_counts=document_counts,
        native_edge_index=native_edge_index,
        incident_counts=incident_counts,
        distinct_action_counts=distinct_counts,
        model_labels=model_labels,
        magic_edge_type_names=magic_edge_type_names,
        node_type_names=node_type_names,
        test_index=int(config["test_index"]),
        global_node_offset=_global_offset(dataset, int(config["test_index"])),
        dataset=dataset,
        source=str(output.resolve()),
    )
    state.to_npz(output)
    report = audit_state(dataset, graph, state)
    report.update({
        "raw_distinct_events": accepted,
        "native_flash_edges": int(native_edge_index.shape[1]),
        "zero_feature_nodes": int(np.sum(~nonzero)),
        "magic_edges_without_raw_match": int(unmatched),
        "out_of_vocabulary_label_nodes": int(
            np.sum(model_labels >= class_count)
        ),
    })
    return state, report


def audit_state(dataset: str, graph, state: FlashSemanticState) -> Dict[str, Any]:
    state.validate_for_graph(graph, dataset)
    schema = build_observed_action_schema(
        graph,
        dataset,
        action_dim=9,
        edge_type_names=state.magic_edge_type_names,
        node_type_names=state.node_type_names,
    )
    node_types = graph.ndata["attr"].argmax(dim=1).cpu().tolist()
    edge_types = graph.edata["attr"].argmax(dim=1).cpu().tolist()
    src, dst = graph.edges()
    legal = {
        (spec.edge_type, pair[0], pair[1])
        for spec in schema.actions
        for pair in spec.valid_pairs
    }
    covered_by_action = Counter()
    covered = 0
    for source, destination, edge_type in zip(
        src.cpu().tolist(), dst.cpu().tolist(), edge_types
    ):
        triple = (
            int(edge_type),
            int(node_types[int(source)]),
            int(node_types[int(destination)]),
        )
        if triple in legal:
            covered += 1
            covered_by_action[int(edge_type)] += 1
    total = int(graph.number_of_edges())
    actions = []
    for spec in schema.actions:
        count = int(covered_by_action[spec.edge_type])
        actions.append({
            "slot": spec.action_id,
            "event": spec.edge_name,
            "magic_edge_type": spec.edge_type,
            "valid_pairs": [list(pair) for pair in spec.valid_pairs],
            "covered_edges": count,
            "single_coverage": count / total if total else 0.0,
        })
    return {
        "dataset": dataset,
        "test_index": state.test_index,
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": total,
        "schema_version": schema.version,
        "covered_edges": covered,
        "coverage": covered / total if total else 0.0,
        "actions": actions,
    }


def prepare(
    dataset: str,
    archive: Path,
    member: str,
    output: Path,
    *,
    force: bool,
    min_coverage: float,
) -> Dict[str, Any]:
    config = CONFIG[dataset]
    if output.exists() and not force:
        raise FileExistsError(f"state exists: {output}; pass --force to replace it")
    if not archive.is_file():
        raise FileNotFoundError(archive)
    members = _archive_members(archive)
    if member not in members:
        raise FileNotFoundError(
            f"{archive} lacks {member}. Available members: {sorted(members)}. "
            "Download the complete DARPA E3 archive before building this state."
        )
    database = output.with_suffix(".build.sqlite3")
    if database.exists():
        database.unlink()
    try:
        entity_types, event_count = _stream_archive_to_sqlite(
            archive, member, database, str(config["exec_key"])
        )
        LOGGER.info("Loading FLASH Word2Vec: %s", config["w2v"])
        import scipy.linalg
        if not hasattr(scipy.linalg, "triu"):
            scipy.linalg.triu = np.triu
        from gensim.models import Word2Vec
        word2vec = Word2Vec.load(str(config["w2v"]))
        graph = load_entity_level_dataset(dataset, "test", int(config["test_index"]))
        _state, report = _build_state(
            dataset, database, entity_types, graph, word2vec, output
        )
        report["raw_target_event_rows"] = event_count
        unmatched_ratio = (
            report["magic_edges_without_raw_match"] / report["graph_edges"]
            if report["graph_edges"] else 0.0
        )
        unresolved_actions = [
            action["event"]
            for action in report["actions"]
            if str(action["event"]).startswith("EDGE_TYPE_")
        ]
        if unmatched_ratio > 0.01:
            output.unlink(missing_ok=True)
            raise ValueError(
                f"FLASH/MAGIC UUID ordering failed: {unmatched_ratio:.2%} of "
                "MAGIC edges could not be matched to raw CDM events"
            )
        if unresolved_actions:
            output.unlink(missing_ok=True)
            raise ValueError(
                "selected attack actions lack native CDM semantics: "
                f"{unresolved_actions}"
            )
        if report["coverage"] < float(min_coverage):
            output.unlink(missing_ok=True)
            raise ValueError(
                f"semantic action coverage {report['coverage']:.2%} is below "
                f"required {min_coverage:.2%}; state was rejected"
            )
        report_path = output.with_name(output.stem + "_audit.json")
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        LOGGER.info("Saved state=%s and audit=%s", output, report_path)
        return report
    finally:
        if database.exists():
            database.unlink()


def evaluate(dataset: str, state_path: Path, device: str) -> Dict[str, Any]:
    config = CONFIG[dataset]
    graph = load_entity_level_dataset(dataset, "test", int(config["test_index"]))
    state = FlashSemanticState.load(state_path)
    detector = FlashDetector(dataset_name=dataset, device=device, top_k=1)
    detector.set_evaluation_context(int(config["test_index"]))
    detector.attach_semantic_state(state)
    detector.initialize_tp_baseline(graph)
    result = {
        "dataset": dataset,
        "test_index": int(config["test_index"]),
        "ensemble_size": len(detector.model_ensemble),
        "threshold": detector.best_threshold,
        "alerts": len(detector.initial_alert_nodes),
        "tp": detector.true_tp_count,
        "fp": detector.false_positive_count,
        **detector.initial_metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit", "evaluate"))
    parser.add_argument("--dataset", required=True, choices=("theia", "trace"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--member")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=0.90)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    config = CONFIG[args.dataset]
    archive = (args.archive or config["archive"]).resolve()
    member = args.member or str(config["member"])
    state_path = (args.state or config["state"]).resolve()
    if args.command == "prepare":
        prepare(
            args.dataset, archive, member, state_path,
            force=args.force, min_coverage=args.min_coverage,
        )
    elif args.command == "audit":
        graph = load_entity_level_dataset(
            args.dataset, "test", int(config["test_index"])
        )
        report = audit_state(
            args.dataset, graph, FlashSemanticState.load(state_path)
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        evaluate(args.dataset, state_path, args.device)


if __name__ == "__main__":
    main()
