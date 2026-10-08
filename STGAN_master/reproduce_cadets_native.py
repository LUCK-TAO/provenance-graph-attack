"""Standalone, low-memory reproduction of the released STGAN CADETS test.

This module intentionally has no dependency on MAGIC or FLASH.  It preserves
the released evaluator's data semantics while replacing its 4 GB in-memory
JSON/Pandas merge with an on-disk SQLite deduplication step.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import struct
import time
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from model.GAT import GAT
from model.MultiHeadSelfAttention import MultiHeadSelfAttentionModel
from model.TGN import TemporalGraphNetwork


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "cadets"
RAW_JSON = DATA_DIR / "ta1-cadets-e3-official-2.json"
EDGE_FILE = DATA_DIR / "cadets_test.txt"
GROUND_TRUTH = DATA_DIR / "cadets.json"
WORD_VECTORS = ROOT / "train" / "word2vec_cadets_compat.npz"
WEIGHT_DIR = ROOT / "trained_weights" / "cadets"
ROW_CACHE = DATA_DIR / "cadets_stgan_official_rows.sqlite3"
STATE_FILE = DATA_DIR / "cadets_stgan_official_test.npz"
RESULT_FILE = ROOT / "eval_result" / "stgan_cadets_official_reproduction.json"
ALERT_FILE = ROOT / "eval_result" / "stgan_cadets_official_alert_uuids.json"
TP_FILE = ROOT / "eval_result" / "stgan_cadets_official_initial_tp_uuids.json"

LOGGER = logging.getLogger("stgan.cadets.official")

NODE_TYPES = {
    "SUBJECT_PROCESS": 0,
    "FILE_OBJECT_FILE": 1,
    "FILE_OBJECT_UNIX_SOCKET": 2,
    "UnnamedPipeObject": 3,
    "NetFlowObject": 4,
    "FILE_OBJECT_DIR": 5,
}


def _nested_uuid(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    for key, item in value.items():
        if str(key).endswith(".UUID"):
            return None if item is None else str(item)
    return None


def _optional_string(value: Any) -> str:
    if isinstance(value, dict):
        if "string" in value:
            value = value["string"]
        elif len(value) == 1:
            value = next(iter(value.values()))
    return "" if value is None else str(value)


def _event_from_record(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    datum = record.get("datum")
    if not isinstance(datum, dict):
        return None
    for key, value in datum.items():
        if str(key).endswith(".Event") and isinstance(value, dict):
            return value
    return None


def _event_objects(event: dict[str, Any]) -> Iterable[tuple[str, str]]:
    first = _nested_uuid(event.get("predicateObject"))
    if first and first != "null":
        yield first, _optional_string(event.get("predicateObjectPath"))
    second = _nested_uuid(event.get("predicateObject2"))
    if second and second != "null":
        yield second, _optional_string(event.get("predicateObject2Path"))


def _event_exec(event: dict[str, Any]) -> str:
    properties = event.get("properties")
    if not isinstance(properties, dict):
        return ""
    mapping = properties.get("map", properties)
    if not isinstance(mapping, dict):
        return ""
    return _optional_string(mapping.get("exec"))


def _load_node_types(path: Path) -> tuple[dict[str, int], list[int], dict[str, int]]:
    uuid_to_raw: dict[str, int] = {}
    raw_types: list[int] = []
    stats = {"rows": 0, "bad_rows": 0, "timestamp_decreases": 0}
    previous_timestamp: Optional[int] = None

    def register(node_uuid: str, type_name: str) -> None:
        if type_name not in NODE_TYPES:
            raise ValueError(f"unknown CADETS node type {type_name!r}")
        label = NODE_TYPES[type_name]
        existing = uuid_to_raw.get(node_uuid)
        if existing is None:
            uuid_to_raw[node_uuid] = len(raw_types)
            raw_types.append(label)
        elif raw_types[existing] != label:
            raise ValueError(f"inconsistent type for UUID {node_uuid}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 6:
                stats["bad_rows"] += 1
                continue
            src, src_type, dst, dst_type, _, timestamp_text = parts
            register(src, src_type)
            register(dst, dst_type)
            timestamp = int(timestamp_text)
            if previous_timestamp is not None and timestamp < previous_timestamp:
                stats["timestamp_decreases"] += 1
            previous_timestamp = timestamp
            stats["rows"] += 1
    return uuid_to_raw, raw_types, stats


def _row_digest(
    src: int, dst: int, action: str, timestamp: int, executable: str, path: str
) -> bytes:
    digest = hashlib.sha256()
    digest.update(struct.pack("<IIQ", src, dst, timestamp))
    for value in (action, executable, path):
        encoded = value.encode("utf-8", errors="surrogatepass")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    return digest.digest()


def build_row_cache(
    raw_json: Path = RAW_JSON,
    edge_file: Path = EDGE_FILE,
    cache_path: Path = ROW_CACHE,
    *,
    force: bool = False,
) -> tuple[dict[str, int], list[int]]:
    for required in (raw_json, edge_file):
        if not required.is_file():
            raise FileNotFoundError(required)
    uuid_to_raw, raw_types, edge_stats = _load_node_types(edge_file)
    LOGGER.info("edge audit: %s; unique nodes=%d", edge_stats, len(raw_types))
    if cache_path.exists():
        if not force:
            with sqlite3.connect(cache_path) as connection:
                complete = connection.execute(
                    "SELECT value FROM metadata WHERE key='complete'"
                ).fetchone()
            if complete and complete[0] == "1":
                LOGGER.info("reusing completed row cache: %s", cache_path)
                return uuid_to_raw, raw_types
            raise RuntimeError(f"incomplete cache exists: {cache_path}; use --force")
        cache_path.unlink()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-262144;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE events (
            row_hash BLOB PRIMARY KEY,
            seq INTEGER NOT NULL,
            src INTEGER NOT NULL,
            dst INTEGER NOT NULL,
            action TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            executable TEXT NOT NULL,
            path TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )
    insert_sql = (
        "INSERT OR IGNORE INTO events "
        "(row_hash,seq,src,dst,action,timestamp,executable,path) "
        "VALUES (?,?,?,?,?,?,?,?)"
    )
    batch: list[tuple[Any, ...]] = []
    parsed_events = accepted_rows = malformed = 0
    started = time.perf_counter()
    with raw_json.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if b"Event" not in line:
                continue
            try:
                event = _event_from_record(json.loads(line))
            except Exception:
                malformed += 1
                continue
            if event is None:
                continue
            parsed_events += 1
            actor_uuid = _nested_uuid(event.get("subject"))
            actor = uuid_to_raw.get(actor_uuid or "")
            if actor is None:
                continue
            action = _optional_string(event.get("type"))
            executable = _event_exec(event)
            try:
                timestamp = int(event.get("timestampNanos", 0))
            except (TypeError, ValueError):
                timestamp = 0
            for ordinal, (object_uuid, path) in enumerate(_event_objects(event)):
                obj = uuid_to_raw.get(object_uuid)
                if obj is None:
                    continue
                seq = line_number * 2 + ordinal
                batch.append(
                    (
                        _row_digest(actor, obj, action, timestamp, executable, path),
                        seq,
                        actor,
                        obj,
                        action,
                        timestamp,
                        executable,
                        path,
                    )
                )
                accepted_rows += 1
            if len(batch) >= 20_000:
                connection.executemany(insert_sql, batch)
                connection.commit()
                batch.clear()
            if line_number % 1_000_000 == 0:
                unique_rows = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                LOGGER.info(
                    "raw lines=%d accepted=%d unique=%d elapsed=%.1f min",
                    line_number,
                    accepted_rows,
                    unique_rows,
                    (time.perf_counter() - started) / 60.0,
                )
    if batch:
        connection.executemany(insert_sql, batch)
        connection.commit()
    unique_rows = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    metadata = {
        "complete": "1",
        "source": str(raw_json.resolve()),
        "parsed_events": str(parsed_events),
        "accepted_rows_before_dedup": str(accepted_rows),
        "unique_rows": str(unique_rows),
        "malformed": str(malformed),
    }
    connection.executemany("INSERT INTO metadata VALUES (?,?)", metadata.items())
    connection.commit()
    connection.close()
    LOGGER.info("completed row cache: %s", metadata)
    return uuid_to_raw, raw_types


def _position_vector(position: int, dim: int = 30) -> np.ndarray:
    scale = np.exp(
        np.arange(0, dim, 2, dtype=np.float32) * (-np.log(10000.0) / dim)
    )
    result = np.zeros(dim, dtype=np.float32)
    result[0::2] = np.sin(float(position) * scale)
    result[1::2] = np.cos(float(position) * scale)
    return result


def build_state(
    cache_path: Path = ROW_CACHE,
    edge_file: Path = EDGE_FILE,
    vector_path: Path = WORD_VECTORS,
    output_path: Path = STATE_FILE,
    *,
    force: bool = False,
) -> Path:
    if output_path.exists() and not force:
        raise FileExistsError(f"state exists: {output_path}; use --force")
    uuid_to_raw, raw_types, _ = _load_node_types(edge_file)
    raw_to_uuid = [""] * len(raw_types)
    for node_uuid, raw_id in uuid_to_raw.items():
        raw_to_uuid[raw_id] = node_uuid
    with np.load(vector_path, allow_pickle=False) as payload:
        keys = np.asarray(payload["keys"], dtype=np.str_)
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
    vocabulary = {str(key): i for i, key in enumerate(keys)}

    raw_to_node = np.full(len(raw_types), -1, dtype=np.int64)
    uuids: list[str] = []
    labels: list[int] = []
    word_sums = np.zeros((len(raw_types), 30), dtype=np.float32)
    position_sums = np.zeros((len(raw_types), 30), dtype=np.float32)
    token_counts = np.zeros(len(raw_types), dtype=np.int64)
    sources = array("I")
    destinations = array("I")
    times = array("f")

    def node_id(raw_id: int) -> int:
        existing = int(raw_to_node[raw_id])
        if existing >= 0:
            return existing
        value = len(uuids)
        raw_to_node[raw_id] = value
        uuids.append(raw_to_uuid[raw_id])
        labels.append(raw_types[raw_id])
        return value

    connection = sqlite3.connect(cache_path)
    cursor = connection.execute(
        "SELECT src,dst,action,timestamp,executable,path "
        "FROM events ORDER BY timestamp,seq"
    )
    started = time.perf_counter()
    for row_number, (raw_src, raw_dst, action, timestamp, executable, path) in enumerate(cursor, 1):
        src = node_id(int(raw_src))
        dst = node_id(int(raw_dst))
        sources.append(src)
        destinations.append(dst)
        times.append(float(timestamp))
        tokens = (executable, action) + ((path,) if path else ())
        for node in (src, dst):
            for token in tokens:
                vector_index = vocabulary.get(str(token))
                if vector_index is None:
                    continue
                position = int(token_counts[node])
                word_sums[node] += vectors[vector_index]
                position_sums[node] += _position_vector(position)
                token_counts[node] = position + 1
        if row_number % 500_000 == 0:
            LOGGER.info(
                "state rows=%d nodes=%d elapsed=%.1f min",
                row_number,
                len(uuids),
                (time.perf_counter() - started) / 60.0,
            )
    connection.close()
    node_count = len(uuids)
    embeddings = np.zeros((node_count, 30), dtype=np.float32)
    nonzero = token_counts[:node_count] > 0
    embeddings[nonzero] = (
        word_sums[:node_count][nonzero] + position_sums[:node_count][nonzero]
    ) / token_counts[:node_count][nonzero, None]
    edge_index = np.vstack(
        (
            np.frombuffer(sources, dtype=np.uint32).astype(np.int64),
            np.frombuffer(destinations, dtype=np.uint32).astype(np.int64),
        )
    )
    edge_time = np.frombuffer(times, dtype=np.float32).copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        version=np.asarray(1, dtype=np.int64),
        uuids=np.asarray(uuids, dtype=np.str_),
        labels=np.asarray(labels, dtype=np.int64),
        embeddings=embeddings,
        token_counts=token_counts[:node_count],
        edge_index=edge_index,
        edge_time=edge_time,
    )
    LOGGER.info(
        "saved native state: %s nodes=%d edges=%d zero_feature_nodes=%d",
        output_path,
        node_count,
        edge_index.shape[1],
        int(np.sum(~nonzero)),
    )
    return output_path


def _load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    payload = torch.load(path, map_location=device)
    translated = dict(payload)
    for key in list(payload):
        if key.endswith(".lin.weight"):
            stem = key[: -len("lin.weight")]
            translated[f"{stem}lin_src.weight"] = payload[key]
            translated[f"{stem}lin_dst.weight"] = payload[key]
            del translated[key]
    model.load_state_dict(translated, strict=True)


def _expand_two_hops(seed: set[int], edges: np.ndarray) -> set[int]:
    current = set(seed)
    result: set[int] = set()
    for _ in range(2):
        if not current:
            break
        values = np.fromiter(current, dtype=np.int64)
        mask = np.isin(edges[0], values) | np.isin(edges[1], values)
        neighbors = set(edges[:, mask].reshape(-1).tolist())
        result.update(neighbors)
        current = neighbors
    return result


def _metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate(
    state_path: Path = STATE_FILE,
    ground_truth_path: Path = GROUND_TRUTH,
    result_path: Path = RESULT_FILE,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    with np.load(state_path, allow_pickle=False) as payload:
        uuids = np.asarray(payload["uuids"], dtype=np.str_)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        edge_index = np.asarray(payload["edge_index"], dtype=np.int64)
        edge_time = np.asarray(payload["edge_time"], dtype=np.float32)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    graph = Data(
        x=torch.from_numpy(np.eye(6, dtype=np.float32)[labels]),
        y=torch.from_numpy(labels),
        edge_index=torch.from_numpy(edge_index),
        edge_time=torch.from_numpy(edge_time),
    )
    graph.n_id = torch.arange(len(labels), dtype=torch.long)
    text_embeddings = torch.from_numpy(embeddings)
    persistent = np.ones(len(labels), dtype=bool)
    loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=26000, shuffle=False)
    started = time.perf_counter()
    for snapshot in range(50):
        gat = GAT(in_channel=6, out_channel=30).to(device)
        tgn = TemporalGraphNetwork(in_channels=60, out_channels=32).to(device)
        attention = MultiHeadSelfAttentionModel(
            in_channels=32, out_channels=6, num_classes=6, num_heads=4
        ).to(device)
        _load_checkpoint(gat, WEIGHT_DIR / f"gat{snapshot}.pth", device)
        _load_checkpoint(tgn, WEIGHT_DIR / f"tgn{snapshot}.pth", device)
        _load_checkpoint(attention, WEIGHT_DIR / f"selfattention{snapshot}.pth", device)
        gat.eval(); tgn.eval(); attention.eval()
        with torch.inference_mode():
            for subgraph in loader:
                node_ids = subgraph.n_id.cpu().numpy().astype(np.int64, copy=False)
                x = subgraph.x.to(device)
                sub_edges = subgraph.edge_index.to(device)
                sub_times = subgraph.edge_time.to(device)
                output = gat(x, sub_edges)
                text = text_embeddings.index_select(0, subgraph.n_id).to(device)
                output = tgn(torch.cat((output, text), dim=1), sub_edges, sub_times)
                logits = attention(output.unsqueeze(1))
                correct = logits.argmax(dim=1).eq(subgraph.y.to(device)).cpu().numpy()
                persistent[node_ids[correct]] = False
        LOGGER.info(
            "snapshot=%d/50 persistent_alerts=%d elapsed=%.1f min",
            snapshot + 1,
            int(np.sum(persistent)),
            (time.perf_counter() - started) / 60.0,
        )
        del gat, tgn, attention
        if device.type == "cuda":
            torch.cuda.empty_cache()

    alerts_idx = set(np.flatnonzero(persistent).tolist())
    uuid_to_index = {str(value): i for i, value in enumerate(uuids.tolist())}
    ground_truth_uuids = set(json.loads(ground_truth_path.read_text(encoding="utf-8")))
    ground_truth_idx = {uuid_to_index[value] for value in ground_truth_uuids if value in uuid_to_index}
    all_idx = set(range(len(uuids)))
    raw_tp = alerts_idx & ground_truth_idx
    raw_fp = alerts_idx - ground_truth_idx
    raw_fn = ground_truth_idx - alerts_idx
    raw_tn = all_idx - (ground_truth_idx | alerts_idx)
    raw_metrics = _metrics(len(raw_tp), len(raw_fp), len(raw_fn), len(raw_tn))

    two_hop_gt = _expand_two_hops(ground_truth_idx, edge_index)
    two_hop_tp = _expand_two_hops(raw_tp, edge_index)
    tolerated_fp = raw_fp - two_hop_gt
    tolerated_tp = raw_tp | (raw_fn & two_hop_tp)
    tolerated_fn = raw_fn - two_hop_tp
    test_only_two_hop_metrics = _metrics(
        len(tolerated_tp), len(tolerated_fp), len(tolerated_fn), len(raw_tn)
    )
    alert_uuids = sorted(str(uuids[index]) for index in alerts_idx)
    initial_tp_uuids = sorted(str(uuids[index]) for index in raw_tp)
    ALERT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALERT_FILE.write_text(
        json.dumps(alert_uuids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    TP_FILE.write_text(
        json.dumps(initial_tp_uuids, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Cross-check the vectorized implementation against the exact helper
    # shipped in the released source tree.
    from utils.helper import helper as released_helper

    helper_tp, helper_fp = released_helper(
        set(alert_uuids), set(str(value) for value in uuids.tolist()),
        ground_truth_uuids, edge_index, uuids,
    )
    if set(helper_tp) != {str(uuids[index]) for index in tolerated_tp}:
        raise AssertionError("released helper TP set differs from vectorized cross-check")
    if set(helper_fp) != {str(uuids[index]) for index in tolerated_fp}:
        raise AssertionError("released helper FP set differs from vectorized cross-check")
    helper_fn = ground_truth_uuids - set(helper_tp)
    helper_tn = set(str(value) for value in uuids.tolist()) - (
        ground_truth_uuids | set(alert_uuids)
    )
    released_metrics = _metrics(
        len(helper_tp), len(helper_fp), len(helper_fn), len(helper_tn)
    )
    paper = {"precision": 0.9840, "accuracy": 0.9994, "recall": 0.9983, "f1": 0.9911}
    report = {
        "status": "PASS",
        "scope": "standalone released STGAN CADETS evaluator",
        "uses_magic": False,
        "uses_flash": False,
        "nodes": len(uuids),
        "edges_after_released_drop_duplicates": int(edge_index.shape[1]),
        "ground_truth_nodes_total": len(ground_truth_uuids),
        "ground_truth_nodes_in_test": len(ground_truth_idx),
        "persistent_alerts": len(alerts_idx),
        "alert_uuid_file": str(ALERT_FILE.resolve()),
        "initial_direct_tp_uuid_file": str(TP_FILE.resolve()),
        "raw_metrics": raw_metrics,
        "released_two_hop_metrics": released_metrics,
        "test_only_two_hop_diagnostic": test_only_two_hop_metrics,
        "released_helper_crosscheck": {
            "tp": len(helper_tp),
            "fp": len(helper_fp),
            "fn": len(helper_fn),
            "tn": len(helper_tn),
            "matches": True,
        },
        "paper_metrics": paper,
        "absolute_delta_from_paper": {
            key: abs(float(released_metrics[key]) - value) for key, value in paper.items()
        },
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate", "all"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.command in {"prepare", "all"}:
        build_row_cache(force=args.force)
        build_state(force=args.force)
    if args.command in {"evaluate", "all"}:
        evaluate(device_name=args.device)


if __name__ == "__main__":
    main()
