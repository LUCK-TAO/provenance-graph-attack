"""Rebuild the CADETS attack graph in the persisted UUID index space.

The archived graph predates the UUID alignment required by native detector
sidecars. This builder reads the released entity edge list, applies the
original direction convention, keeps the first event for every directed node
pair (the original NetworkX ``DiGraph`` behavior), and writes a DGL graph whose
node IDs exactly match ``uuid_to_index_test0.json``.
"""

from __future__ import annotations

import argparse
from array import array
import json
import os
from pathlib import Path
import pickle

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from action_schema import CADETS_EDGE_TYPES, CADETS_NODE_TYPES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = ROOT / "MAGIC_main" / "data" / "cadets" / "test0.pkl"
DEFAULT_UUID_MAP = (
    ROOT / "MAGIC_main" / "data" / "cadets" / "uuid_to_index_test0.json"
)
DEFAULT_EDGE_DIR = ROOT / "STGAN_master" / "data" / "cadets"


def rebuild(graph_path: Path, uuid_map_path: Path, edge_dir: Path) -> None:
    edge_path = edge_dir / "cadets_test.txt"
    if not edge_path.is_file():
        raise FileNotFoundError(edge_path)
    mapping = {
        str(key): int(value)
        for key, value in json.loads(
            uuid_map_path.read_text(encoding="utf-8")
        ).items()
    }
    node_count = len(mapping)
    if sorted(mapping.values()) != list(range(node_count)):
        raise ValueError("UUID indices must be a dense zero-based range")

    node_name_to_id = {name: key for key, name in CADETS_NODE_TYPES.items()}
    edge_name_to_id = {name: key for key, name in CADETS_EDGE_TYPES.items()}
    node_types = torch.full((node_count,), -1, dtype=torch.long)
    sources = array("I")
    destinations = array("I")
    edge_types = array("B")
    seen_pairs: set[int] = set()

    rows = 0
    with edge_path.open("r", encoding="utf-8") as handle:
        for rows, line in enumerate(handle, start=1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 6:
                raise ValueError(f"malformed CADETS edge row {rows}")
            src_uuid, src_name, dst_uuid, dst_name, event_name, _ = parts
            src = mapping[src_uuid]
            dst = mapping[dst_uuid]
            src_type = node_name_to_id[src_name]
            dst_type = node_name_to_id[dst_name]
            for node, node_type in ((src, src_type), (dst, dst_type)):
                existing = int(node_types[node])
                if existing not in (-1, node_type):
                    raise ValueError(
                        f"inconsistent type for node {node}: {existing} vs {node_type}"
                    )
                node_types[node] = node_type

            if any(token in event_name for token in ("READ", "RECV", "LOAD")):
                src, dst = dst, src
            pair_key = src * node_count + dst
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            sources.append(src)
            destinations.append(dst)
            edge_types.append(edge_name_to_id[event_name])

    missing_types = int((node_types < 0).sum())
    if missing_types:
        raise ValueError(f"edge list did not assign types to {missing_types} nodes")
    src_tensor = torch.from_numpy(
        np.frombuffer(sources, dtype=np.uint32).astype(np.int64)
    )
    dst_tensor = torch.from_numpy(
        np.frombuffer(destinations, dtype=np.uint32).astype(np.int64)
    )
    edge_type_tensor = torch.from_numpy(
        np.frombuffer(edge_types, dtype=np.uint8).astype(np.int64)
    )
    graph = dgl.graph((src_tensor, dst_tensor), num_nodes=node_count)
    graph.ndata["type"] = node_types
    graph.ndata["attr"] = F.one_hot(
        node_types, num_classes=len(CADETS_NODE_TYPES)
    ).float()
    graph.edata["type"] = edge_type_tensor
    graph.edata["attr"] = F.one_hot(
        edge_type_tensor, num_classes=len(CADETS_EDGE_TYPES)
    ).float()

    if graph.number_of_nodes() != 357174 or graph.number_of_edges() != 840300:
        raise AssertionError(
            "unexpected rebuilt CADETS size: "
            f"nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}, "
            f"rows={rows}"
        )
    temporary = graph_path.with_suffix(".rebuilt.pkl")
    with temporary.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, graph_path)
    print(
        f"rebuilt {graph_path}: nodes={graph.number_of_nodes()}, "
        f"edges={graph.number_of_edges()}, source_rows={rows}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--uuid-map", type=Path, default=DEFAULT_UUID_MAP)
    parser.add_argument("--edge-dir", type=Path, default=DEFAULT_EDGE_DIR)
    args = parser.parse_args()
    rebuild(args.graph, args.uuid_map, args.edge_dir)


if __name__ == "__main__":
    main()
