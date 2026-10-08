"""Align the independently reproduced STGAN state to the attack graph IDs.

The standalone native reproducer numbers nodes by first appearance in the
released event stream.  The common attack framework numbers the same UUIDs by
the MAGIC/DGL test graph map.  This converter performs only that bijective UUID
permutation and adds the semantic name tables required by the attack action
schema; model features, labels, timestamps, and native topology all come from
the independently reproduced official state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MAGIC_main.action_schema import CADETS_EDGE_TYPES, CADETS_NODE_TYPES

DEFAULT_OFFICIAL = ROOT / "STGAN_master" / "data" / "cadets" / "cadets_stgan_official_test.npz"
DEFAULT_UUID_MAP = ROOT / "MAGIC_main" / "data" / "cadets" / "uuid_to_index_test0.json"
DEFAULT_TEMPLATE = ROOT / "STGAN_master" / "data" / "cadets" / "cadets_stgan_native_test0.npz"
DEFAULT_OUTPUT = ROOT / "STGAN_master" / "data" / "cadets" / "cadets_stgan_official_attack_test0.npz"


def _optional_array(payload, name: str, dtype):
    if name not in payload.files:
        return np.asarray([], dtype=dtype)
    return np.asarray(payload[name], dtype=dtype)


def build(official: Path, uuid_map_path: Path, template: Path, output: Path, force: bool) -> Path:
    if output.exists() and not force:
        raise FileExistsError(f"output exists: {output}; pass --force to replace it")

    uuid_to_magic = json.loads(uuid_map_path.read_text(encoding="utf-8"))
    with np.load(official, allow_pickle=False) as source:
        uuids = np.asarray(source["uuids"], dtype=np.str_)
        labels = np.asarray(source["labels"], dtype=np.int64)
        embeddings = np.asarray(source["embeddings"], dtype=np.float32)
        token_counts = np.asarray(source["token_counts"], dtype=np.int64)
        edge_index = np.asarray(source["edge_index"], dtype=np.int64)
        edge_time = np.asarray(source["edge_time"], dtype=np.float32)

    node_count = len(uuids)
    official_to_magic = np.asarray(
        [uuid_to_magic.get(str(node_uuid), -1) for node_uuid in uuids],
        dtype=np.int64,
    )
    if np.any(official_to_magic < 0):
        raise ValueError(f"UUID map misses {int(np.sum(official_to_magic < 0))} official nodes")
    if len(np.unique(official_to_magic)) != node_count:
        raise ValueError("official UUID to attack-graph index mapping is not bijective")
    if official_to_magic.min() != 0 or official_to_magic.max() != node_count - 1:
        raise ValueError("attack-graph indices are not a dense zero-based range")

    magic_embeddings = np.empty_like(embeddings)
    magic_labels = np.empty_like(labels)
    magic_token_counts = np.empty_like(token_counts)
    magic_uuids = np.empty_like(uuids)
    magic_embeddings[official_to_magic] = embeddings
    magic_labels[official_to_magic] = labels
    magic_token_counts[official_to_magic] = token_counts
    magic_uuids[official_to_magic] = uuids
    magic_edge_index = official_to_magic[edge_index]

    if template.is_file():
        with np.load(template, allow_pickle=False) as names:
            edge_type_ids = _optional_array(names, "magic_edge_type_ids", np.int64)
            edge_type_names = _optional_array(names, "magic_edge_type_names", np.str_)
            node_type_ids = _optional_array(names, "node_type_ids", np.int64)
            node_type_names = _optional_array(names, "node_type_names", np.str_)
    else:
        # The template contributes names only; detector features and topology
        # still come exclusively from the independently reproduced state.
        # Falling back to the canonical CADETS schema lets cleaned workspaces
        # rebuild the formal attack state without regenerating a second large
        # semantic sidecar merely to recover these static labels.
        edge_items = sorted(CADETS_EDGE_TYPES.items())
        node_items = sorted(CADETS_NODE_TYPES.items())
        edge_type_ids = np.asarray([key for key, _ in edge_items], dtype=np.int64)
        edge_type_names = np.asarray([value for _, value in edge_items], dtype=np.str_)
        node_type_ids = np.asarray([key for key, _ in node_items], dtype=np.int64)
        node_type_names = np.asarray([value for _, value in node_items], dtype=np.str_)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        version=np.asarray(2, dtype=np.int64),
        dataset=np.asarray("cadets"),
        test_index=np.asarray(0, dtype=np.int64),
        global_node_offset=np.asarray(0, dtype=np.int64),
        original_node_count=np.asarray(node_count, dtype=np.int64),
        base_embeddings=magic_embeddings,
        token_counts=magic_token_counts,
        model_labels=magic_labels,
        native_edge_index=magic_edge_index,
        native_edge_time=edge_time,
        magic_edge_type_ids=edge_type_ids,
        magic_edge_type_names=edge_type_names,
        node_type_ids=node_type_ids,
        node_type_names=node_type_names,
        uuids=magic_uuids,
        official_source=np.asarray(str(official.resolve())),
        uuid_map_source=np.asarray(str(uuid_map_path.resolve())),
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "nodes": node_count,
                "edges": int(magic_edge_index.shape[1]),
                "uuid_mapping": "bijective",
                "feature_source": str(official.resolve()),
            },
            indent=2,
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--uuid-map", type=Path, default=DEFAULT_UUID_MAP)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.official, args.uuid_map, args.template, args.output, args.force)


if __name__ == "__main__":
    main()
