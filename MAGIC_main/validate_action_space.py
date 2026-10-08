"""Validate a dataset action schema against a serialized MAGIC graph."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import torch

try:
    from .action_schema import (
        build_observed_action_schema,
        get_action_schema,
    )
    from .candidate_selector import CandidateSelector
except ImportError:
    from action_schema import build_observed_action_schema, get_action_schema
    from candidate_selector import CandidateSelector


ROOT = Path(__file__).resolve().parent


def validate(dataset: str, test_index: int = 0):
    graph_path = ROOT / "data" / dataset / f"test{test_index}.pkl"
    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)

    schema = get_action_schema(dataset, allow_observed_fallback=True)
    if schema is None:
        schema = build_observed_action_schema(
            graph, dataset=dataset, action_dim=9
        )

    node_types = torch.argmax(graph.ndata["attr"], dim=1).cpu().tolist()
    edge_types = torch.argmax(graph.edata["attr"], dim=1).cpu().tolist()
    src_nodes, dst_nodes = graph.edges()
    triples = [
        (
            int(node_types[int(src)]),
            int(edge_type),
            int(node_types[int(dst)]),
        )
        for src, dst, edge_type in zip(
            src_nodes.cpu().tolist(),
            dst_nodes.cpu().tolist(),
            edge_types,
        )
    ]
    observed = set(triples)
    allowed = {
        (src_type, action.edge_type, dst_type)
        for action in schema.actions
        for src_type, dst_type in action.valid_pairs
    }
    missing = sorted(
        (
            action.name,
            src_type,
            action.edge_type,
            dst_type,
        )
        for action in schema.actions
        for src_type, dst_type in action.valid_pairs
        if (src_type, action.edge_type, dst_type) not in observed
    )
    if missing:
        raise AssertionError(f"schema triples missing from graph: {missing}")

    selector = CandidateSelector(action_schema=schema)
    selector.build_type_index(graph)
    representative_pools = {}
    for node_type, node_ids in sorted(selector.type_index.items()):
        target = int(node_ids[0])
        pools = selector.legal_candidate_pools(
            graph, target, top_k=5
        )
        representative_pools[str(node_type)] = {
            str(action_id): len(candidates)
            for action_id, candidates in sorted(pools.items())
        }
        for action_id, candidates in pools.items():
            for candidate in candidates:
                operation = selector.resolve_operation(
                    graph, target, action_id, candidate
                )
                triple = (
                    operation["src_type"],
                    operation["edge_type"],
                    operation["dst_type"],
                )
                if triple not in allowed:
                    raise AssertionError(f"illegal candidate triple {triple}")

    covered = sum(triple in allowed for triple in triples)
    return {
        "dataset": dataset,
        "test_index": int(test_index),
        "graph": {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
            "node_feature_dim": int(graph.ndata["attr"].shape[1]),
            "edge_feature_dim": int(graph.edata["attr"].shape[1]),
            "node_type_counts": dict(
                sorted(Counter(int(value) for value in node_types).items())
            ),
        },
        "schema": {
            "version": schema.version,
            "action_dim": schema.action_dim,
            "covered_edges": covered,
            "edge_coverage": covered / max(len(triples), 1),
            "actions": [
                {
                    "id": action.action_id,
                    "name": action.name,
                    "edge_type": action.edge_type,
                    "edge_name": action.edge_name,
                    "valid_pairs": list(action.valid_pairs),
                }
                for action in schema.actions
            ],
        },
        "representative_legal_pool_sizes": representative_pools,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("cadets", "theia", "trace"), required=True
    )
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.dataset, args.test_index)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
