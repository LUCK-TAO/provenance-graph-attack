"""Dataset-specific, semantically legal graph attack actions.

An action is legal only when its ``(source node type, edge type,
destination node type)`` triple has been observed in the clean CADETS graph
construction used by MAGIC.  The action ID describes an event family; the
target node may occupy either endpoint when the schema allows it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, Iterable, Mapping, Optional, Tuple


NodePair = Tuple[int, int]


CADETS_NODE_TYPES: Dict[int, str] = {
    0: "SUBJECT_PROCESS",
    1: "FILE_OBJECT_FILE",
    2: "FILE_OBJECT_UNIX_SOCKET",
    3: "UnnamedPipeObject",
    4: "NetFlowObject",
    5: "FILE_OBJECT_DIR",
}

CADETS_EDGE_TYPES: Dict[int, str] = {
    0: "EVENT_CLOSE",
    1: "EVENT_CREATE_OBJECT",
    2: "EVENT_CONNECT",
    3: "EVENT_ACCEPT",
    4: "EVENT_WRITE",
    5: "EVENT_READ",
    6: "EVENT_OPEN",
    7: "EVENT_EXIT",
    8: "EVENT_FORK",
    9: "EVENT_EXECUTE",
    10: "EVENT_MMAP",
    11: "EVENT_MODIFY_PROCESS",
    12: "EVENT_LSEEK",
    13: "EVENT_CHANGE_PRINCIPAL",
    14: "EVENT_SENDTO",
    15: "EVENT_RECVMSG",
    16: "EVENT_RECVFROM",
    17: "EVENT_UNLINK",
    18: "EVENT_RENAME",
    19: "EVENT_FCNTL",
    20: "EVENT_MODIFY_FILE_ATTRIBUTES",
    21: "EVENT_LINK",
    22: "EVENT_SIGNAL",
    23: "EVENT_SENDMSG",
    24: "EVENT_FLOWS_TO",
    25: "EVENT_TRUNCATE",
    26: "EVENT_BIND",
}


@dataclass(frozen=True)
class ActionSpec:
    """One stable action category and all legal endpoint pairs for it."""

    action_id: int
    name: str
    edge_type: int
    edge_name: str
    valid_pairs: Tuple[NodePair, ...]

    def candidate_types(self, target_type: int) -> Tuple[int, ...]:
        candidates = set()
        for src_type, dst_type in self.valid_pairs:
            if src_type == target_type:
                candidates.add(dst_type)
            if dst_type == target_type:
                candidates.add(src_type)
        return tuple(sorted(candidates))

    def resolve_pair(
        self, target_type: int, candidate_type: int
    ) -> Optional[Tuple[bool, NodePair]]:
        """Return ``(target_is_source, (src_type, dst_type))`` if legal."""
        direct = (target_type, candidate_type)
        reverse = (candidate_type, target_type)
        if direct in self.valid_pairs:
            return True, direct
        if reverse in self.valid_pairs:
            return False, reverse
        return None

    def supports_target(self, target_type: int) -> bool:
        return bool(self.candidate_types(target_type))


class ActionSchema:
    """Immutable action vocabulary for one dataset representation."""

    def __init__(
        self,
        dataset: str,
        version: str,
        node_types: Dict[int, str],
        edge_types: Dict[int, str],
        actions: Iterable[ActionSpec],
    ):
        self.dataset = str(dataset)
        self.version = str(version)
        self.node_types = dict(node_types)
        self.edge_types = dict(edge_types)
        self.actions = tuple(sorted(actions, key=lambda item: item.action_id))
        expected_ids = tuple(range(len(self.actions)))
        actual_ids = tuple(action.action_id for action in self.actions)
        if actual_ids != expected_ids:
            raise ValueError(
                f"action IDs must be contiguous: expected={expected_ids}, "
                f"actual={actual_ids}"
            )
        for action in self.actions:
            if action.edge_type not in self.edge_types:
                raise ValueError(
                    f"{action.name} uses unknown edge type {action.edge_type}"
                )
            for src_type, dst_type in action.valid_pairs:
                if src_type not in self.node_types or dst_type not in self.node_types:
                    raise ValueError(
                        f"{action.name} uses unknown node pair "
                        f"({src_type}, {dst_type})"
                    )

    @property
    def action_dim(self) -> int:
        return len(self.actions)

    def get(self, action_id: int) -> ActionSpec:
        action_id = int(action_id)
        if not 0 <= action_id < self.action_dim:
            raise ValueError(f"invalid action ID {action_id}")
        return self.actions[action_id]

    def candidate_types(
        self, action_id: int, target_type: int
    ) -> Tuple[int, ...]:
        return self.get(action_id).candidate_types(int(target_type))

    def resolve(
        self, action_id: int, target_type: int, candidate_type: int
    ) -> Optional[Tuple[bool, NodePair]]:
        return self.get(action_id).resolve_pair(
            int(target_type), int(candidate_type)
        )

    def legal_action_ids(self, target_type: int) -> Tuple[int, ...]:
        return tuple(
            action.action_id
            for action in self.actions
            if action.supports_target(int(target_type))
        )


# These triples were verified against MAGIC_main/data/cadets/test0.pkl.
# They cover every CADETS node type while keeping a compact nine-category
# policy output.  Rare edge types remain outside the attack vocabulary.
CADETS_ACTION_SCHEMA = ActionSchema(
    dataset="cadets",
    version="cadets-semantic-v1",
    node_types=CADETS_NODE_TYPES,
    edge_types=CADETS_EDGE_TYPES,
    actions=(
        ActionSpec(
            0, "OPEN_OBJECT", 6, "EVENT_OPEN",
            ((0, 1), (0, 5)),
        ),
        ActionSpec(
            1, "CREATE_OBJECT", 1, "EVENT_CREATE_OBJECT",
            ((0, 2), (0, 3), (0, 5)),
        ),
        ActionSpec(
            2, "WRITE_OBJECT", 4, "EVENT_WRITE",
            ((0, 1), (0, 3), (0, 4)),
        ),
        ActionSpec(
            3, "READ_OBJECT", 5, "EVENT_READ",
            ((1, 0), (2, 0), (3, 0), (4, 0)),
        ),
        ActionSpec(
            4, "CLOSE_OBJECT", 0, "EVENT_CLOSE",
            ((0, 1), (0, 2), (0, 3), (0, 4)),
        ),
        ActionSpec(
            5, "CONNECT_NETWORK", 2, "EVENT_CONNECT",
            ((0, 2), (0, 4)),
        ),
        ActionSpec(
            6, "ACCEPT_NETWORK", 3, "EVENT_ACCEPT",
            ((0, 2), (0, 4)),
        ),
        ActionSpec(
            7, "EXECUTE_FILE", 9, "EVENT_EXECUTE",
            ((0, 1),),
        ),
        ActionSpec(
            8, "FORK_PROCESS", 8, "EVENT_FORK",
            ((0, 0),),
        ),
    ),
)


def build_observed_action_schema(
    graph,
    dataset: str,
    action_dim: int = 9,
    *,
    edge_type_names: Optional[Mapping[int, str]] = None,
    node_type_names: Optional[Mapping[int, str]] = None,
) -> ActionSchema:
    """Build a compact legal schema from the supplied dataset graph.

    Slots first cover every node type observed in the graph, then remaining
    slots are filled by edge frequency. Every endpoint pair stored in an
    action was observed in the graph, so changing node-feature dimensions
    cannot silently reuse CADETS semantics. This is the fallback for THEIA,
    TRACE, and future datasets until a named semantic profile is added.
    """
    import torch
    from collections import Counter, defaultdict

    node_dim = int(graph.ndata["attr"].shape[1])
    edge_dim = int(graph.edata["attr"].shape[1])
    node_type_values = torch.argmax(
        graph.ndata["attr"], dim=1
    ).cpu().tolist()
    edge_type_values = torch.argmax(
        graph.edata["attr"], dim=1
    ).cpu().tolist()
    src_nodes, dst_nodes = graph.edges()
    edge_counts = Counter(int(value) for value in edge_type_values)
    pair_counts = defaultdict(Counter)
    for src, dst, edge_type in zip(
        src_nodes.cpu().tolist(),
        dst_nodes.cpu().tolist(),
        edge_type_values,
    ):
        pair_counts[int(edge_type)][
            (
                int(node_type_values[int(src)]),
                int(node_type_values[int(dst)]),
            )
        ] += 1

    frequency_ranked = [
        edge_type
        for edge_type, _ in sorted(
            edge_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    edge_endpoint_types = {
        edge_type: {
            node_type
            for pair in pair_counter
            for node_type in pair
        }
        for edge_type, pair_counter in pair_counts.items()
    }
    observed_node_types = set(int(value) for value in node_type_values)
    uncovered_types = set(observed_node_types)
    ranked_edge_types = []
    # Coverage-first greedy set cover: no observed node type should become
    # unattackable merely because its relations are rare.
    while uncovered_types and len(ranked_edge_types) < int(action_dim):
        available = [
            edge_type
            for edge_type in frequency_ranked
            if edge_type not in ranked_edge_types
            and edge_endpoint_types.get(edge_type, set()) & uncovered_types
        ]
        if not available:
            break
        best = max(
            available,
            key=lambda edge_type: (
                len(edge_endpoint_types[edge_type] & uncovered_types),
                edge_counts[edge_type],
                -edge_type,
            ),
        )
        ranked_edge_types.append(best)
        uncovered_types -= edge_endpoint_types[best]
    for edge_type in frequency_ranked:
        if len(ranked_edge_types) >= int(action_dim):
            break
        if edge_type not in ranked_edge_types:
            ranked_edge_types.append(edge_type)
    semantic_edges = {
        int(key): str(value) for key, value in (edge_type_names or {}).items()
    }
    semantic_nodes = {
        int(key): str(value) for key, value in (node_type_names or {}).items()
    }
    actions = []
    for action_id, edge_type in enumerate(ranked_edge_types):
        pairs = tuple(
            pair
            for pair, _ in sorted(
                pair_counts[edge_type].items(),
                key=lambda item: (-item[1], item[0]),
            )
        )
        event_name = semantic_edges.get(
            int(edge_type), f"EDGE_TYPE_{edge_type}"
        )
        display_name = event_name.removeprefix("EVENT_")
        actions.append(
            ActionSpec(
                action_id=action_id,
                name=f"{display_name}_SLOT_{action_id + 1}",
                edge_type=int(edge_type),
                edge_name=event_name,
                valid_pairs=pairs,
            )
        )
    # Keep the PPO output width fixed even for a dataset with fewer than nine
    # observed edge families. Empty actions are always masked as illegal.
    fallback_edge_type = ranked_edge_types[0] if ranked_edge_types else 0
    while len(actions) < int(action_dim):
        action_id = len(actions)
        actions.append(
            ActionSpec(
                action_id=action_id,
                name=f"UNAVAILABLE_ACTION_{action_id}",
                edge_type=int(fallback_edge_type),
                edge_name=f"EDGE_TYPE_{fallback_edge_type}",
                valid_pairs=(),
            )
        )

    signature_payload = repr(
        tuple(
            (action.edge_type, action.edge_name, action.valid_pairs)
            for action in actions
        )
    ).encode("utf-8")
    signature = hashlib.sha256(signature_payload).hexdigest()[:12]
    return ActionSchema(
        dataset=str(dataset).lower(),
        version=(
            f"{str(dataset).lower()}-observed-cover{int(action_dim)}-"
            f"n{node_dim}-e{edge_dim}-{signature}-v2"
        ),
        node_types={
            node_type: semantic_nodes.get(node_type, f"NODE_TYPE_{node_type}")
            for node_type in range(node_dim)
        },
        edge_types={
            edge_type: semantic_edges.get(edge_type, f"EDGE_TYPE_{edge_type}")
            for edge_type in range(edge_dim)
        },
        actions=actions,
    )


def get_action_schema(
    dataset: str, *, allow_observed_fallback: bool = False
) -> Optional[ActionSchema]:
    dataset = str(dataset).lower()
    if dataset == "cadets":
        return CADETS_ACTION_SCHEMA
    if allow_observed_fallback:
        return None
    raise ValueError(
        f"No semantically validated action schema exists for dataset "
        f"{dataset!r}. Build an observed ActionSchema from its graph first."
    )
