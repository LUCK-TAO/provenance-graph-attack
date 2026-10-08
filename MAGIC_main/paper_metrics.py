"""Detector-independent definitions of the four primary paper metrics."""

from __future__ import annotations


def compute_primary_metrics(
    *,
    escaped_initial_tp: int,
    attacked_nodes: int,
    attacked_initial_tp: int,
    direct_escaped_initial_tp: int,
    delta_edges: int,
    delta_nodes: int,
    initial_edges: int,
    initial_nodes: int,
):
    """Compute the four primary metrics from explicit auditable counts.

    Undefined ratio metrics return ``None`` instead of silently becoming zero.
    """
    escaped_initial_tp = int(escaped_initial_tp)
    attacked_nodes = int(attacked_nodes)
    attacked_initial_tp = int(attacked_initial_tp)
    direct_escaped_initial_tp = int(direct_escaped_initial_tp)
    delta_edges = abs(int(delta_edges))
    delta_nodes = abs(int(delta_nodes))
    initial_edges = int(initial_edges)
    initial_nodes = int(initial_nodes)

    total_edits = delta_edges + delta_nodes
    initial_graph_size = initial_edges + initial_nodes
    return {
        "direct_tp_asr": (
            direct_escaped_initial_tp / attacked_initial_tp
            if attacked_initial_tp else None
        ),
        "evasion_amplification": (
            escaped_initial_tp / attacked_nodes
            if attacked_nodes else None
        ),
        "graph_modification_rate": (
            total_edits / initial_graph_size if initial_graph_size else None
        ),
        "average_attack_budget": (
            total_edits / escaped_initial_tp if escaped_initial_tp else None
        ),
    }
