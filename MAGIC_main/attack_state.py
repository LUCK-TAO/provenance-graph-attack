"""Per-target graph modification state."""


class AttackState:
    """Track cumulative modifications made while attacking one target."""

    def __init__(
        self,
        target_node,
        initial_in_degree,
        initial_out_degree,
        action_dim=9,
    ):
        self.target_node = int(target_node)
        self.num_added_edges = 0
        self.num_incoming_edges = 0
        self.num_outgoing_edges = 0
        self.current_in_degree = int(initial_in_degree)
        self.current_out_degree = int(initial_out_degree)
        self.action_dim = int(action_dim)
        self.action_history = []
        # Kept under the legacy name because detector adapters may read it.
        self.edge_type_count = [0] * self.action_dim

    def record_add_edge(self, action_type, is_incoming=False):
        action_type = int(action_type)
        self.num_added_edges += 1
        if is_incoming:
            self.num_incoming_edges += 1
            self.current_in_degree += 1
        else:
            self.num_outgoing_edges += 1
            self.current_out_degree += 1
        if 0 <= action_type < self.action_dim:
            self.edge_type_count[action_type] += 1
        self.action_history.append(("add_edge", action_type))

    def get_summary(self):
        return (
            f"edges={self.num_added_edges}, "
            f"incoming={self.num_incoming_edges}, "
            f"outgoing={self.num_outgoing_edges}, "
            f"in_deg={self.current_in_degree}, "
            f"out_deg={self.current_out_degree}"
        )

    def to_dict(self):
        return {
            "target_node": self.target_node,
            "num_added_edges": self.num_added_edges,
            "num_incoming_edges": self.num_incoming_edges,
            "num_outgoing_edges": self.num_outgoing_edges,
            "current_in_degree": self.current_in_degree,
            "current_out_degree": self.current_out_degree,
            "action_history": self.action_history.copy(),
            "edge_type_count": self.edge_type_count.copy(),
            "action_dim": self.action_dim,
        }
