import unittest

import dgl
import torch

from MAGIC_main.action_schema import CADETS_ACTION_SCHEMA
from MAGIC_main.candidate_selector import CandidateSelector


class CountingCandidateSelector(CandidateSelector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.neighborhood_calls = 0

    def _get_khop_neighbors(self, graph, target_node, k=None):
        self.neighborhood_calls += 1
        return super()._get_khop_neighbors(graph, target_node, k=k)


class CandidateSelectorCacheTests(unittest.TestCase):
    @staticmethod
    def _graph():
        graph = dgl.graph(
            (
                torch.tensor([0, 0, 2, 3, 4, 5, 6]),
                torch.tensor([1, 2, 3, 4, 5, 6, 7]),
            ),
            num_nodes=12,
        )
        node_types = torch.tensor([0, 1, 0, 1, 0, 3, 0, 4, 0, 1, 0, 5])
        graph.ndata["attr"] = torch.nn.functional.one_hot(
            node_types, num_classes=6
        ).float()
        return graph

    @staticmethod
    def _selector():
        return CountingCandidateSelector(
            action_schema=CADETS_ACTION_SCHEMA,
            selection_mode="semantic_mixed",
            max_hop=2,
            local_ratio=0.5,
            seed=42,
        )

    def test_cache_preserves_pools_while_pending_edges_change(self):
        graph = self._graph()
        cached = self._selector()
        uncached = self._selector()
        pending = set()

        for _ in range(5):
            cached_pools = cached.legal_candidate_pools(
                graph, 0, top_k=6, pending_edges=pending
            )
            uncached.clear_cache()
            expected_pools = uncached.legal_candidate_pools(
                graph, 0, top_k=6, pending_edges=pending
            )
            self.assertEqual(cached_pools, expected_pools)

            action_id = min(cached_pools)
            candidate = cached_pools[action_id][0]
            operation = cached.resolve_operation(graph, 0, action_id, candidate)
            pending.add((operation["src"], operation["dst"]))

        self.assertEqual(cached.neighborhood_calls, 1)
        self.assertEqual(uncached.neighborhood_calls, 5)

    def test_edge_count_change_invalidates_neighborhood_cache(self):
        graph = self._graph()
        selector = self._selector()
        selector.legal_candidate_pools(graph, 0, top_k=6)
        modified = dgl.add_edges(graph, torch.tensor([8]), torch.tensor([9]))
        selector.legal_candidate_pools(modified, 0, top_k=6)
        self.assertEqual(selector.neighborhood_calls, 2)


if __name__ == "__main__":
    unittest.main()
