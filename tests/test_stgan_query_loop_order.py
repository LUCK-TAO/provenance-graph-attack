import unittest
import tempfile
from pathlib import Path
import sys

import dgl
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from Detector.STGAN import STGANDetector
from Detector.STGAN_State import STGANSemanticState
from STGAN_master.stgan_theia import _align_to_audited_flash_state

ROOT = Path(__file__).resolve().parents[1]
MAGIC_ROOT = ROOT / "MAGIC_main"
if str(MAGIC_ROOT) not in sys.path:
    sys.path.insert(0, str(MAGIC_ROOT))
from MAGIC_main.attack_directed_topk import DirectedTopKAttackAgent


class DummyGAT(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("weight", torch.randn(6, 30, generator=generator))

    def forward(self, x, _edge_index):
        return x @ self.weight


class DummyTGN(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("weight", torch.randn(60, 32, generator=generator))

    def forward(self, x, _edge_index, edge_time):
        return x @ self.weight + edge_time.mean()


class DummyAttention(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("weight", torch.randn(32, 6, generator=generator))

    def forward(self, x):
        return x.mean(dim=1) @ self.weight


class STGANLoopOrderTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        node_count = 8
        edge_index = np.asarray(
            [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]],
            dtype=np.int64,
        )
        state = STGANSemanticState(
            original_node_count=node_count,
            base_embeddings=np.arange(node_count * 30, dtype=np.float32).reshape(
                node_count, 30
            ) / 100.0,
            token_counts=np.ones(node_count, dtype=np.int64),
            model_labels=np.asarray([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.int64),
            native_edge_index=edge_index,
            native_edge_time=np.arange(len(edge_index[0]), dtype=np.float32),
            dataset="cadets",
            test_index=0,
        )
        graph = dgl.graph(
            (torch.from_numpy(edge_index[0]), torch.from_numpy(edge_index[1])),
            num_nodes=node_count,
        )
        models = [
            (DummyGAT(10), DummyTGN(20), DummyAttention(30)),
            (DummyGAT(11), DummyTGN(21), DummyAttention(31)),
            (DummyGAT(12), DummyTGN(22), DummyAttention(32)),
        ]
        for triplet in models:
            for model in triplet:
                model.eval()
        detector = STGANDetector.__new__(STGANDetector)
        detector.semantic_state = state
        detector.dataset_name = "cadets"
        detector.test_index = 0
        detector.device = torch.device("cpu")
        detector.snapshot_models = models
        detector.snapshot_paths = [(None, None, None)] * len(models)
        detector.NATIVE_BATCH_SIZE = 3
        return detector, graph, models

    @staticmethod
    def _snapshot_outer_reference(detector, models):
        state = detector.semantic_state
        labels = np.asarray(state.model_labels, dtype=np.int64)
        one_hot = np.eye(6, dtype=np.float32)[labels]
        pyg_graph = Data(
            x=torch.from_numpy(one_hot),
            y=torch.from_numpy(labels),
            edge_index=torch.as_tensor(state.edge_index(), dtype=torch.long),
            edge_time=torch.as_tensor(state.edge_time(), dtype=torch.float32),
        )
        pyg_graph.n_id = torch.arange(state.original_node_count, dtype=torch.long)
        embeddings = torch.as_tensor(state.base_embeddings, dtype=torch.float32)
        scores = np.full(state.original_node_count, np.inf, dtype=np.float32)
        persistent = np.ones(state.original_node_count, dtype=bool)
        loader = NeighborLoader(
            pyg_graph, num_neighbors=[-1, -1], batch_size=3, shuffle=False
        )
        with torch.inference_mode():
            for gat, tgn, attention in models:
                for subgraph in loader:
                    node_ids = subgraph.n_id.numpy().astype(np.int64, copy=False)
                    output = gat(subgraph.x, subgraph.edge_index)
                    output = torch.cat(
                        (output, embeddings.index_select(0, subgraph.n_id)), dim=1
                    )
                    output = tgn(output, subgraph.edge_index, subgraph.edge_time)
                    logits = attention(output.unsqueeze(1))
                    margin = detector._margin(logits, subgraph.y).numpy()
                    np.minimum.at(scores, node_ids, margin)
                    correct = logits.argmax(dim=1).eq(subgraph.y).numpy()
                    persistent[node_ids[correct]] = False
        return scores, np.flatnonzero(persistent).astype(np.int64)

    def test_batch_outer_order_matches_snapshot_outer_order(self):
        detector, graph, models = self._fixture()
        expected_scores, expected_alerts = self._snapshot_outer_reference(
            detector, models
        )
        actual_scores, actual_alerts = detector._score_graph(graph)
        np.testing.assert_array_equal(actual_scores, expected_scores)
        np.testing.assert_array_equal(actual_alerts, expected_alerts)

    def test_theia_edit_updates_embedding_without_positional_encoding(self):
        state = STGANSemanticState(
            original_node_count=2,
            base_embeddings=np.zeros((2, 30), dtype=np.float32),
            token_counts=np.zeros(2, dtype=np.int64),
            model_labels=np.asarray([0, 3], dtype=np.int64),
            native_edge_index=np.asarray([[0], [1]], dtype=np.int64),
            native_edge_time=np.asarray([1.0], dtype=np.float32),
            dataset="theia",
            test_index=0,
            use_positional_encoding=False,
        )
        vector = np.arange(30, dtype=np.float32)
        state.add_edge(0, 1, "EVENT_CONNECT", word_vector=vector, timestamp=2.0)
        np.testing.assert_array_equal(state.base_embeddings[0], vector)
        np.testing.assert_array_equal(state.base_embeddings[1], vector)
        self.assertEqual(state.token_counts.tolist(), [1, 1])
        self.assertEqual(state.edge_index().shape, (2, 2))
        self.assertEqual(state.edge_time().tolist(), [1.0, 2.0])

    def test_theia_release_configuration_uses_twenty_snapshots(self):
        self.assertEqual(STGANDetector.CLASS_COUNTS["theia"], 6)
        self.assertEqual(STGANDetector.NATIVE_SNAPSHOT_COUNTS["theia"], 20)
        self.assertEqual(STGANDetector.TEST_INDICES["theia"], (0,))

    def test_theia_raw_ids_are_permuted_to_audited_magic_ids(self):
        # raw 0->audited 2, raw 1->audited 0, raw 2->audited 1
        own_edges = np.asarray([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
        mapping = np.asarray([2, 0, 1], dtype=np.int64)
        flash_edges = mapping[own_edges]
        built = {
            "native_edge_index": own_edges,
            "model_labels": np.asarray([3, 0, 1], dtype=np.int64),
            "base_embeddings": np.arange(90, dtype=np.float32).reshape(3, 30),
            "token_counts": np.asarray([5, 6, 7], dtype=np.int64),
            "node_map": {"a": 0, "b": 1, "c": 2},
        }
        remapped_labels = np.empty(3, dtype=np.int64)
        remapped_labels[mapping] = built["model_labels"]
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "flash.npz"
            np.savez_compressed(
                state_path,
                native_edge_index=flash_edges,
                model_labels=remapped_labels,
                magic_edge_type_ids=np.asarray([0]),
                magic_edge_type_names=np.asarray(["EVENT_OPEN"]),
                node_type_ids=np.asarray([0]),
                node_type_names=np.asarray(["SUBJECT_PROCESS"]),
            )
            aligned = _align_to_audited_flash_state(built, state_path)
        np.testing.assert_array_equal(aligned["native_edge_index"], flash_edges)
        self.assertEqual(aligned["node_map"], {"a": 2, "b": 0, "c": 1})
        self.assertEqual(aligned["node_permutation_moved"], 3)

    def test_initial_tp_margin_never_selects_false_positive_alerts(self):
        agent = DirectedTopKAttackAgent.__new__(DirectedTopKAttackAgent)
        agent._initial_tp_cycle_order = [2, 4, 6]
        agent.top_k = 2
        agent.detector = type("Detector", (), {"best_threshold": 0.0})()
        scores = {
            1: 0.0001,  # Closer FP: must never enter the TP-only target pool.
            2: 0.4,
            4: 0.1,
            6: -0.2,  # Escaped initial TP: must not be selected again.
        }
        self.assertEqual(agent._initial_tp_margin_targets(scores), [4, 2])

    def test_theia_modification_uses_observed_edge_event_name(self):
        state = STGANSemanticState(
            original_node_count=2,
            base_embeddings=np.zeros((2, 30), dtype=np.float32),
            token_counts=np.zeros(2, dtype=np.int64),
            model_labels=np.asarray([0, 3], dtype=np.int64),
            native_edge_index=np.asarray([[0], [1]], dtype=np.int64),
            native_edge_time=np.asarray([1.0], dtype=np.float32),
            magic_edge_type_names={11: "EVENT_CONNECT"},
            dataset="theia",
            test_index=0,
            use_positional_encoding=False,
        )
        detector = STGANDetector.__new__(STGANDetector)
        detector.dataset_name = "theia"
        detector.semantic_state = state
        detector.word_vectors = {}
        detector._last_attack_timestamp = 1.0
        detector.record_attack_modification(
            kind="edge",
            action_type=1,
            edge_type=11,
            src=0,
            dst=1,
            graph=None,
        )
        np.testing.assert_array_equal(
            state.edge_index(), np.asarray([[0, 0], [1, 1]], dtype=np.int64)
        )
        self.assertEqual(state.edge_time().tolist(), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
