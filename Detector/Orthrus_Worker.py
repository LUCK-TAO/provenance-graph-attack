"""Isolated Orthrus inference worker used by the unified attack framework.

Orthrus' released source uses top-level imports named ``config``, ``model``
and ``factory``. Running it in a spawned worker prevents those modules from
colliding with the existing attack framework. The worker keeps the official
model resident and restores the pre-window temporal state before each query.
"""

from __future__ import annotations

import os
import random
import sys
import traceback
import types
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORTHRUS_ROOT = ROOT / "Orthrus_official"
ORTHRUS_SRC = ORTHRUS_ROOT / "src"


def _prepare_imports() -> None:
    os.environ.setdefault("NLTK_DATA", str(ORTHRUS_ROOT / "nltk_data"))
    (ORTHRUS_ROOT / "nltk_data").mkdir(parents=True, exist_ok=True)
    for name in (
        "config", "model", "factory", "data_utils", "encoders", "decoders",
        "provnet_utils", "temporal",
    ):
        sys.modules.pop(name, None)
    # The attack worker reads prebuilt artifacts and never opens PostgreSQL.
    # The STGAN environment intentionally has no psycopg2, while Orthrus'
    # shared utility module imports it eagerly. Supply a non-operational stub
    # instead of downloading an unused package.
    try:
        import psycopg2  # noqa: F401
    except ModuleNotFoundError:
        psycopg2_stub = types.ModuleType("psycopg2")
        extras_stub = types.ModuleType("psycopg2.extras")
        psycopg2_stub.extras = extras_stub
        psycopg2_stub.connect = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("PostgreSQL is unavailable inside the Orthrus attack worker")
        )
        sys.modules["psycopg2"] = psycopg2_stub
        sys.modules["psycopg2.extras"] = extras_stub
    try:
        import xxhash  # noqa: F401
    except ModuleNotFoundError:
        xxhash_stub = types.ModuleType("xxhash")

        class _XXH64:
            def __init__(self):
                self._hash = hashlib.blake2b(digest_size=8)

            def update(self, value):
                self._hash.update(value)

            def intdigest(self):
                return int.from_bytes(self._hash.digest(), "big")

        xxhash_stub.xxh64 = _XXH64
        sys.modules["xxhash"] = xxhash_stub
    try:
        import nltk  # noqa: F401
    except ModuleNotFoundError:
        nltk_stub = types.ModuleType("nltk")
        tokenize_stub = types.ModuleType("nltk.tokenize")
        tokenize_stub.word_tokenize = lambda text: str(text).split()
        nltk_stub.tokenize = tokenize_stub
        nltk_stub.download = lambda *_args, **_kwargs: True
        sys.modules["nltk"] = nltk_stub
        sys.modules["nltk.tokenize"] = tokenize_stub

    # Inference needs only the released relation dictionaries from config.py.
    # Building a small module here avoids adding the unused yacs dependency to
    # the attack environment and keeps all runtime artifacts on D:.
    config_stub = types.ModuleType("config")
    relations = [
        "EVENT_CONNECT", "EVENT_EXECUTE", "EVENT_OPEN", "EVENT_READ",
        "EVENT_RECVFROM", "EVENT_RECVMSG", "EVENT_SENDMSG", "EVENT_SENDTO",
        "EVENT_WRITE", "EVENT_CLONE",
    ]
    config_stub.rel2id = {
        key: value
        for index, name in enumerate(relations, start=1)
        for key, value in ((index, name), (name, index))
    }
    node_types = ["subject", "file", "netflow"]
    config_stub.ntype2id = {
        key: value
        for index, name in enumerate(node_types, start=1)
        for key, value in ((index, name), (name, index))
    }
    sys.modules["config"] = config_stub
    try:
        import torch_geometric.utils._scatter  # noqa: F401
    except ModuleNotFoundError:
        from torch_scatter import scatter_max

        pyg_scatter_stub = types.ModuleType("torch_geometric.utils._scatter")

        def scatter_argmax(src, index, dim=0, dim_size=None):
            _values, argmax = scatter_max(
                src, index, dim=dim, dim_size=dim_size
            )
            return argmax

        pyg_scatter_stub.scatter_argmax = scatter_argmax
        sys.modules["torch_geometric.utils._scatter"] = pyg_scatter_stub
    source = str(ORTHRUS_SRC)
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)
    os.chdir(ORTHRUS_ROOT)


def _namespace(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _cfg(device: str, dataset_name: str):
    profiles = {
        "cadets": {
            "official": "CADETS_E3", "seed": 4, "dropout": 0.25,
            "node_hid_dim": 256, "node_out_dim": 256, "lr": 0.001,
        },
        "theia": {
            "official": "THEIA_E3", "seed": 2, "dropout": 0.1,
            "node_hid_dim": 128, "node_out_dim": 64, "lr": 0.00001,
        },
    }
    if dataset_name not in profiles:
        raise ValueError(f"unsupported Orthrus worker dataset: {dataset_name}")
    profile = profiles[dataset_name]
    official = profile["official"]
    embed_candidates = list(ORTHRUS_ROOT.glob(
        f"artifacts/edge_featurization/{official}/embed_edges/*/edge_embeds"
    ))
    model_candidates = list(ORTHRUS_ROOT.glob(
        f"artifacts/detection/gnn_training/*/{official}/trained_models"
    ))
    embed_candidates = [path for path in embed_candidates if (path / "test").is_dir()]
    model_candidates = [path for path in model_candidates if (path / "model_epoch_1").is_dir()]
    if len(embed_candidates) != 1 or len(model_candidates) != 1:
        raise RuntimeError(
            f"Orthrus artifact discovery failed: embeds={embed_candidates}, "
            f"models={model_candidates}"
        )
    encoder = _namespace(
        use_node_type_in_node_feats=True,
        neighbor_sampling=[],
        edge_features="edge_type",
        use_node_feats_in_gnn=True,
        neighbor_size=20,
        temporal_dim=100,
        batch_size=1024,
        graph_attention=_namespace(
            dropout=profile["dropout"],
            activation="relu",
            num_heads=8,
        ),
    )
    decoder = _namespace(
        used_methods="predict_edge_type",
        predict_edge_type=_namespace(
            used_method="custom",
            custom=_namespace(dropout=0.0, num_layers=2, activation="relu"),
        ),
    )
    return _namespace(
        _test_mode=False,
        _use_cpu=(str(device).lower() == "cpu"),
        _from_weights=True,
        _from_weights_path=str(ORTHRUS_ROOT / "weights"),
        _seed=profile["seed"],
        dataset=_namespace(name=official, num_node_types=3, num_edge_types=10),
        edge_featurization=_namespace(
            embed_nodes=_namespace(emb_dim=128, used_method="feature_word2vec"),
            embed_edges=_namespace(_edge_embeds_dir=str(embed_candidates[0])),
        ),
        detection=_namespace(
            gnn_training=_namespace(
                _trained_models_dir=str(model_candidates[0]),
                node_hid_dim=profile["node_hid_dim"],
                node_out_dim=profile["node_out_dim"],
                lr=profile["lr"],
                weight_decay=0.00001,
                encoder=encoder,
                decoder=decoder,
            )
        ),
    )


class _Runtime:
    def __init__(self, state_path: str, device: str) -> None:
        _prepare_imports()
        import torch
        from sklearn.cluster import KMeans
        from torch_geometric.data import Data, TemporalData
        from data_utils import extract_msg_from_data, load_model
        from factory import batch_loader_factory, build_model

        self.torch = torch
        self.KMeans = KMeans
        self.Data = Data
        self.TemporalData = TemporalData
        self.batch_loader_factory = batch_loader_factory

        state = np.load(state_path, allow_pickle=False)
        self.dataset_name = str(state["dataset"][0])
        self.alert_mode = (
            str(state["alert_mode"][0])
            if "alert_mode" in state.files else "ano"
        )
        self.native_threshold = (
            float(state["native_threshold"][0])
            if "native_threshold" in state.files else float("nan")
        )
        self.global_node_ids = np.asarray(state["global_node_ids"], dtype=np.int64)
        self.global_to_local = {
            int(global_id): local_id
            for local_id, global_id in enumerate(self.global_node_ids.tolist())
        }
        self.selected_windows = [int(value) for value in state["selected_windows"]]
        self.outside_max = np.asarray(
            state["local_outside_max_scores"], dtype=np.float64
        )
        self.baseline_result_ids = np.asarray(
            state["baseline_result_ids"], dtype=np.int64
        )
        self.baseline_result_scores = np.asarray(
            state["baseline_result_scores"], dtype=np.float64
        )
        self.result_positions = {
            int(node): index for index, node in enumerate(self.baseline_result_ids.tolist())
        }

        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
        self.device = requested
        seed = 4 if self.dataset_name == "cadets" else 2
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.cfg = _cfg(str(self.device), self.dataset_name)
        embed_root = Path(self.cfg.edge_featurization.embed_edges._edge_embeds_dir)

        def split_files(split):
            return sorted(path for path in (embed_root / split).iterdir() if path.is_file())

        files_by_split = {
            split: split_files(split) for split in ("train", "val", "test")
        }

        def load_processed(path):
            raw = torch.load(path, map_location="cpu")
            return extract_msg_from_data([raw], self.cfg)[0]

        # Orthrus' encoder only consults full_data.edge_type. Building the
        # released helper's all_msg tensor would allocate another ~5.9 GB that
        # is never read by this configuration, so stream only the native edge
        # type slice while retaining identical event ordering.
        max_node = 0
        cache_path = (
            ORTHRUS_ROOT / "artifacts" / "streaming_cache"
            / self.cfg.dataset.name / "all_edge_types.npy"
        )
        if cache_path.is_file():
            self.edge_type_cache = np.load(cache_path, mmap_mode="r+")
            self.full_data = Data(edge_type=torch.from_numpy(self.edge_type_cache))
            for split in ("train", "val", "test"):
                for path in files_by_split[split]:
                    raw = torch.load(path, map_location="cpu")
                    max_node = max(
                        max_node,
                        int(torch.maximum(raw.src.max(), raw.dst.max()).item()),
                    )
                    del raw
        else:
            all_edge_types = []
            for split in ("train", "val", "test"):
                for path in files_by_split[split]:
                    raw = torch.load(path, map_location="cpu")
                    all_edge_types.append(raw.msg[:, 131:141].clone())
                    max_node = max(
                        max_node,
                        int(torch.maximum(raw.src.max(), raw.dst.max()).item()),
                    )
                    del raw
            self.full_data = Data(edge_type=torch.cat(all_edge_types, dim=0))
            del all_edge_types
        max_node_num = max_node + 1
        sample = load_processed(files_by_split["test"][0])
        self.test_files = files_by_split["test"]
        self.query_windows = list(range(
            min(self.selected_windows), max(self.selected_windows) + 1
        ))
        self.load_processed = load_processed
        self.model = build_model(sample, self.device, self.cfg, max_node_num)
        model_dir = Path(self.cfg.detection.gnn_training._trained_models_dir) / "model_epoch_1"
        self.model = load_model(self.model, str(model_dir))
        weight_path = Path(self.cfg._from_weights_path) / f"{self.cfg.dataset.name}.pkl"
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

        # Reproduce the native temporal context up to (but excluding) the
        # earliest selected window. The checkpoint contains training history.
        with torch.inference_mode():
            replay_files = (
                files_by_split["val"]
                + files_by_split["test"][: self.selected_windows[0]]
            )
            for path in replay_files:
                graph = load_processed(path)
                graph = graph.to(self.device)
                for batch in self.batch_loader_factory(
                    self.cfg, graph, self.model.graph_reindexer
                ):
                    self.model(batch, self.full_data, inference=True)
                graph.to("cpu")

        # Every query replays all windows from the first selected target to the
        # last one. Snapshot every node touched in that interval so repeated
        # black-box queries cannot leak temporal state into one another.
        restore_mask = np.zeros(max_node_num, dtype=bool)
        native_query_edge_count = 0
        for window in self.query_windows:
            raw = torch.load(self.test_files[window], map_location="cpu")
            native_query_edge_count += int(raw.src.numel())
            restore_mask[raw.src.numpy()] = True
            restore_mask[raw.dst.numpy()] = True
            del raw
        self.native_query_edge_count = native_query_edge_count
        restore_mask[self.global_node_ids] = True
        self.affected_global = torch.as_tensor(
            np.flatnonzero(restore_mask), dtype=torch.long, device=self.device
        )
        loader = self.model.encoder.neighbor_loader
        self.base_neighbors = loader.neighbors[self.affected_global].clone()
        self.base_edge_ids = loader.e_id[self.affected_global].clone()
        self.base_cur_edge_id = int(loader.cur_e_id)
        # Keep one reusable edge-type buffer instead of allocating and freeing
        # a several-hundred-megabyte torch.cat result on every query. Repeated
        # allocation fragmented Windows' committed virtual memory during long
        # multi-seed runs even though each individual query completed.
        initial_capacity = (
            self.base_cur_edge_id + native_query_edge_count + 4096
        )
        edge_type_shape = tuple(self.full_data.edge_type.shape[1:])
        self.query_edge_type_buffer = torch.empty(
            (initial_capacity, *edge_type_shape),
            dtype=self.full_data.edge_type.dtype,
        )
        self.query_edge_type_buffer[: self.base_cur_edge_id].copy_(
            self.full_data.edge_type[: self.base_cur_edge_id]
        )

        # Every candidate is observed in at least one selected window. Node
        # features are immutable, so one observed endpoint vector is enough.
        feature_dim = int(sample.x_src.shape[1])
        self.local_features = torch.zeros(
            (len(self.global_node_ids), feature_dim), dtype=torch.float32
        )
        feature_seen = np.zeros(len(self.global_node_ids), dtype=bool)
        for window in self.selected_windows:
            graph = load_processed(self.test_files[window])
            src = graph.src.cpu().numpy().astype(np.int64, copy=False)
            dst = graph.dst.cpu().numpy().astype(np.int64, copy=False)
            for edge_index, global_id in enumerate(src):
                local = self.global_to_local.get(int(global_id))
                if local is not None and not feature_seen[local]:
                    self.local_features[local] = graph.x_src[edge_index].cpu()
                    feature_seen[local] = True
            for edge_index, global_id in enumerate(dst):
                local = self.global_to_local.get(int(global_id))
                if local is not None and not feature_seen[local]:
                    self.local_features[local] = graph.x_dst[edge_index].cpu()
                    feature_seen[local] = True
            del graph
        if not bool(feature_seen.all()):
            raise RuntimeError(
                f"Orthrus worker lacks features for {int((~feature_seen).sum())} attack nodes"
            )
        del sample

    def _restore_temporal_state(self) -> None:
        loader = self.model.encoder.neighbor_loader
        loader.neighbors[self.affected_global] = self.base_neighbors
        loader.e_id[self.affected_global] = self.base_edge_ids
        loader.cur_e_id = self.base_cur_edge_id

    def _ensure_query_edge_capacity(self, required: int) -> None:
        if required <= int(self.query_edge_type_buffer.shape[0]):
            return
        torch = self.torch
        current = int(self.query_edge_type_buffer.shape[0])
        capacity = max(required + 4096, current * 2)
        replacement = torch.empty(
            (capacity, *self.query_edge_type_buffer.shape[1:]),
            dtype=self.query_edge_type_buffer.dtype,
        )
        replacement[:current].copy_(self.query_edge_type_buffer)
        self.query_edge_type_buffer = replacement

    def _modified_window(
        self,
        window: int,
        modifications: List[Dict[str, int]],
        base,
    ):
        torch = self.torch
        selected = [row for row in modifications if int(row["window"]) == window]
        if not selected:
            return base

        src_local = torch.as_tensor(
            [int(row["src"]) for row in selected], dtype=torch.long
        )
        dst_local = torch.as_tensor(
            [int(row["dst"]) for row in selected], dtype=torch.long
        )
        src = torch.as_tensor(
            self.global_node_ids[src_local.numpy()], dtype=torch.long
        )
        dst = torch.as_tensor(
            self.global_node_ids[dst_local.numpy()], dtype=torch.long
        )
        edge_type_ids = torch.as_tensor(
            [int(row["edge_type"]) for row in selected], dtype=torch.long
        )
        edge_type = torch.nn.functional.one_hot(
            edge_type_ids, num_classes=10
        ).to(torch.float32)
        x_src = self.local_features.index_select(0, src_local)
        x_dst = self.local_features.index_select(0, dst_local)
        msg = torch.cat([x_src, x_dst], dim=1)
        # Legacy attacks placed every injected event at the beginning of the
        # target window.  Full-aware Orthrus attacks instead attach a
        # ``before_time`` anchor: the event is inserted immediately before the
        # native edge currently responsible for the target's maximum loss.
        # Keep nanosecond timestamps strictly before the anchor when possible.
        first_time = int(base.t[0])
        anchor_counts: Dict[int, int] = {}
        added_time_values = []
        for row in selected:
            anchor = int(row.get("before_time", first_time))
            offset = anchor_counts.get(anchor, 0) + 1
            anchor_counts[anchor] = offset
            added_time_values.append(anchor - offset)
        added_times = torch.as_tensor(added_time_values, dtype=base.t.dtype)
        attack_mod_ids = torch.as_tensor(
            [int(row.get("probe_id", -1)) for row in selected],
            dtype=torch.long,
        )

        combined_t = torch.cat([added_times, base.t.cpu()])
        # NumPy lexsort gives deterministic ordering and places injected edges
        # before native edges when timestamps tie.
        tie_break = np.concatenate([
            np.arange(len(selected), dtype=np.int64),
            len(selected) + np.arange(int(base.t.numel()), dtype=np.int64),
        ])
        order = np.lexsort((tie_break, combined_t.numpy()))
        order = torch.as_tensor(order, dtype=torch.long)

        modified = self.TemporalData(
            src=torch.cat([src, base.src.cpu()]).index_select(0, order),
            dst=torch.cat([dst, base.dst.cpu()]).index_select(0, order),
            t=combined_t.index_select(0, order),
            msg=torch.cat([msg, base.msg.cpu()]).index_select(0, order),
        )
        modified.x_src = torch.cat([x_src, base.x_src.cpu()]).index_select(0, order)
        modified.x_dst = torch.cat([x_dst, base.x_dst.cpu()]).index_select(0, order)
        modified.edge_type = torch.cat([edge_type, base.edge_type.cpu()]).index_select(0, order)
        modified.attack_mod_id = torch.cat([
            attack_mod_ids,
            torch.full((int(base.t.numel()),), -1, dtype=torch.long),
        ]).index_select(0, order)
        modified.edge_feats = modified.edge_type
        modified.edge_index = torch.stack([modified.src, modified.dst])
        return modified

    def query(
        self,
        modifications: Iterable[Dict[str, int]],
        inspect_nodes: Iterable[int] = (),
    ) -> Dict[str, Any]:
        torch = self.torch
        modifications = [dict(row) for row in modifications]
        self._restore_temporal_state()
        required = (
            self.base_cur_edge_id
            + self.native_query_edge_count
            + len(modifications)
        )
        self._ensure_query_edge_capacity(required)
        cursor = self.base_cur_edge_id
        for window in self.query_windows:
            base = self.load_processed(self.test_files[window])
            graph = self._modified_window(window, modifications, base)
            edge_types = graph.edge_type.cpu()
            end = cursor + int(edge_types.shape[0])
            self.query_edge_type_buffer[cursor:end].copy_(edge_types)
            cursor = end
            del graph, base
        query_full_data = self.Data(
            edge_type=self.query_edge_type_buffer[:cursor]
        )
        window_max = np.zeros(len(self.global_node_ids), dtype=np.float64)
        inspect_nodes = sorted({
            int(node) for node in inspect_nodes
            if 0 <= int(node) < len(self.global_node_ids)
        })
        inspect_global = {
            int(self.global_node_ids[local]): local for local in inspect_nodes
        }
        max_loss_edges: Dict[int, Dict[str, Any]] = {}
        added_edge_losses: Dict[int, float] = {}

        with torch.inference_mode():
            for window in self.query_windows:
                base = self.load_processed(self.test_files[window])
                graph = self._modified_window(window, modifications, base)
                graph = graph.to(self.device)
                for batch in self.batch_loader_factory(
                    self.cfg, graph, self.model.graph_reindexer
                ):
                    losses = self.model(batch, query_full_data, inference=True)
                    edge_index = (
                        batch.original_edge_index
                        if hasattr(batch, "original_edge_index")
                        else batch.edge_index
                    )
                    endpoints = edge_index.detach().cpu().numpy().astype(np.int64)
                    values = losses.detach().cpu().numpy().astype(np.float64)
                    times = batch.t.detach().cpu().numpy().astype(np.int64)
                    edge_type_ids = (
                        batch.edge_type.detach().cpu().argmax(dim=1).numpy()
                        .astype(np.int64)
                    )
                    if hasattr(batch, "attack_mod_id"):
                        mod_ids = (
                            batch.attack_mod_id.detach().cpu().numpy()
                            .astype(np.int64)
                        )
                    else:
                        mod_ids = np.full(values.shape[0], -1, dtype=np.int64)

                    for edge_offset, value in enumerate(values.tolist()):
                        probe_id = int(mod_ids[edge_offset])
                        if probe_id >= 0:
                            added_edge_losses[probe_id] = max(
                                float(value),
                                added_edge_losses.get(probe_id, float("-inf")),
                            )
                        if not inspect_global:
                            continue
                        src_global = int(endpoints[0, edge_offset])
                        dst_global = int(endpoints[1, edge_offset])
                        for endpoint_global in {src_global, dst_global}:
                            local = inspect_global.get(endpoint_global)
                            if local is None:
                                continue
                            previous = max_loss_edges.get(local)
                            if previous is None or float(value) > previous["loss"]:
                                max_loss_edges[local] = {
                                    "loss": float(value),
                                    "window": int(window),
                                    "time": int(times[edge_offset]),
                                    "src_global": src_global,
                                    "dst_global": dst_global,
                                    "src_local": int(
                                        self.global_to_local.get(src_global, -1)
                                    ),
                                    "dst_local": int(
                                        self.global_to_local.get(dst_global, -1)
                                    ),
                                    "edge_type": int(edge_type_ids[edge_offset]),
                                    "is_injected": probe_id >= 0,
                                    "probe_id": probe_id,
                                }
                    for endpoint_values in endpoints:
                        positions = np.searchsorted(self.global_node_ids, endpoint_values)
                        valid = positions < len(self.global_node_ids)
                        valid_indices = np.flatnonzero(valid)
                        if valid_indices.size:
                            valid[valid_indices] &= (
                                self.global_node_ids[positions[valid_indices]]
                                == endpoint_values[valid_indices]
                            )
                        np.maximum.at(window_max, positions[valid], values[valid])
                graph.to("cpu")
                del graph, base, batch, losses

        local_scores = np.maximum(self.outside_max, window_max)
        for local in inspect_nodes:
            outside = float(self.outside_max[local])
            current = max_loss_edges.get(local)
            if current is None or outside > float(current["loss"]):
                max_loss_edges[local] = {
                    "loss": outside,
                    "window": -1,
                    "time": -1,
                    "src_global": -1,
                    "dst_global": -1,
                    "src_local": -1,
                    "dst_local": -1,
                    "edge_type": -1,
                    "is_injected": False,
                    "probe_id": -1,
                    "outside_replay_interval": True,
                }
        global_scores = self.baseline_result_scores.copy()
        for local, global_id in enumerate(self.global_node_ids):
            position = self.result_positions.get(int(global_id))
            if position is not None:
                global_scores[position] = local_scores[local]

        if self.alert_mode == "full":
            alert_positions = np.flatnonzero(global_scores > self.native_threshold)
            alert_global = set(
                int(node) for node in self.baseline_result_ids[alert_positions]
            )
            centers = np.asarray([], dtype=np.float64)
        else:
            ranked_positions = np.argsort(global_scores, kind="stable")
            top_positions = ranked_positions[-20:]
            top_scores = global_scores[top_positions]
            labels = self.KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(
                top_scores.reshape(-1, 1)
            )
            centers = np.asarray([
                top_scores[labels == cluster].mean() for cluster in range(2)
            ])
            high_cluster = int(np.argmax(centers))
            alert_global = set(
                int(node)
                for node in self.baseline_result_ids[
                    top_positions[labels == high_cluster]
                ]
            )
        alert_local = sorted(
            self.global_to_local[node]
            for node in alert_global
            if node in self.global_to_local
        )
        del query_full_data
        return {
            "local_scores": local_scores,
            "native_alerts": alert_local,
            "global_alert_count": len(alert_global),
            "cluster_centers": centers.tolist(),
            "max_loss_edges": {
                int(node): edge for node, edge in max_loss_edges.items()
            },
            "added_edge_losses": {
                int(key): float(value)
                for key, value in added_edge_losses.items()
            },
        }


def worker_main(connection, state_path: str, device: str) -> None:
    try:
        runtime = _Runtime(state_path, device)
        connection.send({"status": "ready", "device": str(runtime.device)})
        while True:
            request = connection.recv()
            command = request.get("command")
            if command == "close":
                connection.send({"status": "closed"})
                return
            if command != "query":
                raise ValueError(f"unknown Orthrus worker command: {command}")
            result = runtime.query(
                request.get("modifications", ()),
                request.get("inspect_nodes", ()),
            )
            connection.send({"status": "ok", "result": result})
    except BaseException as error:
        try:
            connection.send({
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
        finally:
            connection.close()
