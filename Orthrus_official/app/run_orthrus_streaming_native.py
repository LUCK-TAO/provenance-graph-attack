"""Memory-bounded native Orthrus reproduction for large datasets.

The released Orthrus implementation eagerly materializes every high-dimensional
edge message from train/validation/test before either training or testing.  That
is unnecessary for the released configuration: the temporal encoder consults
only ``full_data.edge_type``.  This runner preserves the official model,
pretrained weights, temporal event order, validation-before-test state updates,
ground truth and evaluation code while loading one time window at a time.

It is intended to be executed in the official Docker image, from ``/home``.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from numpy.lib.format import open_memmap
from torch_geometric.data import Data
from tqdm import tqdm


SRC = Path("/home/src")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import get_runtime_required_args, get_yml_cfg  # noqa: E402
from data_utils import extract_msg_from_data, load_model, save_model  # noqa: E402
from detection import evaluation  # noqa: E402
from factory import batch_loader_factory, build_model  # noqa: E402
from provnet_utils import (  # noqa: E402
    gen_nodeid2msg,
    get_device,
    init_database_connection,
    log,
    ns_time_to_datetime_US,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="THEIA_E3", choices=["THEIA_E3"])
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "checkpoint", "infer", "evaluate"],
    )
    parser.add_argument("--rebuild-edge-type-cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def native_cfg(dataset: str, use_cpu: bool):
    # Exact command published by Orthrus for THEIA_E3 (README, lines 83-86).
    command = [
        dataset,
        "--from_weights",
        "--detection.gnn_training.encoder.graph_attention.dropout=0.1",
        "--seed=2",
    ]
    if use_cpu:
        command.append("--cpu")
    parsed = get_runtime_required_args(args=command)
    return get_yml_cfg(parsed)


def seed_everything(cfg) -> None:
    seed = int(cfg._seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_files(cfg, split: str) -> list[Path]:
    root = Path(cfg.edge_featurization.embed_edges._edge_embeds_dir) / split
    files = sorted(path for path in root.iterdir() if path.is_file())
    if not files:
        raise FileNotFoundError(f"no embedded {split} windows found in {root}")
    return files


def load_raw(path: Path):
    return torch.load(path, map_location="cpu")


def load_processed(path: Path, cfg):
    return extract_msg_from_data([load_raw(path)], cfg)[0]


def inspect_dataset(files_by_split: dict[str, list[Path]]) -> tuple[int, int]:
    total_edges = 0
    max_node = -1
    for split in ("train", "val", "test"):
        for path in tqdm(files_by_split[split], desc=f"inspect {split}"):
            raw = load_raw(path)
            total_edges += int(raw.src.numel())
            max_node = max(
                max_node,
                int(torch.maximum(raw.src.max(), raw.dst.max()).item()),
            )
            del raw
    return total_edges, max_node + 1


def build_edge_type_cache(
    cfg,
    files_by_split: dict[str, list[Path]],
    total_edges: int,
    rebuild: bool,
) -> Path:
    cache_dir = Path("/home/artifacts/streaming_cache") / cfg.dataset.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "all_edge_types.npy"
    expected = (total_edges, int(cfg.dataset.num_edge_types))
    if cache_path.exists() and not rebuild:
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape == expected and cached.dtype == np.float32:
            log(f"Using edge-type cache {cache_path}, shape={cached.shape}")
            return cache_path
        del cached

    log(f"Building edge-type cache {cache_path}, shape={expected}")
    cached = open_memmap(cache_path, mode="w+", dtype=np.float32, shape=expected)
    offset = 0
    # feature_word2vec messages are laid out as:
    # src_type(3), src_emb(128), edge_type(10), dst_type(3), dst_emb(128).
    edge_start = int(cfg.dataset.num_node_types) + int(cfg.edge_featurization.embed_nodes.emb_dim)
    edge_end = edge_start + int(cfg.dataset.num_edge_types)
    for split in ("train", "val", "test"):
        for path in tqdm(files_by_split[split], desc=f"cache {split}"):
            raw = load_raw(path)
            count = int(raw.src.numel())
            cached[offset : offset + count] = (
                raw.msg[:, edge_start:edge_end].cpu().numpy().astype(np.float32, copy=False)
            )
            offset += count
            del raw
    cached.flush()
    del cached
    if offset != total_edges:
        raise RuntimeError(f"edge cache length mismatch: {offset} != {total_edges}")
    return cache_path


def full_data_from_cache(cache_path: Path) -> Data:
    # r+ keeps the NumPy array writable, avoiding PyTorch's read-only mmap warning.
    array = np.load(cache_path, mmap_mode="r+")
    return Data(edge_type=torch.from_numpy(array))


@torch.no_grad()
def test_window_compact(data, full_data, model, split: str, cfg, device) -> None:
    """Run the official score computation and save evaluation-required columns.

    Orthrus' node evaluation reads only loss/srcnode/dstnode.  The released
    tester additionally expands source/destination labels and relation strings
    with a Python dictionary per event, although those columns are used only by
    the optional tracing stage.  Keeping the three native evaluation columns
    makes THEIA inference much faster and leaves all scores and labels intact.
    """
    model.eval()
    losses, sources, destinations, times = [], [], [], []
    for batch in batch_loader_factory(cfg, data, model.graph_reindexer):
        batch_losses = model(batch, full_data, inference=True)
        edge_index = (
            batch.original_edge_index
            if hasattr(batch, "original_edge_index")
            else batch.edge_index
        )
        losses.append(batch_losses.detach().cpu())
        sources.append(edge_index[0].detach().cpu())
        destinations.append(edge_index[1].detach().cpu())
        times.append(batch.t.detach().cpu())

    loss = torch.cat(losses).numpy()
    src = torch.cat(sources).numpy()
    dst = torch.cat(destinations).numpy()
    event_time = torch.cat(times).numpy()
    time_interval = (
        ns_time_to_datetime_US(int(data.t[0]))
        + "~"
        + ns_time_to_datetime_US(int(event_time[-1]))
    )
    output = (
        Path(cfg.detection.gnn_testing._edge_losses_dir)
        / split
        / "model_epoch_1"
    )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "loss": loss,
        "srcnode": src,
        "dstnode": dst,
    }).to_csv(output / f"{time_interval}.csv", index=False)


@torch.no_grad()
def create_native_checkpoint(cfg, files_by_split, max_node_num: int, device) -> None:
    """Create the checkpoint state that official --from_weights testing needs.

    During the one official from-weights training epoch, model parameters are
    later overwritten by the released THEIA_E3.pkl.  The only training output
    that affects native testing is LastNeighborLoader after all training edges
    have been inserted.  Inserting the same endpoints in the same batch order
    reproduces that state without performing discarded forward/backward passes.
    """
    output = Path(cfg.detection.gnn_training._trained_models_dir) / "model_epoch_1"
    if (output / "state_dict.pkl").exists() and (output / "neighbor_loader.pkl").exists():
        log(f"Native temporal checkpoint already exists: {output}")
        return

    sample = load_processed(files_by_split["train"][0], cfg)
    model = build_model(sample, device, cfg, max_node_num)
    loader = model.encoder.neighbor_loader
    loader.reset_state()
    batch_size = int(cfg.detection.gnn_training.encoder.batch_size)
    for path in tqdm(files_by_split["train"], desc="prime training history"):
        raw = load_raw(path)
        for start in range(0, int(raw.src.numel()), batch_size):
            end = min(start + batch_size, int(raw.src.numel()))
            loader.insert(raw.src[start:end].to(device), raw.dst[start:end].to(device))
        del raw
    save_model(model, str(output))
    log(
        f"Saved native temporal checkpoint {output}; "
        f"training events={loader.cur_e_id}"
    )
    del sample, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def run_streaming_inference(cfg, files_by_split, full_data, max_node_num: int, device) -> None:
    checkpoint = Path(cfg.detection.gnn_training._trained_models_dir) / "model_epoch_1"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"missing native checkpoint: {checkpoint}")

    sample = load_processed(files_by_split["test"][0], cfg)
    model = build_model(sample, device, cfg, max_node_num)
    model = load_model(model, str(checkpoint))
    weights = Path(cfg._from_weights_path) / f"{cfg.dataset.name}.pkl"
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    del sample

    for split in ("val", "test"):
        log(f"Streaming native inference: {split}")
        for path in tqdm(files_by_split[split], desc=f"{split} model_epoch_1"):
            graph = load_processed(path, cfg).to(device)
            test_window_compact(graph, full_data, model, split, cfg, device)
            graph.to("cpu")
            del graph
            gc.collect()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    cfg = native_cfg(args.dataset, args.cpu)
    seed_everything(cfg)
    wandb.init(mode="disabled", project="orthrus_repo", name=f"{args.dataset}_streaming")
    files_by_split = None
    total_edges = max_node_num = None
    device = get_device(cfg)
    if args.phase != "evaluate":
        files_by_split = {
            split: split_files(cfg, split) for split in ("train", "val", "test")
        }
        total_edges, max_node_num = inspect_dataset(files_by_split)
        log(f"Dataset {cfg.dataset.name}: edges={total_edges}, nodes={max_node_num}")

    if args.phase in ("all", "checkpoint"):
        create_native_checkpoint(cfg, files_by_split, max_node_num, device)

    if args.phase in ("all", "infer"):
        cache_path = build_edge_type_cache(
            cfg,
            files_by_split,
            total_edges,
            rebuild=args.rebuild_edge_type_cache,
        )
        full_data = full_data_from_cache(cache_path)
        run_streaming_inference(cfg, files_by_split, full_data, max_node_num, device)
        del full_data
        gc.collect()

    if args.phase in ("all", "evaluate"):
        evaluation.main(cfg)

    wandb.finish()
    log("Streaming native Orthrus reproduction completed.")


if __name__ == "__main__":
    main()
