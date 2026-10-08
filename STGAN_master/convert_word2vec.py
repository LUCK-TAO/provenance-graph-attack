"""Convert the released Gensim pickle to a version-neutral NumPy file.

Run this script in any isolated environment that can open the released model.
The output contains exactly the vocabulary strings and float32 vectors used by
inference; no model is retrained and no random state is needed afterwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "train" / "word2vec_cadets.model"
DEFAULT_OUTPUT = ROOT / "train" / "word2vec_cadets_compat.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"output exists: {args.output}; pass --force")
    model = Word2Vec.load(str(args.input))
    keys = np.asarray(model.wv.index_to_key, dtype=np.str_)
    vectors = np.asarray(model.wv.vectors, dtype=np.float32)
    if vectors.shape != (len(keys), 30):
        raise ValueError(f"unexpected released vector matrix: {vectors.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        version=np.asarray(1, dtype=np.int64),
        source_model=np.asarray(str(args.input.resolve())),
        keys=keys,
        vectors=vectors,
    )
    print(f"saved={args.output.resolve()} vocabulary={len(keys)} dim={vectors.shape[1]}")


if __name__ == "__main__":
    main()
