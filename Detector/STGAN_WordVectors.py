"""Neutral NumPy representation of STGAN's released Word2Vec vocabulary."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class STGANWordVectors:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as payload:
            self.keys = np.asarray(payload["keys"], dtype=np.str_)
            self.vectors = np.asarray(payload["vectors"], dtype=np.float32)
            self.source_model = (
                str(np.asarray(payload["source_model"]).item())
                if "source_model" in payload.files
                else ""
            )
        if self.vectors.shape != (len(self.keys), 30):
            raise ValueError("STGAN neutral Word2Vec file must contain N x 30 vectors")
        self.key_to_index = {str(key): index for index, key in enumerate(self.keys)}
        self.vector_size = 30

    def __contains__(self, key: str) -> bool:
        return str(key) in self.key_to_index

    def __getitem__(self, key: str) -> np.ndarray:
        return self.vectors[self.key_to_index[str(key)]]
