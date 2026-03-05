"""FAISS retrieval index helpers for training loops."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np


def _require_faiss() -> Any:
    try:
        return importlib.import_module("faiss")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "faiss is required for retrieval-augmented training. Install `faiss-cpu`."
        ) from exc


@dataclass
class FaissRetrievalIndex:
    """Flat cosine FAISS index with explicit rebuild support."""

    index: Any
    dimension: int
    ntotal: int
    version: int

    @classmethod
    def build(cls, vectors: np.ndarray) -> FaissRetrievalIndex:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"Expected non-empty 2D vectors, got shape {arr.shape}")

        faiss = _require_faiss()
        normalized = arr.copy()
        faiss.normalize_L2(normalized)
        index = faiss.IndexFlatIP(int(normalized.shape[1]))
        index.add(normalized)
        return cls(
            index=index,
            dimension=int(normalized.shape[1]),
            ntotal=int(normalized.shape[0]),
            version=1,
        )

    def rebuild(self, vectors: np.ndarray) -> None:
        rebuilt = FaissRetrievalIndex.build(vectors)
        self.index = rebuilt.index
        self.dimension = rebuilt.dimension
        self.ntotal = rebuilt.ntotal
        self.version += 1

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            raise ValueError("k must be > 0")

        arr = np.asarray(queries, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query shape [batch, {self.dimension}], got {arr.shape}"
            )

        faiss = _require_faiss()
        q = arr.copy()
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, min(k, self.ntotal))
        return np.asarray(distances), np.asarray(indices)


def should_refresh_index(step: int, refresh_every: int) -> bool:
    """Whether to rebuild index at this training step."""
    if refresh_every <= 0:
        raise ValueError("refresh_every must be > 0")
    return step == 1 or step % refresh_every == 0
