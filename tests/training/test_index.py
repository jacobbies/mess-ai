"""Tests for FAISS training index wrappers."""

from __future__ import annotations

import numpy as np
import pytest

import mess.training.index as training_index
from mess.training.index import FaissRetrievalIndex, should_refresh_index

pytestmark = pytest.mark.unit


class _FakeIndexFlatIP:
    def __init__(self, dim: int) -> None:
        self.d = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = np.asarray(vectors, dtype=np.float32).copy()

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sims = np.asarray(queries, dtype=np.float32) @ self._vectors.T
        topk_idx = np.argsort(-sims, axis=1)[:, :k]
        topk_scores = np.take_along_axis(sims, topk_idx, axis=1)
        return topk_scores.astype(np.float32), topk_idx.astype(np.int64)


class _FakeFaiss:
    IndexFlatIP = _FakeIndexFlatIP

    @staticmethod
    def normalize_L2(vectors: np.ndarray) -> None:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        vectors /= norms


def test_faiss_retrieval_index_search_and_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_index, "_require_faiss", lambda: _FakeFaiss)

    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    index = FaissRetrievalIndex.build(vectors)
    distances, indices = index.search(vectors[0], k=2)

    assert indices.shape == (1, 2)
    assert int(indices[0, 0]) == 0
    assert distances[0, 0] == pytest.approx(1.0, rel=1e-5)

    index.rebuild(vectors[:, ::-1].copy())
    assert index.version == 2


def test_should_refresh_index_schedule() -> None:
    assert should_refresh_index(step=1, refresh_every=10)
    assert should_refresh_index(step=10, refresh_every=10)
    assert not should_refresh_index(step=9, refresh_every=10)
