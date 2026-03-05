"""Smoke tests for retrieval-augmented projection-head training."""

from __future__ import annotations

import numpy as np
import pytest

import mess.training.index as training_index
from mess.datasets.clip_index import ClipRecord
from mess.training import RetrievalSSLConfig, train_projection_head

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


def _synthetic_records(n: int) -> list[ClipRecord]:
    rows: list[ClipRecord] = []
    for idx in range(n):
        rows.append(
            ClipRecord(
                clip_id=f"clip_{idx:03d}",
                dataset_id="smd",
                recording_id=f"rec_{idx % 4}",
                track_id=f"track_{idx % 6}",
                segment_idx=idx % 8,
                start_sec=float((idx % 8) * 2.5),
                end_sec=float((idx % 8) * 2.5 + 5.0),
                split="train",
                embedding_path=f"/tmp/fake_{idx % 6}.npy",
            )
        )
    return rows


def test_train_projection_head_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_index, "_require_faiss", lambda: _FakeFaiss)

    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((32, 64)).astype(np.float32)
    records = _synthetic_records(32)

    config = RetrievalSSLConfig(
        num_steps=8,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        projection_dim=16,
        hidden_dim=32,
        search_k=12,
        positives_per_query=2,
        negatives_per_query=4,
        min_time_separation_sec=1.0,
        refresh_every=2,
        ema_decay=0.99,
        seed=123,
        device="cpu",
    )

    result = train_projection_head(vectors, records, config)

    assert result.steps_completed > 0
    assert result.input_dim == 64
    assert result.output_dim == 16
    assert result.final_index_version >= 1
    assert result.metrics
    assert "network.0.weight" in result.online_state_dict
    assert np.isfinite(result.metrics[-1]["loss"])
