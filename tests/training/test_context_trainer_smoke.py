"""Smoke tests for segment transformer contextualizer training."""

from __future__ import annotations

import numpy as np
import pytest

import mess.training.index as training_index
from mess.datasets.clip_index import ClipRecord
from mess.training.context_config import ContextualizerConfig
from mess.training.context_trainer import (
    TrackSegments,
    collate_track_batch,
    group_records_by_track,
    train_contextualizer,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fake FAISS (mirrors test_trainer_smoke.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _synthetic_tracks(
    n_tracks: int,
    input_dim: int = 64,
    min_segments: int = 3,
    max_segments: int = 8,
    seed: int = 42,
) -> list[TrackSegments]:
    rng = np.random.default_rng(seed)
    tracks: list[TrackSegments] = []

    for i in range(n_tracks):
        n_seg = rng.integers(min_segments, max_segments + 1)
        track_id = f"track_{i:03d}"
        recording_id = f"rec_{i % 4}"
        dataset_id = "synthetic"

        segments = rng.standard_normal((n_seg, input_dim)).astype(np.float32)

        clip_records = [
            ClipRecord(
                clip_id=f"{dataset_id}:{track_id}:{seg:05d}",
                dataset_id=dataset_id,
                recording_id=recording_id,
                track_id=track_id,
                segment_idx=seg,
                start_sec=float(seg * 2.5),
                end_sec=float(seg * 2.5 + 5.0),
                split="train",
                embedding_path=f"/tmp/fake_{track_id}.npy",
            )
            for seg in range(n_seg)
        ]

        tracks.append(
            TrackSegments(
                track_id=track_id,
                recording_id=recording_id,
                dataset_id=dataset_id,
                segments=segments,
                clip_records=clip_records,
            )
        )

    return tracks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_group_records_by_track() -> None:
    records = [
        ClipRecord(
            clip_id="a:t1:00000",
            dataset_id="a",
            recording_id="r1",
            track_id="t1",
            segment_idx=0,
            start_sec=0.0,
            end_sec=5.0,
            split="train",
            embedding_path="/tmp/t1.npy",
        ),
        ClipRecord(
            clip_id="a:t1:00001",
            dataset_id="a",
            recording_id="r1",
            track_id="t1",
            segment_idx=1,
            start_sec=2.5,
            end_sec=7.5,
            split="train",
            embedding_path="/tmp/t1.npy",
        ),
        ClipRecord(
            clip_id="a:t2:00000",
            dataset_id="a",
            recording_id="r2",
            track_id="t2",
            segment_idx=0,
            start_sec=0.0,
            end_sec=5.0,
            split="train",
            embedding_path="/tmp/t2.npy",
        ),
    ]

    grouped = group_records_by_track(records)
    assert set(grouped.keys()) == {"t1", "t2"}
    assert len(grouped["t1"]) == 2
    assert len(grouped["t2"]) == 1
    # Check sorted by segment_idx
    assert grouped["t1"][0].segment_idx == 0
    assert grouped["t1"][1].segment_idx == 1


def test_collate_track_batch() -> None:
    tracks = _synthetic_tracks(4, input_dim=32)
    indices = np.array([0, 2])

    import torch

    segments, lengths = collate_track_batch(tracks, indices, torch.device("cpu"))

    assert segments.ndim == 3
    assert segments.shape[0] == 2
    assert lengths.shape == (2,)
    assert int(lengths[0]) == tracks[0].segments.shape[0]
    assert int(lengths[1]) == tracks[2].segments.shape[0]


def test_train_contextualizer_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_index, "_require_faiss", lambda: _FakeFaiss)

    tracks = _synthetic_tracks(16, input_dim=64)

    config = ContextualizerConfig(
        input_dim=64,
        context_dim=32,
        num_transformer_layers=1,
        num_heads=4,
        ff_dim=64,
        max_segments=16,
        dropout=0.0,
        pool_mode="mean",
        num_steps=4,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        warmup_steps=2,
        temperature=0.1,
        global_loss_weight=1.0,
        local_loss_weight=0.5,
        search_k=12,
        positives_per_query=1,
        negatives_per_query=4,
        min_time_separation_sec=1.0,
        refresh_every=2,
        ema_decay=0.99,
        seed=123,
        device="cpu",
    )

    result = train_contextualizer(tracks, config)

    assert result.steps_completed > 0
    assert result.input_dim == 64
    assert result.context_dim == 32
    assert result.final_index_version >= 1
    assert result.metrics
    assert "loss_total" in result.metrics[-1]
    assert "loss_global" in result.metrics[-1]
    assert "loss_local" in result.metrics[-1]
    assert np.isfinite(result.metrics[-1]["loss_total"])
    assert "input_proj.weight" in result.online_state_dict
    assert "input_proj.weight" in result.target_state_dict
