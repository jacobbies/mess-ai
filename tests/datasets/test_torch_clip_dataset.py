"""Tests for clip-specialized torch dataset with optional target supervision."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mess.datasets.clip_index import ClipIndex, ClipRecord
from mess.datasets.torch_clip_dataset import ClipDataset

pytestmark = pytest.mark.unit


class _DummyEmbeddingStore:
    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors

    def get(self, clip_id: str) -> np.ndarray:
        return self._vectors[clip_id]


class _DummyTargetStore:
    def __init__(self, values: dict[str, dict[str, float] | None]) -> None:
        self._values = values

    def get(self, clip_id: str) -> dict[str, float] | None:
        return self._values.get(clip_id)


def _make_index() -> ClipIndex:
    records = [
        ClipRecord(
            clip_id="smd:track_a:00000",
            dataset_id="smd",
            recording_id="rec_a",
            track_id="track_a",
            segment_idx=0,
            start_sec=0.0,
            end_sec=5.0,
            split="train",
            embedding_path="/tmp/track_a.npy",
        ),
        ClipRecord(
            clip_id="smd:track_a:00001",
            dataset_id="smd",
            recording_id="rec_a",
            track_id="track_a",
            segment_idx=1,
            start_sec=2.5,
            end_sec=7.5,
            split="train",
            embedding_path="/tmp/track_a.npy",
        ),
    ]
    return ClipIndex(records)


class TestClipDataset:
    def test_returns_embedding_and_meta(self):
        index = _make_index()
        vectors = {
            "smd:track_a:00000": np.ones((13, 768), dtype=np.float32),
            "smd:track_a:00001": np.zeros((13, 768), dtype=np.float32),
        }
        dataset = ClipDataset(index=index, embedding_store=_DummyEmbeddingStore(vectors))

        sample = dataset[0]
        assert sample["embedding"].shape == (13, 768)
        assert sample["meta"]["segment_idx"] == 0
        assert sample["targets"] is None
        assert sample["has_targets"] is False

    def test_optional_targets_build_dense_target_tensor(self):
        index = _make_index()
        vectors = {
            "smd:track_a:00000": np.ones((13, 768), dtype=np.float32),
            "smd:track_a:00001": np.zeros((13, 768), dtype=np.float32),
        }
        targets = {
            "smd:track_a:00000": {"expression.rubato": 0.4},
            "smd:track_a:00001": None,
        }
        dataset = ClipDataset(
            index=index,
            embedding_store=_DummyEmbeddingStore(vectors),
            target_store=_DummyTargetStore(targets),
            target_keys=["expression.rubato", "dynamics.dynamic_range"],
        )

        first = dataset[0]
        second = dataset[1]

        assert first["has_targets"] is True
        expected_first = torch.tensor([0.4, float("nan")])
        assert torch.allclose(first["target_values"], expected_first, equal_nan=True)
        assert torch.equal(first["target_mask"], torch.tensor([True, False]))

        assert second["has_targets"] is False
        assert torch.equal(second["target_mask"], torch.tensor([False, False]))

    def test_collate_fn_handles_mixed_supervision(self):
        index = _make_index()
        vectors = {
            "smd:track_a:00000": np.ones((13, 768), dtype=np.float32),
            "smd:track_a:00001": np.zeros((13, 768), dtype=np.float32),
        }
        targets = {
            "smd:track_a:00000": {"expression.rubato": 0.4},
            "smd:track_a:00001": None,
        }
        dataset = ClipDataset(
            index=index,
            embedding_store=_DummyEmbeddingStore(vectors),
            target_store=_DummyTargetStore(targets),
            target_keys=["expression.rubato"],
        )

        batch = ClipDataset.collate_fn([dataset[0], dataset[1]])
        assert batch["embedding"].shape == (2, 13, 768)
        assert torch.equal(batch["has_targets"], torch.tensor([True, False]))
        assert batch["target_values"].shape == (2, 1)
        assert batch["target_mask"].shape == (2, 1)
