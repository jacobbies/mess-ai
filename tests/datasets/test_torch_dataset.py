"""Tests for generic torch dataset adapter."""

from __future__ import annotations

import pytest
from torch.utils.data import ConcatDataset

from mess.datasets.clip_index import ClipIndex, ClipRecord
from mess.datasets.torch_dataset import GeneralTorchDataset

pytestmark = pytest.mark.unit


def _make_index(prefix: str) -> ClipIndex:
    records = [
        ClipRecord(
            clip_id=f"{prefix}:track_a:00000",
            dataset_id=prefix,
            recording_id=f"{prefix}_rec",
            track_id="track_a",
            segment_idx=0,
            start_sec=0.0,
            end_sec=5.0,
            split="train",
            embedding_path=f"/tmp/{prefix}_track_a.npy",
        ),
        ClipRecord(
            clip_id=f"{prefix}:track_a:00001",
            dataset_id=prefix,
            recording_id=f"{prefix}_rec",
            track_id="track_a",
            segment_idx=1,
            start_sec=2.5,
            end_sec=7.5,
            split="train",
            embedding_path=f"/tmp/{prefix}_track_a.npy",
        ),
    ]
    return ClipIndex(records)


class TestGeneralTorchDataset:
    def test_returns_meta_without_resolvers(self):
        index = _make_index("smd")
        dataset = GeneralTorchDataset(index)

        sample = dataset[0]
        assert sample["clip_id"] == "smd:track_a:00000"
        assert sample["meta"]["track_id"] == "track_a"
        assert "embedding" not in sample

    def test_uses_resolvers(self):
        index = _make_index("smd")
        resolver_calls = []

        def embedding_resolver(clip_id: str) -> list[float]:
            resolver_calls.append(clip_id)
            return [1.0, 2.0]

        dataset = GeneralTorchDataset(
            index,
            field_resolvers={"embedding": embedding_resolver},
        )

        sample = dataset[1]
        assert resolver_calls == ["smd:track_a:00001"]
        assert sample["embedding"] == [1.0, 2.0]

    def test_concat_dataset_supports_multi_dataset_training(self):
        left = GeneralTorchDataset(_make_index("smd"))
        right = GeneralTorchDataset(_make_index("maestro"))
        combo = ConcatDataset([left, right])

        assert len(combo) == 4
        sample = combo[3]
        assert sample["clip_id"].startswith("maestro:")
