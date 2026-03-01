"""Tests for clip-index-backed local embedding and target stores."""

from __future__ import annotations

import numpy as np
import pytest

from mess.datasets.clip_index import ClipIndex, ClipRecord
from mess.datasets.stores import NpySegmentEmbeddingStore, NPZSegmentTargetStore

pytestmark = pytest.mark.unit


def _index_for_file(path: str, n_segments: int = 2) -> ClipIndex:
    records = []
    for idx in range(n_segments):
        records.append(
            ClipRecord(
                clip_id=f"smd:track_a:{idx:05d}",
                dataset_id="smd",
                recording_id="rec_a",
                track_id="track_a",
                segment_idx=idx,
                start_sec=idx * 2.5,
                end_sec=(idx * 2.5) + 5.0,
                split="train",
                embedding_path=path,
            )
        )
    return ClipIndex(records)


class TestNpySegmentEmbeddingStore:
    def test_get_returns_matrix_for_segment_embeddings(self, tmp_path):
        path = tmp_path / "track_a.npy"
        data = np.random.default_rng(0).standard_normal((2, 13, 768)).astype(np.float32)
        np.save(path, data)
        index = _index_for_file(str(path), n_segments=2)

        store = NpySegmentEmbeddingStore(index)
        value = store.get("smd:track_a:00001")

        assert value.shape == (13, 768)
        np.testing.assert_allclose(value, data[1], atol=1e-6)

    def test_get_layer_returns_768_vector(self, tmp_path):
        path = tmp_path / "track_a.npy"
        data = np.random.default_rng(1).standard_normal((2, 13, 768)).astype(np.float32)
        np.save(path, data)
        index = _index_for_file(str(path), n_segments=2)

        store = NpySegmentEmbeddingStore(index, layer=3)
        value = store.get("smd:track_a:00000")

        assert value.shape == (768,)
        np.testing.assert_allclose(value, data[0, 3], atol=1e-6)

    def test_raw_embeddings_reduce_time_dimension(self, tmp_path):
        path = tmp_path / "track_a.npy"
        data = np.random.default_rng(2).standard_normal((2, 13, 4, 768)).astype(np.float32)
        np.save(path, data)
        index = _index_for_file(str(path), n_segments=2)

        store = NpySegmentEmbeddingStore(index)
        value = store.get("smd:track_a:00001")

        expected = data[1].mean(axis=1)
        np.testing.assert_allclose(value, expected, atol=1e-6)

    def test_unknown_clip_raises(self, tmp_path):
        path = tmp_path / "track_a.npy"
        np.save(path, np.zeros((1, 13, 768), dtype=np.float32))
        index = _index_for_file(str(path), n_segments=1)

        store = NpySegmentEmbeddingStore(index)
        with pytest.raises(KeyError, match="Unknown clip_id"):
            store.get("missing")


class TestNPZSegmentTargetStore:
    def test_get_returns_flattened_targets_for_segment(self, tmp_path):
        target_dir = tmp_path / "targets"
        target_dir.mkdir()
        np.savez(
            target_dir / "track_a_segment_targets.npz",
            timbre={
                "spectral_centroid": np.array([10.0, 20.0], dtype=np.float32),
            },
            dynamics={
                "dynamic_range": np.array([1.0, 2.0], dtype=np.float32),
            },
        )

        path = tmp_path / "track_a.npy"
        np.save(path, np.zeros((2, 13, 768), dtype=np.float32))
        index = _index_for_file(str(path), n_segments=2)

        store = NPZSegmentTargetStore(index=index, targets_dir=target_dir)
        values = store.get("smd:track_a:00001")

        assert values is not None
        assert values["timbre.spectral_centroid"] == pytest.approx(20.0)
        assert values["dynamics.dynamic_range"] == pytest.approx(2.0)

    def test_missing_target_file_returns_none(self, tmp_path):
        path = tmp_path / "track_a.npy"
        np.save(path, np.zeros((1, 13, 768), dtype=np.float32))
        index = _index_for_file(str(path), n_segments=1)

        store = NPZSegmentTargetStore(index=index, targets_dir=tmp_path / "missing")
        assert store.get("smd:track_a:00000") is None

    def test_nan_values_are_dropped(self, tmp_path):
        target_dir = tmp_path / "targets"
        target_dir.mkdir()
        np.savez(
            target_dir / "track_a_segment_targets.npz",
            expression={
                "rubato": np.array([np.nan, 0.5], dtype=np.float32),
            },
        )

        path = tmp_path / "track_a.npy"
        np.save(path, np.zeros((2, 13, 768), dtype=np.float32))
        index = _index_for_file(str(path), n_segments=2)
        store = NPZSegmentTargetStore(index=index, targets_dir=target_dir)

        assert store.get("smd:track_a:00000") is None
        values = store.get("smd:track_a:00001")
        assert values == {"expression.rubato": 0.5}
