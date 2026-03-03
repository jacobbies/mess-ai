"""Tests for clip-index-backed local embedding and target stores."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mess.datasets.clip_index import ClipIndex, ClipRecord
from mess.datasets.stores import (
    NpySegmentEmbeddingStore,
    NPZSegmentTargetStore,
    TorchCodecAudioStore,
)

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


class TestTorchCodecAudioStore:
    def test_decodes_clip_ranges_with_custom_path_resolver(self, tmp_path, mocker):
        embedding_path = tmp_path / "track_a.npy"
        np.save(embedding_path, np.zeros((2, 13, 768), dtype=np.float32))
        index = _index_for_file(str(embedding_path), n_segments=2)

        audio_path = tmp_path / "track_a.wav"
        audio_path.write_bytes(b"audio")

        decoder_cls = mocker.patch("mess.datasets.stores.AudioDecoder")
        decoder = decoder_cls.return_value
        decoder.get_samples_played_in_range.return_value = SimpleNamespace(
            data=torch.ones(1, 500, dtype=torch.float32)
        )

        store = TorchCodecAudioStore(
            index=index,
            audio_path_resolver=lambda _record: audio_path,
            sample_rate=100,
            num_channels=1,
        )

        first = store.get("smd:track_a:00000")
        second = store.get("smd:track_a:00001")

        decoder_cls.assert_called_once_with(
            str(audio_path),
            sample_rate=100,
            num_channels=1,
        )
        assert decoder.get_samples_played_in_range.call_count == 2
        decoder.get_samples_played_in_range.assert_any_call(0.0, 5.0)
        decoder.get_samples_played_in_range.assert_any_call(2.5, 7.5)
        assert first.shape == (500,)
        assert second.shape == (500,)

    def test_audio_root_resolution_finds_track_audio(self, tmp_path, mocker):
        embedding_path = tmp_path / "track_a.npy"
        np.save(embedding_path, np.zeros((1, 13, 768), dtype=np.float32))
        index = _index_for_file(str(embedding_path), n_segments=1)

        audio_root = tmp_path / "audio"
        nested = audio_root / "nested"
        nested.mkdir(parents=True)
        audio_path = nested / "track_a.wav"
        audio_path.write_bytes(b"audio")

        decoder_cls = mocker.patch("mess.datasets.stores.AudioDecoder")
        decoder = decoder_cls.return_value
        decoder.get_samples_played_in_range.return_value = SimpleNamespace(
            data=torch.ones(1, 10, dtype=torch.float32)
        )

        store = TorchCodecAudioStore(index=index, audio_root=audio_root)
        value = store.get("smd:track_a:00000")

        decoder_cls.assert_called_once_with(
            str(audio_path),
            sample_rate=24000,
            num_channels=1,
        )
        assert value.shape == (10,)

    def test_missing_audio_raises_file_not_found(self, tmp_path, mocker):
        embedding_path = tmp_path / "track_a.npy"
        np.save(embedding_path, np.zeros((1, 13, 768), dtype=np.float32))
        index = _index_for_file(str(embedding_path), n_segments=1)

        decoder_cls = mocker.patch("mess.datasets.stores.AudioDecoder")
        decoder_cls.return_value.get_samples_played_in_range.return_value = SimpleNamespace(
            data=torch.ones(1, 10, dtype=torch.float32)
        )
        store = TorchCodecAudioStore(index=index, audio_root=tmp_path / "audio")
        with pytest.raises(FileNotFoundError, match="No audio file found"):
            store.get("smd:track_a:00000")
