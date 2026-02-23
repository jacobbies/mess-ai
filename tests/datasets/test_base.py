"""Tests for mess.datasets.base.BaseDataset."""

from pathlib import Path

import pytest

from mess.datasets.base import BaseDataset


class ConcreteTestDataset(BaseDataset):
    """Concrete subclass for testing abstract BaseDataset."""

    @property
    def dataset_id(self):
        return "test"

    @property
    def audio_dir(self):
        return self.data_root / "audio" / "test"

    @property
    def embeddings_dir(self):
        return self.data_root / "embeddings" / "test-emb"

    @property
    def name(self):
        return "Test Dataset"

    @property
    def description(self):
        return "A test dataset"


class TestAbstract:
    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError):
            BaseDataset(data_root=Path("/tmp"))


class TestGetAudioFiles:
    def test_with_wav_files(self, tmp_path):
        audio_dir = tmp_path / "audio" / "test"
        audio_dir.mkdir(parents=True)
        (audio_dir / "a.wav").touch()
        (audio_dir / "b.wav").touch()
        (audio_dir / "c.txt").touch()  # not .wav

        ds = ConcreteTestDataset(data_root=tmp_path)
        files = ds.get_audio_files()
        assert len(files) == 2
        assert all(f.suffix == ".wav" for f in files)

    def test_sorted_order(self, tmp_path):
        audio_dir = tmp_path / "audio" / "test"
        audio_dir.mkdir(parents=True)
        (audio_dir / "z_track.wav").touch()
        (audio_dir / "a_track.wav").touch()

        ds = ConcreteTestDataset(data_root=tmp_path)
        files = ds.get_audio_files()
        assert files[0].stem == "a_track"
        assert files[1].stem == "z_track"

    def test_recursive_nested_audio_discovery(self, tmp_path):
        audio_dir = tmp_path / "audio" / "test"
        (audio_dir / "year_2020").mkdir(parents=True)
        (audio_dir / "year_2021").mkdir(parents=True)
        (audio_dir / "year_2020" / "piece_a.wav").touch()
        (audio_dir / "year_2021" / "piece_b.wav").touch()

        ds = ConcreteTestDataset(data_root=tmp_path)
        files = ds.get_audio_files()

        assert len(files) == 2
        assert [f.stem for f in files] == ["piece_a", "piece_b"]

    def test_case_insensitive_wav_extension(self, tmp_path):
        audio_dir = tmp_path / "audio" / "test"
        audio_dir.mkdir(parents=True)
        (audio_dir / "a.WAV").touch()
        (audio_dir / "b.wav").touch()

        ds = ConcreteTestDataset(data_root=tmp_path)
        files = ds.get_audio_files()

        assert len(files) == 2

    def test_empty_dir(self, tmp_path):
        audio_dir = tmp_path / "audio" / "test"
        audio_dir.mkdir(parents=True)

        ds = ConcreteTestDataset(data_root=tmp_path)
        assert ds.get_audio_files() == []

    def test_nonexistent_dir(self, tmp_path):
        ds = ConcreteTestDataset(data_root=tmp_path)
        assert ds.get_audio_files() == []


class TestGetFeaturePath:
    def test_aggregated(self, tmp_path):
        ds = ConcreteTestDataset(data_root=tmp_path)
        path = ds.get_feature_path("track01", "aggregated")
        expected = tmp_path / "embeddings" / "test-emb" / "aggregated" / "track01.npy"
        assert path == expected

    def test_raw(self, tmp_path):
        ds = ConcreteTestDataset(data_root=tmp_path)
        path = ds.get_feature_path("track01", "raw")
        expected = tmp_path / "embeddings" / "test-emb" / "raw" / "track01.npy"
        assert path == expected

    def test_segments(self, tmp_path):
        ds = ConcreteTestDataset(data_root=tmp_path)
        path = ds.get_feature_path("track01", "segments")
        expected = tmp_path / "embeddings" / "test-emb" / "segments" / "track01.npy"
        assert path == expected


class TestExistsAndLen:
    def test_exists_with_files(self, tmp_path):
        audio_dir = tmp_path / "audio" / "test"
        audio_dir.mkdir(parents=True)
        (audio_dir / "track.wav").touch()

        ds = ConcreteTestDataset(data_root=tmp_path)
        assert ds.exists() is True
        assert len(ds) == 1

    def test_not_exists_empty(self, tmp_path):
        ds = ConcreteTestDataset(data_root=tmp_path)
        assert ds.exists() is False
        assert len(ds) == 0


class TestAggregatedDir:
    def test_property(self, tmp_path):
        ds = ConcreteTestDataset(data_root=tmp_path)
        expected = tmp_path / "embeddings" / "test-emb" / "aggregated"
        assert ds.aggregated_dir == expected
