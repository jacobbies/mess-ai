"""Tests for mess.datasets.base.BaseDataset."""

from pathlib import Path

import numpy as np
import pytest

from mess.datasets.base import BaseDataset
from mess.datasets.metadata_table import DatasetMetadataTable

pytestmark = pytest.mark.unit


class ConcreteTestDataset(BaseDataset):
    """Concrete subclass for testing declarative BaseDataset definitions."""

    dataset_id = "test"
    audio_subdir = "audio/test"
    embeddings_subdir = "embeddings/test-emb"
    name = "Test Dataset"
    description = "A test dataset"


class LegacyPropertyDataset(BaseDataset):
    """Legacy property-based subclasses remain supported."""

    @property
    def dataset_id(self):
        return "legacy"

    @property
    def audio_dir(self):
        return self.data_root / "legacy-audio"

    @property
    def embeddings_dir(self):
        return self.data_root / "legacy-embeddings"


class TestAbstract:
    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError):
            BaseDataset(data_root=Path("/tmp"))

    def test_supports_legacy_property_overrides(self, tmp_path):
        ds = LegacyPropertyDataset(data_root=tmp_path)
        assert ds.dataset_id == "legacy"
        assert ds.audio_dir == tmp_path / "legacy-audio"
        assert ds.embeddings_dir == tmp_path / "legacy-embeddings"


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


class TestDatasetArtifactPaths:
    def test_paths(self, tmp_path):
        ds = ConcreteTestDataset(data_root=tmp_path)
        assert ds.segments_dir == tmp_path / "embeddings" / "test-emb" / "segments"
        assert ds.metadata_dir == tmp_path / "metadata"
        assert ds.metadata_table_path == tmp_path / "metadata" / "test_metadata.csv"
        assert ds.clip_index_path == tmp_path / "metadata" / "test_clip_index.csv"


class TestMetadataLoading:
    def test_load_metadata_table_returns_none_if_missing(self, tmp_path):
        ds = ConcreteTestDataset(data_root=tmp_path)
        assert ds.load_metadata_table() is None

    def test_load_metadata_table_required_raises_if_missing(self, tmp_path):
        ds = ConcreteTestDataset(data_root=tmp_path)
        with pytest.raises(FileNotFoundError, match="Metadata table not found"):
            ds.load_metadata_table(required=True)

    def test_load_metadata_table_from_default_path(self, tmp_path):
        ds = ConcreteTestDataset(data_root=tmp_path)
        ds.metadata_dir.mkdir(parents=True, exist_ok=True)
        table = DatasetMetadataTable.from_rows(
            [
                {
                    "track_id": "track_a",
                    "recording_id": "recording_a",
                    "work_id": "work_a",
                }
            ]
        )
        table.to_csv(ds.metadata_table_path)

        loaded = ds.load_metadata_table()
        assert loaded is not None
        assert loaded.recording_id_for_track("track_a") == "recording_a"


class TestClipIndexHelpers:
    def test_build_clip_index_uses_metadata_table_when_available(self, tmp_path):
        ds = ConcreteTestDataset(data_root=tmp_path)
        ds.segments_dir.mkdir(parents=True, exist_ok=True)
        np.save(ds.segments_dir / "track_a.npy", np.zeros((2, 13, 768), dtype=np.float32))

        ds.metadata_dir.mkdir(parents=True, exist_ok=True)
        DatasetMetadataTable.from_rows(
            [
                {
                    "track_id": "track_a",
                    "recording_id": "recording_1",
                    "work_id": "work_1",
                }
            ]
        ).to_csv(ds.metadata_table_path)

        index = ds.build_clip_index()
        assert len(index) == 2
        assert index[0].recording_id == "recording_1"
        assert index[0].work_id == "work_1"

    def test_build_clip_index_falls_back_to_track_id_without_metadata(self, tmp_path):
        ds = ConcreteTestDataset(data_root=tmp_path)
        ds.segments_dir.mkdir(parents=True, exist_ok=True)
        np.save(ds.segments_dir / "track_a.npy", np.zeros((1, 13, 768), dtype=np.float32))

        index = ds.build_clip_index()
        assert index[0].recording_id == "track_a"
        assert index[0].work_id == ""

    def test_build_clip_index_with_missing_explicit_metadata_path_raises(self, tmp_path):
        ds = ConcreteTestDataset(data_root=tmp_path)
        ds.segments_dir.mkdir(parents=True, exist_ok=True)
        np.save(ds.segments_dir / "track_a.npy", np.zeros((1, 13, 768), dtype=np.float32))

        with pytest.raises(FileNotFoundError, match="Metadata table not found"):
            ds.build_clip_index(metadata_path=tmp_path / "metadata" / "missing.csv")

    def test_save_and_load_clip_index_roundtrip(self, tmp_path):
        ds = ConcreteTestDataset(data_root=tmp_path)
        ds.segments_dir.mkdir(parents=True, exist_ok=True)
        np.save(ds.segments_dir / "track_a.npy", np.zeros((1, 13, 768), dtype=np.float32))

        index = ds.build_clip_index()
        saved_path = ds.save_clip_index(index)
        reloaded = ds.load_clip_index(saved_path)

        assert saved_path == ds.clip_index_path
        assert len(reloaded) == 1
        assert reloaded[0].clip_id == index[0].clip_id
