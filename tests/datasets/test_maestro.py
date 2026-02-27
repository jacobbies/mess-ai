"""Tests for mess.datasets.maestro.MAESTRODataset."""

import pytest

from mess.datasets.maestro import MAESTRODataset

pytestmark = pytest.mark.unit


class TestMAESTRODataset:
    def test_dataset_id(self, tmp_path):
        ds = MAESTRODataset(data_root=tmp_path)
        assert ds.dataset_id == "maestro"

    def test_audio_dir(self, tmp_path):
        ds = MAESTRODataset(data_root=tmp_path)
        assert ds.audio_dir == tmp_path / "audio" / "maestro"

    def test_embeddings_dir(self, tmp_path):
        ds = MAESTRODataset(data_root=tmp_path)
        assert ds.embeddings_dir == tmp_path / "embeddings" / "maestro-emb"

    def test_name(self, tmp_path):
        ds = MAESTRODataset(data_root=tmp_path)
        assert "MAESTRO" in ds.name
