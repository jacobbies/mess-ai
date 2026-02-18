"""Tests for mess.datasets.smd.SMDDataset."""

from mess.datasets.smd import SMDDataset


class TestSMDDataset:
    def test_dataset_id(self, tmp_path):
        ds = SMDDataset(data_root=tmp_path)
        assert ds.dataset_id == "smd"

    def test_audio_dir(self, tmp_path):
        ds = SMDDataset(data_root=tmp_path)
        assert ds.audio_dir == tmp_path / "audio" / "smd" / "wav-44"

    def test_embeddings_dir(self, tmp_path):
        ds = SMDDataset(data_root=tmp_path)
        assert ds.embeddings_dir == tmp_path / "embeddings" / "smd-emb"

    def test_name(self, tmp_path):
        ds = SMDDataset(data_root=tmp_path)
        assert "SMD" in ds.name
