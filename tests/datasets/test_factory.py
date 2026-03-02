"""Tests for mess.datasets.factory.DatasetFactory."""

import pytest

from mess.datasets.base import BaseDataset
from mess.datasets.factory import DatasetFactory
from mess.datasets.maestro import MAESTRODataset
from mess.datasets.smd import SMDDataset

pytestmark = pytest.mark.unit


class TestGetDataset:
    def test_smd_returns_smd_dataset(self, tmp_path):
        ds = DatasetFactory.get_dataset("smd", data_root=tmp_path)
        assert isinstance(ds, SMDDataset)

    def test_maestro_returns_maestro_dataset(self, tmp_path):
        ds = DatasetFactory.get_dataset("maestro", data_root=tmp_path)
        assert isinstance(ds, MAESTRODataset)

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            DatasetFactory.get_dataset("invalid")


class TestGetAvailableDatasets:
    def test_returns_smd_and_maestro(self):
        available = DatasetFactory.get_available_datasets()
        assert "smd" in available
        assert "maestro" in available


class TestRegisterDataset:
    def test_register_custom(self, tmp_path):
        class CustomDataset(BaseDataset):
            @property
            def dataset_id(self):
                return "custom"

            @property
            def audio_dir(self):
                return self.data_root / "audio" / "custom"

            @property
            def embeddings_dir(self):
                return self.data_root / "embeddings" / "custom-emb"

            @property
            def name(self):
                return "Custom"

            @property
            def description(self):
                return "Test dataset"

        DatasetFactory.register_dataset("custom", CustomDataset)
        ds = DatasetFactory.get_dataset("custom", data_root=tmp_path)
        assert isinstance(ds, CustomDataset)
        assert ds.dataset_id == "custom"


class TestCreateDataset:
    def test_alias_for_get_dataset(self, tmp_path):
        ds = DatasetFactory.create_dataset("smd", data_root=tmp_path)
        assert isinstance(ds, SMDDataset)


class TestCustomDataRoot:
    def test_passthrough(self, tmp_path):
        ds = DatasetFactory.get_dataset("smd", data_root=tmp_path)
        assert ds.data_root == tmp_path
