"""Tests for datasets package exports and lazy imports."""

from __future__ import annotations

import pytest

import mess.datasets as datasets

pytestmark = pytest.mark.unit


class TestDatasetsInit:
    def test_lazy_import_dataset_classes(self):
        assert datasets.SMDDataset.__name__ == "SMDDataset"
        assert datasets.MAESTRODataset.__name__ == "MAESTRODataset"

    def test_lazy_import_new_metadata_exports(self):
        assert datasets.DatasetMetadataTable.__name__ == "DatasetMetadataTable"
        assert callable(datasets.build_clip_records)
