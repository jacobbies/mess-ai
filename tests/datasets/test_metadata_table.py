"""Tests for canonical dataset metadata table contract."""

from __future__ import annotations

import pytest

from mess.datasets.metadata_table import DatasetMetadataTable

pytestmark = pytest.mark.unit


def _rows() -> list[dict[str, str]]:
    return [
        {
            "track_id": "track_a",
            "recording_id": "recording_a",
            "work_id": "beethoven_op27_no2_mvt1",
            "composer": "Beethoven",
        },
        {
            "track_id": "track_b",
            "recording_id": "recording_b",
            "work_id": "chopin_op10_no3",
            "composer": "Chopin",
        },
    ]


class TestDatasetMetadataTable:
    def test_from_rows_requires_canonical_columns(self):
        with pytest.raises(ValueError, match="missing required columns"):
            DatasetMetadataTable.from_rows([{"track_id": "t"}])

    def test_from_rows_rejects_empty_required_values(self):
        with pytest.raises(ValueError, match="empty required values"):
            DatasetMetadataTable.from_rows(
                [{"track_id": "track_a", "recording_id": "rec_a", "work_id": ""}]
            )

    def test_from_rows_rejects_duplicate_track_ids(self):
        rows = _rows()
        rows.append(
            {
                "track_id": "track_a",
                "recording_id": "recording_c",
                "work_id": "other_work",
            }
        )
        with pytest.raises(ValueError, match="Duplicate track_id"):
            DatasetMetadataTable.from_rows(rows)

    def test_lookup_helpers_return_expected_ids(self):
        table = DatasetMetadataTable.from_rows(_rows())
        assert table.recording_id_for_track("track_a") == "recording_a"
        assert table.work_id_for_track("track_b") == "chopin_op10_no3"
        assert table.recording_id_for_track("missing") is None

    def test_text_for_track_supports_keyword_views(self):
        table = DatasetMetadataTable.from_rows(_rows())
        full = table.text_for_track("track_a")
        assert full is not None
        assert "Beethoven" in full
        focused = table.text_for_track("track_a", fields=["composer", "work_id"])
        assert focused == "Beethoven beethoven_op27_no2_mvt1"
        assert table.text_for_track("missing") is None

    def test_to_csv_and_from_csv_roundtrip(self, tmp_path):
        table = DatasetMetadataTable.from_rows(_rows())
        path = tmp_path / "metadata.csv"
        table.to_csv(path)

        reloaded = DatasetMetadataTable.from_csv(path)
        assert len(reloaded) == 2
        first = reloaded.get_row("track_a")
        assert first is not None
        assert first["composer"] == "Beethoven"

    def test_from_path_rejects_unknown_extension(self, tmp_path):
        path = tmp_path / "metadata.json"
        path.write_text("[]")
        with pytest.raises(ValueError, match="Unsupported metadata table format"):
            DatasetMetadataTable.from_path(path)
