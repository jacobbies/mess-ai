"""Tests for clip-level index contract and split behavior."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from mess.datasets.clip_index import ClipIndex, ClipRecord
from scripts.build_clip_index import build_clip_records

pytestmark = pytest.mark.unit


def _make_records() -> list[ClipRecord]:
    return [
        ClipRecord(
            clip_id="smd:a:00000",
            dataset_id="smd",
            recording_id="rec_a",
            track_id="a",
            segment_idx=0,
            start_sec=0.0,
            end_sec=5.0,
            split="unspecified",
            embedding_path="/tmp/a.npy",
        ),
        ClipRecord(
            clip_id="smd:a:00001",
            dataset_id="smd",
            recording_id="rec_a",
            track_id="a",
            segment_idx=1,
            start_sec=2.5,
            end_sec=7.5,
            split="unspecified",
            embedding_path="/tmp/a.npy",
        ),
        ClipRecord(
            clip_id="smd:b:00000",
            dataset_id="smd",
            recording_id="rec_b",
            track_id="b",
            segment_idx=0,
            start_sec=0.0,
            end_sec=5.0,
            split="unspecified",
            embedding_path="/tmp/b.npy",
        ),
    ]


class TestClipIndex:
    def test_duplicate_clip_id_raises(self):
        record = _make_records()[0]
        with pytest.raises(ValueError, match="Duplicate clip_id"):
            ClipIndex([record, record])

    def test_filter_by_track(self):
        index = ClipIndex(_make_records())
        filtered = index.filter(track_ids={"a"})
        assert len(filtered) == 2
        assert {row.track_id for row in filtered} == {"a"}

    def test_assign_recording_splits_is_recording_consistent(self):
        index = ClipIndex(_make_records())
        split_index = index.assign_recording_splits(
            train_ratio=0.5, val_ratio=0.0, test_ratio=0.5, seed=7
        )
        by_recording = {}
        for row in split_index:
            by_recording.setdefault(row.recording_id, set()).add(row.split)
        assert all(len(splits) == 1 for splits in by_recording.values())
        assert {row.split for row in split_index} <= {"train", "test"}

    def test_to_csv_and_from_csv_roundtrip(self, tmp_path):
        index = ClipIndex(_make_records())
        output = tmp_path / "clip_index.csv"
        index.to_csv(output)

        reloaded = ClipIndex.from_csv(output)
        assert len(reloaded) == len(index)
        assert [r.clip_id for r in reloaded] == [r.clip_id for r in index]

    def test_from_rows_missing_required_column_raises(self):
        rows = [
            {
                "clip_id": "x",
                "dataset_id": "smd",
            }
        ]
        with pytest.raises(ValueError, match="missing required columns"):
            ClipIndex.from_rows(rows)


class TestBuildClipRecords:
    def test_build_clip_records_generates_expected_count(self, tmp_path):
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()
        np.save(segments_dir / "track_a.npy", np.zeros((3, 13, 768), dtype=np.float32))
        np.save(segments_dir / "track_b.npy", np.zeros((2, 13, 768), dtype=np.float32))

        records = build_clip_records("smd", segments_dir, segment_duration=5.0, overlap_ratio=0.5)
        assert len(records) == 5
        assert records[0].clip_id == "smd:track_a:00000"
        assert records[1].start_sec == pytest.approx(2.5)

    def test_build_clip_records_uses_recording_map(self, tmp_path):
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()
        np.save(segments_dir / "track_a.npy", np.zeros((1, 13, 768), dtype=np.float32))

        records = build_clip_records(
            "smd",
            segments_dir,
            recording_map={"track_a": "recording_1"},
        )
        assert records[0].recording_id == "recording_1"


class TestIndexCsvManual:
    def test_from_csv_with_manual_rows(self, tmp_path):
        path = tmp_path / "manual.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "clip_id",
                    "dataset_id",
                    "recording_id",
                    "track_id",
                    "segment_idx",
                    "start_sec",
                    "end_sec",
                    "split",
                    "embedding_path",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "clip_id": "smd:t:00000",
                    "dataset_id": "smd",
                    "recording_id": "rec_t",
                    "track_id": "t",
                    "segment_idx": "0",
                    "start_sec": "0.0",
                    "end_sec": "5.0",
                    "split": "train",
                    "embedding_path": "/tmp/t.npy",
                }
            )

        index = ClipIndex.from_csv(path)
        assert len(index) == 1
        assert index[0].split == "train"
