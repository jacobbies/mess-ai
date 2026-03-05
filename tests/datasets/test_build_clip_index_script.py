"""Tests for script helpers in scripts/build_clip_index.py."""

from __future__ import annotations

import csv

import pytest

from scripts.build_clip_index import _load_id_maps

pytestmark = pytest.mark.unit


def test_load_id_maps_strips_whitespace_and_ignores_blank_track_ids(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["track_id", "recording_id", "work_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "track_id": "  track_a  ",
                "recording_id": "  recording_1  ",
                "work_id": "  work_1  ",
            }
        )
        writer.writerow(
            {
                "track_id": "   ",
                "recording_id": "recording_ignored",
                "work_id": "work_ignored",
            }
        )

    recording_map, work_map = _load_id_maps(
        manifest_path=manifest,
        track_col="track_id",
        recording_col="recording_id",
        work_col="work_id",
    )

    assert recording_map == {"track_a": "recording_1"}
    assert work_map == {"track_a": "work_1"}
