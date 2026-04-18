"""Smoke tests for scripts/setup_demo_data.py."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.workflow]

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "setup_demo_data.py"


def _run_setup_demo(data_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--data-root",
            str(data_root),
            "--duration-seconds",
            "0.25",
            "--sample-rate",
            "8000",
        ],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_setup_demo_data_creates_audio_and_metadata(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    completed = _run_setup_demo(data_root)

    assert "Created  : 3 file(s)" in completed.stdout

    audio_dir = data_root / "audio" / "smd" / "wav-44"
    audio_files = sorted(audio_dir.glob("*.wav"))
    assert len(audio_files) == 3

    metadata_path = data_root / "metadata" / "smd_metadata.csv"
    assert metadata_path.exists()

    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    track_ids = {row["track_id"] for row in rows}
    assert track_ids == {
        "Beethoven_Op027No1-01",
        "Chopin_Op028-15",
        "Bach_BWV849-01",
    }


def test_setup_demo_data_is_idempotent_without_force(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _run_setup_demo(data_root)
    completed = _run_setup_demo(data_root)

    assert "Created  : 0 file(s)" in completed.stdout
    assert "Skipped  : 3 file(s)" in completed.stdout
