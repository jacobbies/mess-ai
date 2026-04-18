"""Script lifecycle contract tests for CLI entrypoints."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCRIPT_STATUS_FILE = SCRIPTS_DIR / "script_status.json"
EXPECTED_STATUS_KEYS = {"maintained"}


def _load_script_status() -> dict[str, set[str]]:
    payload: Any = json.loads(SCRIPT_STATUS_FILE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "scripts/script_status.json must contain a JSON object"
    assert EXPECTED_STATUS_KEYS.issubset(set(payload)), (
        f"scripts/script_status.json must include keys {sorted(EXPECTED_STATUS_KEYS)}"
    )

    status: dict[str, set[str]] = {}
    for key in EXPECTED_STATUS_KEYS:
        value = payload[key]
        assert isinstance(value, list), f"scripts/script_status.json '{key}' must be a list"
        names = {str(entry) for entry in value}
        assert names, f"scripts/script_status.json '{key}' cannot be empty"
        assert all(name.endswith(".py") for name in names), (
            f"scripts/script_status.json '{key}' entries must be Python file names"
        )
        status[key] = names
    return status


def _tracked_script_names() -> set[str]:
    """Return tracked script names; fallback to filesystem if git is unavailable."""
    try:
        completed = subprocess.run(
            ["git", "ls-files", "scripts/*.py"],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        names = {
            Path(line.strip()).name
            for line in completed.stdout.splitlines()
            if line.strip()
        }
        if names:
            return names
    except Exception:
        pass
    return {path.name for path in SCRIPTS_DIR.glob("*.py")}


def test_script_status_manifest_covers_all_tracked_scripts() -> None:
    status = _load_script_status()
    maintained = status["maintained"]

    tracked_scripts = _tracked_script_names()
    missing_classification = tracked_scripts - maintained
    assert not missing_classification, (
        "All tracked scripts/*.py files must be classified as maintained. "
        f"Unclassified tracked scripts={sorted(missing_classification)}"
    )


def test_active_scripts_exist() -> None:
    status = _load_script_status()
    for script_name in sorted(status["maintained"]):
        assert (SCRIPTS_DIR / script_name).exists(), f"Missing active script: {script_name}"
