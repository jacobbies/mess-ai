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
EXPECTED_STATUS_KEYS = {"maintained", "research", "deprecated"}


def _load_script_status() -> dict[str, set[str]]:
    payload: Any = json.loads(SCRIPT_STATUS_FILE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "scripts/script_status.json must contain a JSON object"
    assert set(payload) == EXPECTED_STATUS_KEYS, (
        f"scripts/script_status.json keys must be {sorted(EXPECTED_STATUS_KEYS)}"
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
    research = status["research"]
    deprecated = status["deprecated"]

    assert maintained.isdisjoint(research)
    assert maintained.isdisjoint(deprecated)
    assert research.isdisjoint(deprecated)

    tracked_scripts = _tracked_script_names()
    classified_active = maintained | research
    missing_classification = tracked_scripts - classified_active
    assert not missing_classification, (
        "All tracked scripts/*.py files must be classified as maintained or research. "
        f"Unclassified tracked scripts={sorted(missing_classification)}"
    )


def test_active_scripts_exist() -> None:
    status = _load_script_status()
    for script_name in sorted(status["maintained"] | status["research"]):
        assert (SCRIPTS_DIR / script_name).exists(), f"Missing active script: {script_name}"


def test_deprecated_scripts_are_removed_and_documented() -> None:
    status = _load_script_status()
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    status_manifest_text = SCRIPT_STATUS_FILE.read_text(encoding="utf-8")
    status_text = (SCRIPTS_DIR / "_NEEDS_UPDATE.txt").read_text(encoding="utf-8")

    assert "No pending outdated scripts" in status_text
    assert "scripts/script_status.json" in readme_text
    for script_name in sorted(status["deprecated"]):
        assert not (SCRIPTS_DIR / script_name).exists(), (
            f"Deprecated script still exists: {script_name}"
        )
        assert script_name in readme_text
        assert script_name in status_manifest_text
        assert script_name in status_text
