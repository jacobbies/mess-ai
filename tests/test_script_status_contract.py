"""Script status contract tests for maintained vs retired CLI entrypoints."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

MAINTAINED_SCRIPTS = {
    "build_clip_index.py",
    "extract_features.py",
    "run_probing.py",
    "demo_recommendations.py",
    "publish_faiss_index.py",
    "train_retrieval_ssl.py",
}

RETIRED_SCRIPTS = {
    "build_layer_indices.py",
    "demo_layer_search.py",
    "evaluate_layer_indices.py",
    "evaluate_similarity.py",
}


def test_maintained_scripts_exist() -> None:
    for script_name in MAINTAINED_SCRIPTS:
        assert (SCRIPTS_DIR / script_name).exists(), f"Missing maintained script: {script_name}"


def test_retired_scripts_are_removed_and_documented() -> None:
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    status_text = (SCRIPTS_DIR / "_NEEDS_UPDATE.txt").read_text(encoding="utf-8")

    assert "No pending outdated scripts" in status_text
    for script_name in RETIRED_SCRIPTS:
        assert not (SCRIPTS_DIR / script_name).exists(), (
            f"Retired script still exists: {script_name}"
        )
        assert script_name in readme_text
        assert script_name in status_text
