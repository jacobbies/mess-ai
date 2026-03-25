"""README example contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


def _readme_python_api_block() -> str:
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"## Python API \(Minimal\)\n\n```python\n(.*?)\n```", readme_text, re.S)
    assert match, "README Python API example block not found"
    return match.group(1)


def test_readme_clip_search_example_uses_artifact_backed_api() -> None:
    block = _readme_python_api_block()

    assert "from mess.search import find_latest_artifact_dir" in block
    assert (
        'clip_artifact_dir = find_latest_artifact_dir("data/indices", '
        'artifact_name="clip_index")' in block
    )
    assert "clip_results = search_by_clip(" in block
    assert "artifact=clip_artifact_dir" in block
    assert 'track_id="Beethoven_Op027No1-01"' in block
    assert "start_sec=30.0" in block
    assert "query_track=" not in block
    assert "clip_start=" not in block
    assert 'features_dir="data/embeddings/smd-emb/segments"' not in block
