"""Tests for search package exports and lazy imports."""

from __future__ import annotations

import pytest

import mess.search as search

pytestmark = pytest.mark.unit


def test_lazy_import_core_search_exports() -> None:
    assert callable(search.find_similar)
    assert callable(search.search_by_clip)


def test_lazy_import_artifact_exports() -> None:
    assert search.FAISSArtifact.__name__ == "FAISSArtifact"
    assert search.ArtifactManifest.__name__ == "ArtifactManifest"


def test_public_api_contract_names() -> None:
    expected = {
        "find_similar",
        "load_features",
        "load_segment_features",
        "search_by_clip",
        "build_track_artifact",
        "load_latest_from_s3",
    }
    assert expected.issubset(set(search.__all__))
    assert expected.issubset(set(dir(search)))
