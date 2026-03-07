"""Tests for hybrid semantic + metadata retrieval."""

from __future__ import annotations

import numpy as np
import pytest

from mess.datasets.metadata_table import DatasetMetadataTable
from mess.search.hybrid import hybrid_search

pytestmark = pytest.mark.unit


def _metadata() -> DatasetMetadataTable:
    return DatasetMetadataTable.from_rows(
        [
            {
                "track_id": "query_track",
                "recording_id": "rec_query",
                "work_id": "work_query",
                "composer": "Bach",
                "title": "Prelude in C",
            },
            {
                "track_id": "semantic_top",
                "recording_id": "rec_a",
                "work_id": "work_a",
                "composer": "Beethoven",
                "title": "Sonata No. 14",
            },
            {
                "track_id": "keyword_match",
                "recording_id": "rec_b",
                "work_id": "work_b",
                "composer": "Chopin",
                "title": "Nocturne in E-flat",
            },
            {
                "track_id": "far_track",
                "recording_id": "rec_c",
                "work_id": "work_c",
                "composer": "Mozart",
                "title": "Symphony",
            },
        ]
    )


def _features() -> tuple[np.ndarray, list[str]]:
    vectors = np.array(
        [
            [1.0, 0.0],      # query_track
            [0.99, 0.01],    # semantic_top
            [0.9, 0.1],      # keyword_match
            [0.0, 1.0],      # far_track
        ],
        dtype=np.float32,
    )
    names = ["query_track", "semantic_top", "keyword_match", "far_track"]
    return vectors, names


def test_keyword_weight_can_reorder_results() -> None:
    features, track_names = _features()
    results = hybrid_search(
        query_track="query_track",
        features=features,
        track_names=track_names,
        metadata=_metadata(),
        keyword="chopin",
        semantic_weight=1.0,
        keyword_weight=2.5,
        k=3,
    )
    assert results[0][0] == "keyword_match"


def test_filters_apply_as_hard_constraints() -> None:
    features, track_names = _features()
    results = hybrid_search(
        query_track="query_track",
        features=features,
        track_names=track_names,
        metadata=_metadata(),
        filters={"composer": "Beethoven"},
        k=5,
    )
    assert [track for track, _ in results] == ["semantic_top"]


def test_no_keyword_defaults_to_semantic_order() -> None:
    features, track_names = _features()
    results = hybrid_search(
        query_track="query_track",
        features=features,
        track_names=track_names,
        metadata=_metadata(),
        keyword=None,
        k=3,
    )
    assert [track for track, _ in results] == ["semantic_top", "keyword_match", "far_track"]


def test_zero_effective_weights_raises() -> None:
    features, track_names = _features()
    with pytest.raises(ValueError, match="effective weight"):
        hybrid_search(
            query_track="query_track",
            features=features,
            track_names=track_names,
            metadata=_metadata(),
            semantic_weight=0.0,
            keyword_weight=0.0,
        )
