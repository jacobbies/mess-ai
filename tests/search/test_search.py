"""Tests for mess.search.search — build_index, find_similar with FAISS."""

import numpy as np
import pytest

from mess.search.search import build_index, find_similar


@pytest.fixture
def search_data():
    """5 tracks with 768-dim features and their names."""
    rng = np.random.default_rng(42)
    features = rng.standard_normal((5, 768)).astype(np.float32)
    track_names = [f"track_{i:02d}" for i in range(5)]
    return features, track_names


class TestBuildIndex:
    def test_dimension(self, search_data):
        features, _ = search_data
        index = build_index(features)
        assert index.d == 768

    def test_ntotal(self, search_data):
        features, _ = search_data
        index = build_index(features)
        assert index.ntotal == 5


class TestFindSimilar:
    def test_returns_k_results(self, search_data):
        features, names = search_data
        results = find_similar("track_00", features, names, k=3)
        assert len(results) == 3

    def test_excludes_self(self, search_data):
        features, names = search_data
        results = find_similar("track_00", features, names, k=4)
        result_names = [name for name, _ in results]
        assert "track_00" not in result_names

    def test_includes_self_when_requested(self, search_data):
        features, names = search_data
        results = find_similar(
            "track_00", features, names, k=5, exclude_self=False
        )
        result_names = [name for name, _ in results]
        assert "track_00" in result_names

    def test_unknown_track_raises(self, search_data):
        features, names = search_data
        with pytest.raises(ValueError, match="not found"):
            find_similar("nonexistent", features, names)

    def test_scores_sorted_descending(self, search_data):
        features, names = search_data
        results = find_similar("track_00", features, names, k=4)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_valid_range(self, search_data):
        features, names = search_data
        results = find_similar("track_00", features, names, k=4)
        for _, score in results:
            assert -1.0 <= score <= 1.0 + 1e-6  # small epsilon for float precision
