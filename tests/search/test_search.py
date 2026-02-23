"""Tests for mess.search.search — loading + FAISS search behavior."""

import numpy as np
import pytest

from mess.search.search import (
    ClipSearchResult,
    _require_faiss,
    build_index,
    find_similar,
    load_features,
    load_segment_features,
    search_by_aspects,
    search_by_clip,
)


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


class TestLoadFeatures:
    def test_aggregated_features_flatten_to_one_row_per_track(self, tmp_path):
        arr1 = np.random.default_rng(1).standard_normal((13, 768)).astype(np.float32)
        arr2 = np.random.default_rng(2).standard_normal((13, 768)).astype(np.float32)
        np.save(tmp_path / "track_b.npy", arr1)
        np.save(tmp_path / "track_a.npy", arr2)

        features, track_names = load_features(str(tmp_path))

        assert track_names == ["track_a", "track_b"]  # sorted by filename
        assert features.shape == (2, 13 * 768)

    def test_layer_extraction_returns_768d_per_track(self, tmp_path):
        arr1 = np.random.default_rng(1).standard_normal((13, 768)).astype(np.float32)
        arr2 = np.random.default_rng(2).standard_normal((13, 768)).astype(np.float32)
        np.save(tmp_path / "track_1.npy", arr1)
        np.save(tmp_path / "track_2.npy", arr2)

        features, track_names = load_features(str(tmp_path), layer=7)

        assert track_names == ["track_1", "track_2"]
        assert features.shape == (2, 768)
        np.testing.assert_allclose(features[0], arr1[7], atol=1e-6)

    def test_segments_features_pool_to_one_row_per_track(self, tmp_path):
        segments = np.random.default_rng(3).standard_normal((4, 13, 768)).astype(np.float32)
        np.save(tmp_path / "track_segments.npy", segments)

        features, track_names = load_features(str(tmp_path))

        assert track_names == ["track_segments"]
        assert features.shape == (1, 13 * 768)
        expected = segments.mean(axis=0).reshape(-1)
        np.testing.assert_allclose(features[0], expected, atol=1e-6)

    def test_invalid_layer_raises(self, tmp_path):
        arr = np.random.default_rng(4).standard_normal((13, 768)).astype(np.float32)
        np.save(tmp_path / "track.npy", arr)

        with pytest.raises(ValueError, match="Invalid layer"):
            load_features(str(tmp_path), layer=13)

    def test_invalid_shape_raises(self, tmp_path):
        bad = np.random.default_rng(5).standard_normal((2, 768)).astype(np.float32)
        np.save(tmp_path / "bad_track.npy", bad)

        with pytest.raises(ValueError, match="Unsupported 2D shape"):
            load_features(str(tmp_path))


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


class TestClipLevelSearch:
    @staticmethod
    def _make_segments(vectors):
        arr = np.zeros((len(vectors), 13, 768), dtype=np.float32)
        for i, values in enumerate(vectors):
            arr[i, 0, : len(values)] = np.asarray(values, dtype=np.float32)
        return arr

    def test_load_segment_features_returns_timestamps(self, tmp_path):
        np.save(
            tmp_path / "track_x.npy",
            self._make_segments([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        )

        features, clip_locations = load_segment_features(str(tmp_path), layer=0)

        assert features.shape == (3, 768)
        assert len(clip_locations) == 3
        assert clip_locations[0].track_id == "track_x"
        assert clip_locations[0].segment_idx == 0
        assert clip_locations[0].start_time == pytest.approx(0.0)
        assert clip_locations[0].end_time == pytest.approx(5.0)
        assert clip_locations[1].start_time == pytest.approx(2.5)
        assert clip_locations[2].start_time == pytest.approx(5.0)

    def test_search_by_clip_returns_timestamped_results(self, tmp_path):
        np.save(
            tmp_path / "query_track.npy",
            self._make_segments(
                [[1.0, 0.0], [0.0, 1.0], [0.98, 0.05], [0.2, 0.9]]
            ),
        )
        np.save(
            tmp_path / "candidate_track.npy",
            self._make_segments(
                [[0.95, 0.1], [0.97, 0.04], [0.0, 1.0], [-1.0, 0.0]]
            ),
        )

        results = search_by_clip(
            query_track="query_track",
            clip_start=5.0,
            features_dir=str(tmp_path),
            layer=0,
            k=3,
            dedupe_window_seconds=0.0,
        )

        assert len(results) == 3
        assert all(isinstance(item, ClipSearchResult) for item in results)
        top = results[0]
        assert top.track_id == "candidate_track"
        assert top.start_time == pytest.approx(2.5)
        assert top.end_time == pytest.approx(7.5)

    def test_search_by_clip_dedupes_nearby_regions_per_track(self, tmp_path):
        np.save(
            tmp_path / "query_track.npy",
            self._make_segments([[1.0, 0.0], [0.95, 0.05], [0.9, 0.1], [0.0, 1.0]]),
        )
        np.save(
            tmp_path / "candidate_track.npy",
            self._make_segments(
                [[0.99, 0.01], [0.98, 0.02], [0.97, 0.03], [-1.0, 0.0]]
            ),
        )

        results = search_by_clip(
            query_track="query_track",
            clip_start=0.0,
            features_dir=str(tmp_path),
            layer=0,
            k=4,
            dedupe_window_seconds=5.0,
        )

        candidate_starts = sorted(
            result.start_time for result in results if result.track_id == "candidate_track"
        )
        assert candidate_starts
        for i in range(1, len(candidate_starts)):
            assert candidate_starts[i] - candidate_starts[i - 1] >= 5.0

    def test_search_by_clip_unknown_track_raises(self, tmp_path):
        np.save(tmp_path / "track_a.npy", self._make_segments([[1.0, 0.0], [0.0, 1.0]]))

        with pytest.raises(ValueError, match="not found"):
            search_by_clip(
                query_track="missing",
                clip_start=0.0,
                features_dir=str(tmp_path),
                layer=0,
            )


class TestSearchByAspects:
    def test_prefers_layer_matched_track(self, tmp_path, monkeypatch):
        def make_track(layer0, layer1):
            arr = np.zeros((13, 768), dtype=np.float32)
            arr[0, :2] = np.asarray(layer0, dtype=np.float32)
            arr[1, :2] = np.asarray(layer1, dtype=np.float32)
            return arr

        np.save(tmp_path / "query.npy", make_track([1.0, 0.0], [0.0, 1.0]))
        np.save(tmp_path / "candidate_l0.npy", make_track([0.95, 0.05], [1.0, 0.0]))
        np.save(tmp_path / "candidate_l1.npy", make_track([0.0, 1.0], [0.05, 0.95]))
        np.save(tmp_path / "candidate_mix.npy", make_track([0.8, 0.2], [0.2, 0.8]))

        def fake_resolve_aspects(min_r2=0.5):
            return {
                "brightness": {"layer": 0, "r2_score": 1.0, "confidence": "high"},
                "phrasing": {"layer": 1, "r2_score": 1.0, "confidence": "high"},
            }

        monkeypatch.setattr("mess.probing.resolve_aspects", fake_resolve_aspects)

        layer0_results = search_by_aspects(
            query_track="query",
            aspect_weights={"brightness": 1.0},
            features_dir=str(tmp_path),
            k=3,
        )
        assert layer0_results[0][0] == "candidate_l0"

        layer1_results = search_by_aspects(
            query_track="query",
            aspect_weights={"phrasing": 1.0},
            features_dir=str(tmp_path),
            k=3,
        )
        assert layer1_results[0][0] == "candidate_l1"

        mixed_results = search_by_aspects(
            query_track="query",
            aspect_weights={"brightness": 1.0, "phrasing": 1.0},
            features_dir=str(tmp_path),
            k=3,
        )
        assert mixed_results[0][0] == "candidate_mix"

    def test_rejects_unknown_aspects(self, tmp_path, monkeypatch):
        arr = np.zeros((13, 768), dtype=np.float32)
        np.save(tmp_path / "query.npy", arr)
        np.save(tmp_path / "candidate.npy", arr)

        monkeypatch.setattr(
            "mess.probing.resolve_aspects",
            lambda min_r2=0.5: {
                "brightness": {"layer": 0, "r2_score": 0.9, "confidence": "high"}
            },
        )

        with pytest.raises(ValueError, match="not validated"):
            search_by_aspects(
                query_track="query",
                aspect_weights={"nonexistent_aspect": 1.0},
                features_dir=str(tmp_path),
            )


class TestFaissDependency:
    def test_require_faiss_reports_install_hint(self, monkeypatch):
        def missing_faiss(_name):
            raise ModuleNotFoundError("No module named 'faiss'")

        monkeypatch.setattr("mess.search.search.importlib.import_module", missing_faiss)

        with pytest.raises(ModuleNotFoundError, match="pip install faiss-cpu"):
            _require_faiss()

    def test_load_features_does_not_require_faiss(self, tmp_path, monkeypatch):
        arr = np.random.default_rng(7).standard_normal((13, 768)).astype(np.float32)
        np.save(tmp_path / "track.npy", arr)

        def missing_faiss(_name):
            raise ModuleNotFoundError("No module named 'faiss'")

        monkeypatch.setattr("mess.search.search.importlib.import_module", missing_faiss)

        features, track_names = load_features(str(tmp_path))
        assert track_names == ["track"]
        assert features.shape == (1, 13 * 768)
