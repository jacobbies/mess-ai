"""Tests for mess.extraction.storage — path helpers and save/load roundtrip."""

import numpy as np
import pytest

from mess.extraction.storage import (
    _resolve_base_dir,
    _resolve_filename,
    features_exist,
    features_exist_for_types,
    load_features,
    load_selected_features,
    save_features,
)

pytestmark = pytest.mark.integration


class TestResolveBaseDir:
    def test_no_dataset(self, tmp_path):
        assert _resolve_base_dir(tmp_path) == tmp_path

    def test_with_dataset(self, tmp_path):
        assert _resolve_base_dir(tmp_path, "smd") == tmp_path / "smd"


class TestResolveFilename:
    def test_with_track_id(self):
        assert _resolve_filename("/some/path.wav", track_id="my_track") == "my_track"

    def test_from_path(self):
        assert _resolve_filename("/data/audio/Beethoven_Op27.wav") == "Beethoven_Op27"

    def test_from_path_no_track_id(self):
        assert _resolve_filename("track.wav") == "track"


class TestFeaturesExist:
    def test_returns_true_when_exists(self, tmp_path):
        agg_dir = tmp_path / "aggregated"
        agg_dir.mkdir()
        (agg_dir / "track01.npy").write_bytes(b"dummy")
        assert features_exist("track01.wav", tmp_path, track_id="track01") is True

    def test_returns_false_when_missing(self, tmp_path):
        assert features_exist("track01.wav", tmp_path, track_id="track01") is False

    def test_returns_false_for_empty_output_dir(self):
        assert features_exist("track01.wav", "") is False


class TestFeaturesExistForTypes:
    def test_all_present(self, tmp_path):
        for t in ["raw", "segments", "aggregated"]:
            d = tmp_path / t
            d.mkdir()
            (d / "track.npy").write_bytes(b"dummy")

        assert features_exist_for_types(
            "track.wav", tmp_path, ["raw", "segments", "aggregated"], track_id="track"
        )

    def test_partial_missing(self, tmp_path):
        d = tmp_path / "raw"
        d.mkdir()
        (d / "track.npy").write_bytes(b"dummy")

        assert not features_exist_for_types(
            "track.wav", tmp_path, ["raw", "segments"], track_id="track"
        )

    def test_empty_output_dir(self):
        assert not features_exist_for_types("track.wav", "", ["aggregated"])


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, tmp_path):
        rng = np.random.default_rng(99)
        features = {
            "raw": rng.standard_normal((3, 13, 50, 768)).astype(np.float32),
            "segments": rng.standard_normal((3, 13, 768)).astype(np.float32),
            "aggregated": rng.standard_normal((13, 768)).astype(np.float32),
        }

        save_features(features, "test_track.wav", tmp_path, track_id="test_track")
        loaded = load_features("test_track.wav", tmp_path, track_id="test_track")

        assert loaded is not None
        for key in features:
            np.testing.assert_array_equal(loaded[key], features[key])

    def test_save_creates_subdirectories(self, tmp_path):
        features = {
            "raw": np.zeros((1,)),
            "segments": np.zeros((1,)),
            "aggregated": np.zeros((1,)),
        }
        save_features(features, "t.wav", tmp_path, track_id="t")

        assert (tmp_path / "raw").is_dir()
        assert (tmp_path / "segments").is_dir()
        assert (tmp_path / "aggregated").is_dir()

    def test_load_returns_none_when_missing(self, tmp_path):
        result = load_features("missing.wav", tmp_path, track_id="missing")
        assert result is None


class TestLoadSelectedFeatures:
    def test_loads_subset(self, tmp_path):
        rng = np.random.default_rng(42)
        features = {
            "raw": rng.standard_normal((2, 768)).astype(np.float32),
            "segments": rng.standard_normal((2, 768)).astype(np.float32),
            "aggregated": rng.standard_normal((768,)).astype(np.float32),
        }
        save_features(features, "t.wav", tmp_path, track_id="t")

        loaded = load_selected_features(
            "t.wav", tmp_path, ["aggregated"], track_id="t"
        )
        assert loaded is not None
        assert "aggregated" in loaded
        assert "raw" not in loaded

    def test_returns_none_when_type_missing(self, tmp_path):
        loaded = load_selected_features(
            "t.wav", tmp_path, ["aggregated"], track_id="t"
        )
        assert loaded is None
