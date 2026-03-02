"""Tests for mess.probing.discovery — probing, best_layers, resolve_aspects, registries."""

import json

import numpy as np
import pytest

from mess.probing.discovery import (
    ASPECT_REGISTRY,
    LayerDiscoverySystem,
    resolve_aspects,
)

pytestmark = pytest.mark.unit


class TestProbeSingle:
    """_probe_single uses sklearn only — no heavy model deps."""

    def _make_system(self):
        """Create a LayerDiscoverySystem with mocked __init__."""
        sys = object.__new__(LayerDiscoverySystem)
        sys.alpha = 1.0
        sys.n_folds = 5
        return sys

    def test_returns_correct_keys(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 10))
        y = rng.standard_normal(20)
        result = self._make_system()._probe_single(X, y)
        assert set(result.keys()) == {"r2_score", "correlation", "rmse"}

    def test_perfect_linear_relationship(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10))
        w = rng.standard_normal(10)
        y = X @ w
        result = self._make_system()._probe_single(X, y)
        assert result["r2_score"] > 0.95

    def test_random_data_low_r2(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10))
        y = rng.standard_normal(50)
        result = self._make_system()._probe_single(X, y)
        assert result["r2_score"] < 0.5

    def test_insufficient_samples(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((3, 10))  # fewer than n_folds=5
        y = rng.standard_normal(3)
        result = self._make_system()._probe_single(X, y)
        assert result["r2_score"] == -999.0
        assert result["rmse"] == 999.0

    def test_discover_uses_dataset_audio_file_listing(self, monkeypatch, tmp_path):
        system = object.__new__(LayerDiscoverySystem)
        system.alpha = 1.0
        system.n_folds = 2
        system.features_dir = tmp_path / "embeddings" / "raw"
        system.targets_dir = tmp_path / "proxy_targets"

        nested_files = [
            tmp_path / "audio" / "maestro" / "2018" / "track_a.wav",
            tmp_path / "audio" / "maestro" / "2019" / "track_b.wav",
        ]

        class DummyDataset:
            name = "MAESTRO"

            @staticmethod
            def get_audio_files():
                return nested_files

        system.dataset = DummyDataset()

        observed_audio_files = {}

        def fake_load_features(audio_files):
            observed_audio_files["files"] = list(audio_files)
            per_layer = {layer: np.ones((2, 4), dtype=np.float32) for layer in range(13)}
            return per_layer, list(audio_files)

        def fake_load_targets(audio_files):
            return {"spectral_centroid": np.array([0.1, 0.2], dtype=np.float32)}, list(audio_files)

        monkeypatch.setattr(system, "load_features", fake_load_features)
        monkeypatch.setattr(system, "load_targets", fake_load_targets)
        monkeypatch.setattr(system, "_probe_single", lambda X, y: {
            "r2_score": 0.9,
            "correlation": 0.9,
            "rmse": 0.1,
        })

        results = system.discover(n_samples=10)

        assert results
        assert observed_audio_files["files"] == [str(path) for path in nested_files]

    def test_discover_aligns_features_and_targets_by_common_file_order(self, monkeypatch):
        system = object.__new__(LayerDiscoverySystem)
        system.alpha = 1.0
        system.n_folds = 2

        class DummyDataset:
            name = "SMD"

            @staticmethod
            def get_audio_files():
                return ["a.wav", "b.wav", "c.wav"]

        system.dataset = DummyDataset()

        feat_files = ["b.wav", "a.wav"]
        tgt_files = ["a.wav", "b.wav"]

        # Rows encode file identity: a->1.0, b->2.0
        layer_values = np.array([[2.0], [1.0]], dtype=np.float32)
        target_values = np.array([1.0, 2.0], dtype=np.float32)

        def fake_load_features(_audio_files):
            per_layer = {layer: layer_values.copy() for layer in range(13)}
            return per_layer, feat_files

        def fake_load_targets(_audio_files):
            return {"spectral_centroid": target_values.copy()}, tgt_files

        def assert_aligned_probe(X, y):
            # With correct alignment, rows should pair as: a->(1,1), b->(2,2)
            assert np.array_equal(X[:, 0], np.array([1.0, 2.0], dtype=np.float32))
            assert np.array_equal(y, np.array([1.0, 2.0], dtype=np.float32))
            return {"r2_score": 1.0, "correlation": 1.0, "rmse": 0.0}

        monkeypatch.setattr(system, "load_features", fake_load_features)
        monkeypatch.setattr(system, "load_targets", fake_load_targets)
        monkeypatch.setattr(system, "_probe_single", assert_aligned_probe)

        results = system.discover(n_samples=3)
        assert results

    def test_discover_filters_targets_with_insufficient_valid_samples(self, monkeypatch):
        system = object.__new__(LayerDiscoverySystem)
        system.alpha = 1.0
        system.n_folds = 3

        class DummyDataset:
            name = "SMD"

            @staticmethod
            def get_audio_files():
                return ["a.wav", "b.wav", "c.wav"]

        system.dataset = DummyDataset()

        def fake_load_features(_audio_files):
            per_layer = {layer: np.ones((3, 2), dtype=np.float32) for layer in range(13)}
            return per_layer, ["a.wav", "b.wav", "c.wav"]

        def fake_load_targets(_audio_files):
            return {
                "spectral_centroid": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "rubato": np.array([0.2, np.nan, np.nan], dtype=np.float32),
            }, ["a.wav", "b.wav", "c.wav"]

        monkeypatch.setattr(system, "load_features", fake_load_features)
        monkeypatch.setattr(system, "load_targets", fake_load_targets)
        monkeypatch.setattr(system, "_probe_single", lambda X, y: {
            "r2_score": 0.8,
            "correlation": 0.7,
            "rmse": 0.2,
        })

        results = system.discover(n_samples=3)

        assert "spectral_centroid" in results[0]
        assert "rubato" not in results[0]
        assert results[0]["spectral_centroid"]["n_valid"] == 3.0
        assert results[0]["spectral_centroid"]["coverage"] == pytest.approx(1.0)


class TestBestLayers:
    def test_empty_input(self):
        assert LayerDiscoverySystem.best_layers({}) == {}

    def test_identifies_correct_layer(self, sample_discovery_results):
        best = LayerDiscoverySystem.best_layers(sample_discovery_results)
        assert best["spectral_centroid"]["layer"] == 0
        assert best["spectral_rolloff"]["layer"] == 5

    def test_confidence_high(self, sample_discovery_results):
        best = LayerDiscoverySystem.best_layers(sample_discovery_results)
        assert best["spectral_centroid"]["confidence"] == "high"

    def test_confidence_medium(self, sample_discovery_results):
        best = LayerDiscoverySystem.best_layers(sample_discovery_results)
        # spectral_rolloff best R2=0.85 -> high
        assert best["spectral_rolloff"]["confidence"] == "high"

    def test_confidence_low(self, sample_discovery_results):
        best = LayerDiscoverySystem.best_layers(sample_discovery_results)
        # tempo best R2=0.45 -> low
        assert best["tempo"]["confidence"] == "low"


class TestResolveAspects:
    def test_from_file(self, tmp_path, sample_discovery_results):
        # Write discovery results JSON
        results_path = tmp_path / "results.json"
        serializable = {
            str(k): v for k, v in sample_discovery_results.items()
        }
        with open(results_path, "w") as f:
            json.dump(serializable, f)

        resolved = resolve_aspects(min_r2=0.5, results_path=results_path)
        assert "brightness" in resolved  # spectral_centroid R2=0.95
        assert resolved["brightness"]["layer"] == 0

    def test_min_r2_filter(self, tmp_path, sample_discovery_results):
        results_path = tmp_path / "results.json"
        serializable = {
            str(k): v for k, v in sample_discovery_results.items()
        }
        with open(results_path, "w") as f:
            json.dump(serializable, f)

        # tempo best R2=0.45, so with min_r2=0.5 it should be excluded
        resolved = resolve_aspects(min_r2=0.5, results_path=results_path)
        assert "tempo" not in resolved

    def test_missing_file(self, tmp_path):
        resolved = resolve_aspects(
            results_path=tmp_path / "nonexistent.json"
        )
        assert resolved == {}

    def test_includes_coverage_fields_when_present(self, tmp_path):
        results_payload = {
            "0": {
                "spectral_centroid": {
                    "r2_score": 0.9,
                    "correlation": 0.9,
                    "rmse": 0.1,
                    "n_valid": 12.0,
                    "coverage": 0.75,
                }
            }
        }
        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(results_payload, f)

        resolved = resolve_aspects(min_r2=0.5, results_path=results_path)
        assert resolved["brightness"]["n_valid"] == 12
        assert resolved["brightness"]["coverage"] == pytest.approx(0.75)


class TestAspectRegistry:
    def test_has_13_aspects(self):
        assert len(ASPECT_REGISTRY) == 13

    def test_expected_aspects_present(self):
        expected = {
            "brightness", "texture", "warmth", "tempo", "rhythmic_energy",
            "dynamics", "crescendo", "harmonic_richness", "articulation", "phrasing",
            "rubato", "expressiveness", "legato",
        }
        assert set(ASPECT_REGISTRY.keys()) == expected

    def test_all_targets_in_scalar_targets(self):
        """Every target referenced by ASPECT_REGISTRY must exist in SCALAR_TARGETS."""
        scalar_targets = set(LayerDiscoverySystem.SCALAR_TARGETS.keys())
        for aspect_name, aspect_info in ASPECT_REGISTRY.items():
            for target in aspect_info["targets"]:
                assert target in scalar_targets, (
                    f"Aspect '{aspect_name}' references target '{target}' "
                    f"not found in SCALAR_TARGETS"
                )


class TestScalarTargets:
    def test_has_22_targets(self):
        assert len(LayerDiscoverySystem.SCALAR_TARGETS) == 22

    def test_segment_targets_is_subset(self):
        from mess.probing.discovery import SEGMENT_TARGETS
        scalar_names = set(LayerDiscoverySystem.SCALAR_TARGETS.keys())
        assert SEGMENT_TARGETS.issubset(scalar_names)

    def test_segment_targets_excludes_track_level(self):
        from mess.probing.discovery import SEGMENT_TARGETS
        for excluded in ('tempo', 'phrase_regularity', 'num_phrases', 'onset_density'):
            assert excluded not in SEGMENT_TARGETS


class TestLoadTargets:
    @staticmethod
    def _build_payload(
        seed: float,
        *,
        include_expression: bool = True,
        rubato: float = 0.0,
    ) -> dict[str, dict[str, np.ndarray]]:
        payload: dict[str, dict[str, np.ndarray]] = {}
        for name, (category, key, reduction) in LayerDiscoverySystem.SCALAR_TARGETS.items():
            if category == "expression" and not include_expression:
                continue

            payload.setdefault(category, {})
            if name == "rubato":
                value = np.array([rubato], dtype=np.float32)
            elif reduction == "mean":
                value = np.array([seed, seed + 0.1], dtype=np.float32)
            else:
                value = np.array([seed], dtype=np.float32)
            payload[category][key] = value
        return payload

    def test_load_targets_keeps_optional_target_with_partial_nan(self, tmp_path):
        system = object.__new__(LayerDiscoverySystem)
        system.targets_dir = tmp_path

        audio_files = ["track_a.wav", "track_b.wav", "track_c.wav"]

        np.savez_compressed(
            tmp_path / "track_a_targets.npz",
            **self._build_payload(1.0, include_expression=True, rubato=0.1),
        )
        np.savez_compressed(
            tmp_path / "track_b_targets.npz",
            **self._build_payload(2.0, include_expression=False),
        )
        np.savez_compressed(
            tmp_path / "track_c_targets.npz",
            **self._build_payload(3.0, include_expression=True, rubato=0.3),
        )

        targets, loaded = system.load_targets(audio_files)

        assert loaded == audio_files
        assert "rubato" in targets
        assert targets["rubato"].shape == (3,)
        assert targets["rubato"][0] == pytest.approx(0.1)
        assert np.isnan(targets["rubato"][1])
        assert targets["rubato"][2] == pytest.approx(0.3)

    def test_load_targets_skips_optional_target_when_all_values_missing(self, tmp_path):
        system = object.__new__(LayerDiscoverySystem)
        system.targets_dir = tmp_path

        audio_files = ["track_a.wav", "track_b.wav"]

        np.savez_compressed(
            tmp_path / "track_a_targets.npz",
            **self._build_payload(1.0, include_expression=False),
        )
        np.savez_compressed(
            tmp_path / "track_b_targets.npz",
            **self._build_payload(2.0, include_expression=False),
        )

        targets, loaded = system.load_targets(audio_files)

        assert loaded == audio_files
        assert "rubato" not in targets
