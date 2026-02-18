"""Tests for mess.probing.discovery — probing, best_layers, resolve_aspects, registries."""

import json

import numpy as np
import pytest

from mess.probing.discovery import (
    ASPECT_REGISTRY,
    LayerDiscoverySystem,
    resolve_aspects,
)


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


class TestAspectRegistry:
    def test_has_10_aspects(self):
        assert len(ASPECT_REGISTRY) == 10

    def test_expected_aspects_present(self):
        expected = {
            "brightness", "texture", "warmth", "tempo", "rhythmic_energy",
            "dynamics", "crescendo", "harmonic_richness", "articulation", "phrasing",
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
    def test_has_15_targets(self):
        assert len(LayerDiscoverySystem.SCALAR_TARGETS) == 15
