"""I1 integration tests: target-module wiring, ASPECT_REGISTRY updates,
dense-weight resolution, and dense-weight fusion in search_by_aspects."""

from __future__ import annotations

import numpy as np
import pytest

import mess.probing.targets  # noqa: F401 — registers T1-T7 generators on import
from mess.probing.discovery import ASPECT_REGISTRY, resolve_aspects
from mess.probing.targets._registry import all_names, get_generator
from mess.search.search import search_by_aspects

pytestmark = pytest.mark.unit


class TestTargetRegistryWiring:
    """Importing ``mess.probing.targets`` must populate the registry with T1-T7."""

    def test_all_seven_tunit_targets_registered(self):
        names = set(all_names())
        expected = {
            # T1-T5: curves
            "tis_tension",
            "dynamic_arc",
            "local_tempo",
            "centroid_trajectory",
            "novelty",
            # T6: MIDI curve
            "midi_articulation_hist",
            # T7: three MIDI scalars
            "midi_velocity_std",
            "midi_ioi_std",
            "midi_pedal_ratio",
        }
        missing = expected - names
        assert not missing, f"missing registered targets: {missing}"

    def test_every_registered_target_has_descriptor_and_callable(self):
        for name in all_names():
            descriptor, fn = get_generator(name)
            assert descriptor.name == name, f"descriptor name mismatch for {name}"
            assert callable(fn), f"generator for {name} is not callable"


class TestAspectRegistry:
    """Every registry aspect must reference at least one known probing target."""

    def test_new_aspects_present(self):
        expected_new = {
            "tension",
            "dynamic_arc",
            "rhythmic_flow",
            "brightness_trajectory",
            "structure",
            "micro_articulation",
            "performance_expression",
        }
        missing = expected_new - ASPECT_REGISTRY.keys()
        assert not missing, f"missing new aspects: {missing}"

    def test_dead_aspects_removed(self):
        # The false-promise MIDI aspects that were never computed, plus
        # ``crescendo`` (R² < 0), are gone — replaced by real targets.
        dead = {"rubato", "expressiveness", "legato", "crescendo"}
        assert dead.isdisjoint(ASPECT_REGISTRY.keys()), (
            f"expected dead aspects to be removed: "
            f"{dead & ASPECT_REGISTRY.keys()}"
        )

    def test_new_aspects_point_to_registered_targets(self):
        registered = set(all_names())
        new_aspects = {
            "tension",
            "dynamic_arc",
            "rhythmic_flow",
            "brightness_trajectory",
            "structure",
            "micro_articulation",
            "performance_expression",
        }
        for aspect in new_aspects:
            targets = ASPECT_REGISTRY[aspect]["targets"]
            assert any(t in registered for t in targets), (
                f"aspect {aspect!r} has no registered target among {targets}"
            )


class TestResolveAspectsDenseWeights:
    """resolve_aspects must return dense layer_weights when weighted-sum wins."""

    @pytest.fixture
    def results_file(self, tmp_path):
        import json
        results = {
            # Per-layer section (only layer 0 set; mid R² for the single probe).
            str(layer): {"tis_tension": {"r2_score": 0.30 + 0.01 * layer}}
            for layer in range(13)
        }
        # Weighted-sum section: dominant R² above gain threshold.
        results["weighted_sum"] = {
            "tis_tension": {
                "r2_score": 0.65,
                "layer_weights": [0.0] * 4 + [0.5] + [0.2] + [0.3] + [0.0] * 6,
                "r2_gain_over_best_single": 0.23,
            }
        }
        path = tmp_path / "results.json"
        path.write_text(json.dumps(results))
        return path

    def test_auto_prefers_weighted_sum_when_gain_above_threshold(self, results_file):
        resolved = resolve_aspects(
            min_r2=0.5, results_path=results_file, probe_mode="auto"
        )
        assert "tension" in resolved
        mapping = resolved["tension"]
        assert "layer_weights" in mapping
        assert "layer" not in mapping
        assert len(mapping["layer_weights"]) == 13
        assert abs(sum(mapping["layer_weights"]) - 1.0) < 1e-6

    def test_best_layer_forces_single_layer(self, results_file):
        resolved = resolve_aspects(
            min_r2=0.1, results_path=results_file, probe_mode="best_layer"
        )
        assert "tension" in resolved
        mapping = resolved["tension"]
        assert "layer" in mapping
        assert "layer_weights" not in mapping

    def test_auto_falls_back_when_gain_small(self, tmp_path):
        import json
        results = {
            str(layer): {"tis_tension": {"r2_score": 0.6 if layer == 5 else 0.1}}
            for layer in range(13)
        }
        results["weighted_sum"] = {
            "tis_tension": {
                "r2_score": 0.61,  # barely beats single (gain=0.01 < 0.02)
                "layer_weights": [0.0] * 5 + [1.0] + [0.0] * 7,
                "r2_gain_over_best_single": 0.01,
            }
        }
        path = tmp_path / "results.json"
        path.write_text(json.dumps(results))
        resolved = resolve_aspects(min_r2=0.5, results_path=path, probe_mode="auto")
        assert "layer" in resolved["tension"]
        assert resolved["tension"]["layer"] == 5


class TestSearchByAspectsDenseWeights:
    """search_by_aspects must fuse dense layer_weights correctly."""

    def test_dense_weights_fusion(self, monkeypatch):
        from mess.search import search

        # Four tracks, each with 13 layer vectors of dim 768 (the fusion
        # helper enforces this shape). Query is non-zero on layers 3 and 7;
        # ``match`` mirrors it exactly; the ``only_*`` tracks each carry
        # signal on just one of those layers.
        d = 768
        query = np.zeros((13, d), dtype="float32")
        query[3, 0] = 1.0
        query[7, 1] = 1.0
        match = query.copy()
        only_3 = np.zeros_like(query)
        only_3[3, 0] = 1.0
        only_7 = np.zeros_like(query)
        only_7[7, 1] = 1.0

        layer_features = np.stack([query, match, only_3, only_7], axis=0)
        track_names = ["query", "match", "only_3", "only_7"]

        monkeypatch.setattr(
            search, "_load_layer_features",
            lambda features_dir: (layer_features, track_names),
        )

        # Dense weights split evenly across layers 3 and 7. ``match`` should
        # score highest because it matches the query on both relevant layers.
        weights = [0.0] * 13
        weights[3] = 0.5
        weights[7] = 0.5
        monkeypatch.setattr(
            "mess.probing.resolve_aspects",
            lambda min_r2=0.5: {
                "tension": {
                    "layer_weights": weights,
                    "r2_score": 0.9,
                    "target": "tis_tension",
                    "description": "tonal tension",
                    "confidence": "high",
                },
            },
        )

        results = search_by_aspects(
            query_track="query",
            aspect_weights={"tension": 1.0},
            features_dir="/unused",
            k=3,
        )
        names = [name for name, _ in results]
        assert names[0] == "match", (
            f"expected 'match' (identical on layers 3+7) first; got {names}"
        )
