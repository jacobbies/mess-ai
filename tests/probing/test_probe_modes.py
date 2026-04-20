"""Unit tests for curve + weighted-sum probes and the new target schema.

All tests operate on synthetic data — no real MERT or audio needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mess.probing._schema import (
    DEFAULT_CURVE_SPEC,
    CurveSpec,
    TargetDescriptor,
    TargetType,
    load_target_field,
)
from mess.probing.discovery import LayerDiscoverySystem

pytestmark = pytest.mark.unit


def _system(n_folds: int = 3, alpha: float = 1.0) -> LayerDiscoverySystem:
    """Build a LayerDiscoverySystem without touching the dataset factory."""
    sys = object.__new__(LayerDiscoverySystem)
    sys.alpha = alpha
    sys.n_folds = n_folds
    return sys


class TestScalarProbe:
    """Scalar Ridge probe preserves legacy behavior exactly."""

    def test_returns_legacy_keys_on_linear_data(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 8))
        y = X @ rng.standard_normal(8)
        metrics = _system()._probe_single(X, y)
        assert set(metrics) == {"r2_score", "correlation", "rmse"}
        assert metrics["r2_score"] > 0.95

    def test_returns_noise_below_threshold(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 8))
        y = rng.standard_normal(40)
        metrics = _system()._probe_single(X, y)
        assert metrics["r2_score"] < 0.5


class TestCurveProbe:
    """Multi-output Ridge curve probe reports mean + PC1 metrics."""

    def test_curve_probe_fits_linear_target(self):
        rng = np.random.default_rng(1)
        n, d, t = 60, 16, 20
        X = rng.standard_normal((n, d))
        # True curve: a (d, t) linear map + per-frame sinusoidal noise.
        W = rng.standard_normal((d, t))
        noise = 0.05 * np.sin(
            np.linspace(0, 4 * np.pi, n * t).reshape(n, t) + rng.standard_normal((n, 1))
        )
        y = X @ W + noise

        metrics = _system()._probe_single(X, y)

        assert set(metrics) >= {"r2_mean", "r2_pc1", "rmse_mean"}
        assert metrics["r2_mean"] > 0.8
        assert metrics["rmse_mean"] >= 0.0

    def test_curve_probe_returns_bad_metrics_when_undersampled(self):
        rng = np.random.default_rng(2)
        y = rng.standard_normal((2, 5))
        X = rng.standard_normal((2, 3))
        metrics = _system()._probe_single(X, y)
        assert metrics["r2_mean"] == -999.0


class TestMidiMask:
    """NaN rows (tracks without MIDI) are dropped; coverage/n_valid reported."""

    def test_scalar_with_nan_rows(self):
        rng = np.random.default_rng(3)
        n, d = 40, 8
        X = rng.standard_normal((n, d))
        w = rng.standard_normal(d)
        y = X @ w
        y = y.astype(float)
        y[: n // 2] = np.nan  # drop first half

        metrics = _system()._probe_single(X, y)

        assert metrics["coverage"] == pytest.approx(0.5)
        assert metrics["n_valid"] == pytest.approx(n / 2)
        assert metrics["r2_score"] > 0.8

    def test_curve_with_nan_rows(self):
        rng = np.random.default_rng(3)
        n, d, t = 40, 8, 10
        X = rng.standard_normal((n, d))
        W = rng.standard_normal((d, t))
        y = X @ W
        y[: n // 2, :] = np.nan

        metrics = _system()._probe_single(X, y)

        assert metrics["coverage"] == pytest.approx(0.5)
        assert metrics["n_valid"] == pytest.approx(n / 2)
        assert metrics["r2_mean"] > 0.5


class TestWeightedSumProbe:
    """Weighted-sum probe finds the layer carrying signal."""

    def test_weighted_sum_prefers_signal_bearing_layer(self):
        rng = np.random.default_rng(4)
        n, n_layers, d = 60, 13, 16
        # Layer 5 carries the signal; other layers are noise.
        signal_X = rng.standard_normal((n, d))
        w = rng.standard_normal(d)
        y = signal_X @ w

        X_all_layers = rng.standard_normal((n, n_layers, d))
        X_all_layers[:, 5, :] = signal_X

        metrics = _system()._probe_weighted_sum(X_all_layers, y)

        assert len(metrics["layer_weights"]) == n_layers
        assert sum(metrics["layer_weights"]) == pytest.approx(1.0)
        # Layer 5 should dominate.
        assert metrics["layer_weights"][5] > 0.5
        # Gain over the best single layer should be non-negative (the best
        # single-layer probe is already optimal, so gain can be ~0 or tiny).
        assert metrics["r2_gain_over_best_single"] >= -1e-6
        assert metrics["r2_score"] > 0.8


class TestTargetDescriptorLoading:
    """load_target_field dispatches by descriptor type against a real .npz."""

    @pytest.fixture
    def npz_path(self, tmp_path):
        path = tmp_path / "track_targets.npz"
        curves = {"dynamic_arc": np.linspace(0.0, 1.0, DEFAULT_CURVE_SPEC.n_frames)}
        midi_fields = {
            "velocity_std": np.array([12.3]),
            "pedal_curve": np.ones(DEFAULT_CURVE_SPEC.n_frames),
        }
        np.savez_compressed(
            path,
            timbre={"spectral_centroid": np.array([1234.5])},
            curves=curves,
            midi=midi_fields,
            midi_available=np.array([True]),
        )
        return path

    def test_scalar_loads(self, npz_path):
        desc = TargetDescriptor(
            name="spectral_centroid",
            type=TargetType.SCALAR,
            category="timbre",
        )
        value = load_target_field(npz_path, desc)
        assert value is not None
        assert value[0] == pytest.approx(1234.5)

    def test_curve_loads_with_shape_check(self, npz_path):
        desc = TargetDescriptor(
            name="dynamic_arc",
            type=TargetType.CURVE,
            category="curves",
            curve_spec=DEFAULT_CURVE_SPEC,
        )
        value = load_target_field(npz_path, desc)
        assert value is not None
        assert value.shape == (DEFAULT_CURVE_SPEC.n_frames,)

    def test_midi_scalar_respects_available_flag(self, tmp_path, npz_path):
        # Available = True -> loads.
        desc = TargetDescriptor(
            name="velocity_std",
            type=TargetType.MIDI_SCALAR,
            category="midi",
        )
        value = load_target_field(npz_path, desc)
        assert value is not None and value[0] == pytest.approx(12.3)

        # Available = False -> returns None.
        unavailable = tmp_path / "no_midi_targets.npz"
        np.savez_compressed(
            unavailable,
            midi={"velocity_std": np.array([0.0])},
            midi_available=np.array([False]),
        )
        assert load_target_field(unavailable, desc) is None

    def test_curve_spec_n_frames(self):
        spec = CurveSpec(frame_rate_hz=2.0, duration_s=30.0)
        assert spec.n_frames == 60

    def test_missing_file_returns_none(self, tmp_path):
        desc = TargetDescriptor(
            name="velocity_std",
            type=TargetType.MIDI_SCALAR,
            category="midi",
        )
        assert load_target_field(tmp_path / "nope.npz", desc) is None

    def test_curve_shape_mismatch_returns_none(self, tmp_path):
        # Writes a curve with the wrong number of frames.
        path = tmp_path / "bad_curve.npz"
        np.savez_compressed(path, curves={"dynamic_arc": np.zeros(3)})
        desc = TargetDescriptor(
            name="dynamic_arc",
            type=TargetType.CURVE,
            category="curves",
            curve_spec=DEFAULT_CURVE_SPEC,
        )
        assert load_target_field(path, desc) is None


class TestBackwardsCompat:
    """The legacy ``{0..12: ...}`` results shape must still work end-to-end."""

    def test_best_layers_ignores_weighted_sum_section(self):
        # Sanity: best_layers tolerates a ``"weighted_sum"`` top-level key.
        results = {
            0: {"spectral_centroid": {"r2_score": 0.9}},
            1: {"spectral_centroid": {"r2_score": 0.7}},
            "weighted_sum": {"spectral_centroid": {"r2_score": 0.95}},
        }
        best = LayerDiscoverySystem.best_layers(results)
        assert best["spectral_centroid"]["layer"] == 0
        assert best["spectral_centroid"]["r2_score"] == pytest.approx(0.9)
