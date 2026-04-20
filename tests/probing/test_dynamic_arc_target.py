"""Unit tests for the T2 dynamic-arc envelope target."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mess.probing._schema import DEFAULT_CURVE_SPEC, TargetType
from mess.probing.targets._dynamic_arc import DESCRIPTOR, generate
from mess.probing.targets._registry import get_generator

pytestmark = pytest.mark.unit


_SR = 22050


def _write_wav(path: Path, audio: np.ndarray, sr: int = _SR) -> Path:
    """Write ``audio`` (float32, mono) to a WAV file at ``path``."""
    sf.write(str(path), audio.astype(np.float32), sr, subtype="PCM_16")
    return path


def _linear_ramp(duration_s: float, sr: int = _SR) -> np.ndarray:
    """Amplitude-ramped tone from silence to full scale (strict crescendo)."""
    n = int(round(duration_s * sr))
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float32)
    envelope = np.linspace(0.0, 1.0, n, dtype=np.float32)
    tone = np.sin(2.0 * np.pi * 440.0 * t, dtype=np.float32)
    return (envelope * tone).astype(np.float32)


class TestRegistration:
    """The target registers itself under the expected name."""

    def test_descriptor_metadata(self):
        assert DESCRIPTOR.name == "dynamic_arc"
        assert DESCRIPTOR.type is TargetType.CURVE
        assert DESCRIPTOR.category == "curves"
        assert DESCRIPTOR.curve_spec == DEFAULT_CURVE_SPEC

    def test_get_generator_returns_pair(self):
        descriptor, fn = get_generator("dynamic_arc")
        assert descriptor is DESCRIPTOR
        assert callable(fn)


class TestShape:
    """Any valid input returns the contract shape / dtype."""

    def test_shape_matches_default_spec(self, tmp_path):
        wav = _write_wav(tmp_path / "ramp.wav", _linear_ramp(30.0))
        curve = generate(wav)
        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert curve.dtype == np.float32


class TestCrescendo:
    """Monotonic amplitude ramp produces a rising arc."""

    @pytest.fixture
    def curve(self, tmp_path):
        wav = _write_wav(tmp_path / "crescendo.wav", _linear_ramp(30.0))
        return generate(wav)

    def test_overall_trend_is_increasing(self, curve):
        # Average first-order difference must be positive — smoothing can
        # inject mild jitter, but the bulk motion is upward.
        assert float(np.diff(curve).mean()) > 0.0

    def test_end_louder_than_start(self, curve):
        head = float(curve[:10].mean())
        tail = float(curve[-10:].mean())
        # The ramp covers the full range; tail should sit well above head.
        assert tail - head > 0.5

    def test_monotonic_within_tolerance(self, curve):
        # Allow small dips from Gaussian smoothing, but at least 90% of
        # frame-to-frame deltas should be non-negative.
        deltas = np.diff(curve)
        non_decreasing = float((deltas >= -1e-3).mean())
        assert non_decreasing >= 0.9


class TestNormalization:
    """Curve is min-max normalized to [0, 1]."""

    def test_range_is_unit(self, tmp_path):
        wav = _write_wav(tmp_path / "crescendo.wav", _linear_ramp(30.0))
        curve = generate(wav)
        # Tolerance acknowledges Gaussian smoothing softening the extrema.
        assert curve.min() == pytest.approx(0.0, abs=0.1)
        assert curve.max() == pytest.approx(1.0, abs=0.1)
        assert curve.min() >= 0.0
        assert curve.max() <= 1.0 + 1e-6


class TestEdgeCases:
    """Silent + very short inputs are handled without crashing."""

    def test_silence_returns_flat_zero_curve(self, tmp_path):
        silent = np.zeros(int(30.0 * _SR), dtype=np.float32)
        wav = _write_wav(tmp_path / "silence.wav", silent)
        curve = generate(wav)
        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        np.testing.assert_allclose(curve, 0.0, atol=1e-5)

    def test_short_audio_pads_and_succeeds(self, tmp_path):
        # 5 s of audio — the target will zero-pad to the full 30 s.
        wav = _write_wav(tmp_path / "short.wav", _linear_ramp(5.0))
        curve = generate(wav)
        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        # Final frames correspond to padded silence — should sit at the
        # low end of the normalized range.
        assert float(curve[-5:].mean()) < float(curve[:5].mean()) + 0.5

    def test_stereo_input_is_collapsed_to_mono(self, tmp_path):
        # Two-channel ramp — loader is expected to downmix.
        mono = _linear_ramp(30.0)
        stereo = np.stack([mono, mono], axis=1)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, _SR, subtype="PCM_16")
        curve = generate(path)
        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert float(np.diff(curve).mean()) > 0.0
