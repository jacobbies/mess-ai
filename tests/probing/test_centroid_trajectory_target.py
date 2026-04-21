"""Unit tests for T4 — spectral centroid trajectory target generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mess.probing._schema import DEFAULT_CURVE_SPEC, TargetType
from mess.probing.targets import _centroid_trajectory  # noqa: F401 — registration
from mess.probing.targets._registry import get_generator

pytestmark = pytest.mark.unit

_SR = 22050


def _write_wav(path: Path, audio: np.ndarray, sr: int = _SR) -> Path:
    sf.write(str(path), audio.astype(np.float32), sr)
    return path


def _sine(freq_hz: float, duration_s: float, sr: int = _SR) -> np.ndarray:
    t = np.arange(int(round(duration_s * sr))) / sr
    return 0.5 * np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)


def _log_sweep(
    f_start_hz: float,
    f_end_hz: float,
    duration_s: float,
    sr: int = _SR,
) -> np.ndarray:
    """Log-frequency (exponential) sweep from ``f_start_hz`` to ``f_end_hz``.

    The instantaneous frequency traces ``f(t) = f_start * (f_end/f_start)^(t/T)``
    so the log-centroid of the signal rises linearly in time.
    """
    n = int(round(duration_s * sr))
    t = np.arange(n) / sr
    k = np.log(f_end_hz / f_start_hz) / duration_s
    # phase = integral of 2π f(t) dt
    phase = 2.0 * np.pi * f_start_hz * (np.exp(k * t) - 1.0) / k
    return (0.5 * np.sin(phase)).astype(np.float32)


class TestRegistration:
    """The module registers its descriptor under the expected name."""

    def test_descriptor_registered(self):
        descriptor, fn = get_generator("centroid_trajectory")
        assert descriptor.name == "centroid_trajectory"
        assert descriptor.type is TargetType.CURVE
        assert descriptor.category == "curves"
        assert descriptor.curve_spec == DEFAULT_CURVE_SPEC
        assert callable(fn)


class TestShape:
    """generate() returns a float curve of the schema-declared shape."""

    def test_sine_wave_shape(self, tmp_path):
        audio = _sine(440.0, duration_s=DEFAULT_CURVE_SPEC.duration_s)
        wav = _write_wav(tmp_path / "sine.wav", audio)

        curve = _centroid_trajectory.generate(wav)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert curve.dtype == np.float32
        assert np.all(np.isfinite(curve))


class TestBrightnessRamp:
    """A rising log-frequency sweep should produce a rising centroid curve."""

    def test_ramp_200hz_to_4000hz_is_increasing(self, tmp_path):
        audio = _log_sweep(200.0, 4000.0, duration_s=DEFAULT_CURVE_SPEC.duration_s)
        wav = _write_wav(tmp_path / "sweep.wav", audio)

        curve = _centroid_trajectory.generate(wav)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert np.all(np.isfinite(curve))
        # Mean first-difference should be positive: curve rises overall.
        assert float(np.diff(curve).mean()) > 0.0, (
            f"expected rising curve; mean diff={np.diff(curve).mean():.4f}"
        )
        # Last 10 frames should be brighter than first 10 frames.
        head = float(curve[:10].mean())
        tail = float(curve[-10:].mean())
        assert tail > head, f"tail ({tail:.3f}) should exceed head ({head:.3f})"


class TestZScore:
    """A non-constant curve is standardized to ~zero-mean, unit-std."""

    def test_sweep_is_zscored(self, tmp_path):
        audio = _log_sweep(200.0, 4000.0, duration_s=DEFAULT_CURVE_SPEC.duration_s)
        wav = _write_wav(tmp_path / "sweep.wav", audio)

        curve = _centroid_trajectory.generate(wav)

        assert abs(float(curve.mean())) < 0.1, (
            f"expected mean ≈ 0 after z-scoring; got {curve.mean():.4f}"
        )
        assert abs(float(curve.std()) - 1.0) < 0.1, (
            f"expected std ≈ 1 after z-scoring; got {curve.std():.4f}"
        )


class TestSilence:
    """Silent audio produces a constant (zero) curve."""

    def test_silence_is_zero(self, tmp_path):
        audio = np.zeros(int(DEFAULT_CURVE_SPEC.duration_s * _SR), dtype=np.float32)
        wav = _write_wav(tmp_path / "silence.wav", audio)

        curve = _centroid_trajectory.generate(wav)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert float(curve.std()) == pytest.approx(0.0)
        assert np.allclose(curve, 0.0)


class TestShortAudio:
    """Audio shorter than duration_s is padded to the full frame grid."""

    def test_five_second_clip_pads_without_crashing(self, tmp_path):
        audio = _sine(440.0, duration_s=5.0)
        wav = _write_wav(tmp_path / "short.wav", audio)

        curve = _centroid_trajectory.generate(wav)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert np.all(np.isfinite(curve))


class TestStereoCollapse:
    """Stereo input is collapsed to mono before analysis."""

    def test_stereo_input_is_handled(self, tmp_path):
        mono = _sine(440.0, duration_s=DEFAULT_CURVE_SPEC.duration_s)
        stereo = np.stack([mono, mono * 0.5], axis=-1)  # (samples, 2)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, _SR)

        curve = _centroid_trajectory.generate(path)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert np.all(np.isfinite(curve))
