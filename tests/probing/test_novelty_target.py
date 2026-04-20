"""Unit tests for T5 — Foote novelty curve target generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mess.probing._schema import DEFAULT_CURVE_SPEC, TargetType
from mess.probing.targets import _novelty  # noqa: F401 — triggers registration
from mess.probing.targets._registry import get_generator

pytestmark = pytest.mark.unit

_SR = 22050


def _write_wav(path: Path, audio: np.ndarray, sr: int = _SR) -> Path:
    sf.write(str(path), audio.astype(np.float32), sr)
    return path


def _sine(freq_hz: float, duration_s: float, sr: int = _SR) -> np.ndarray:
    t = np.arange(int(round(duration_s * sr))) / sr
    return 0.5 * np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)


def _noise(duration_s: float, sr: int = _SR, rng_seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return (0.3 * rng.standard_normal(int(round(duration_s * sr)))).astype(np.float32)


class TestRegistration:
    """The module registers its descriptor under the expected name."""

    def test_descriptor_registered(self):
        descriptor, fn = get_generator("novelty")
        assert descriptor.name == "novelty"
        assert descriptor.type is TargetType.CURVE
        assert descriptor.category == "curves"
        assert descriptor.curve_spec == DEFAULT_CURVE_SPEC
        assert callable(fn)


class TestShape:
    """generate() returns a float curve of the schema-declared shape."""

    def test_sine_wave_shape(self, tmp_path):
        audio = _sine(440.0, duration_s=DEFAULT_CURVE_SPEC.duration_s)
        wav = _write_wav(tmp_path / "sine.wav", audio)

        curve = _novelty.generate(wav)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert curve.dtype == np.float32
        assert np.all(np.isfinite(curve))


class TestBoundary:
    """Two distinct timbres concatenated produce a peak at the join."""

    def test_sine_to_noise_peak_near_midpoint(self, tmp_path):
        half = DEFAULT_CURVE_SPEC.duration_s / 2.0
        sine = _sine(440.0, duration_s=half)
        noise = _noise(duration_s=half)
        audio = np.concatenate([sine, noise])
        wav = _write_wav(tmp_path / "sine_then_noise.wav", audio)

        curve = _novelty.generate(wav)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        # Midpoint of a 60-frame curve is ~frame 30. The boundary is the
        # clearest event in the audio, so argmax should land in a tight
        # window around it.
        peak = int(np.argmax(curve))
        assert 25 <= peak <= 35, (
            f"expected novelty peak near frame 30 (the join between sine "
            f"and noise); got argmax={peak}, curve={curve}"
        )


class TestSilence:
    """Silent audio produces a flat-ish (zero) curve and does not crash."""

    def test_silence_is_constant(self, tmp_path):
        audio = np.zeros(
            int(DEFAULT_CURVE_SPEC.duration_s * _SR), dtype=np.float32
        )
        wav = _write_wav(tmp_path / "silence.wav", audio)

        curve = _novelty.generate(wav)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert np.all(np.isfinite(curve))
        assert curve.std() == pytest.approx(0.0)


class TestShortAudio:
    """Audio shorter than duration_s is padded to the full frame grid."""

    def test_two_second_clip_pads_without_crashing(self, tmp_path):
        audio = _sine(440.0, duration_s=2.0)
        wav = _write_wav(tmp_path / "short.wav", audio)

        curve = _novelty.generate(wav)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert np.all(np.isfinite(curve))


class TestStereoCollapse:
    """Stereo input is collapsed to mono before analysis."""

    def test_stereo_input_is_handled(self, tmp_path):
        mono = _sine(440.0, duration_s=DEFAULT_CURVE_SPEC.duration_s)
        stereo = np.stack([mono, mono * 0.5], axis=-1)  # (samples, 2)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, _SR)

        curve = _novelty.generate(path)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert np.all(np.isfinite(curve))
