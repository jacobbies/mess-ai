"""Unit tests for T1 — tonal tension curve (TIS) target generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mess.probing._schema import DEFAULT_CURVE_SPEC, TargetType
from mess.probing.targets import _tis_tension  # noqa: F401 — triggers registration
from mess.probing.targets._registry import get_generator

pytestmark = pytest.mark.unit

_SR = 22050


def _write_wav(path: Path, audio: np.ndarray, sr: int = _SR) -> Path:
    sf.write(str(path), audio.astype(np.float32), sr)
    return path


def _sine(freq_hz: float, duration_s: float, sr: int = _SR) -> np.ndarray:
    t = np.arange(int(round(duration_s * sr))) / sr
    return 0.5 * np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)


def _mix_pitches(freqs: list[float], duration_s: float, sr: int = _SR) -> np.ndarray:
    """Sum of equal-amplitude sinusoids at the listed Hz values."""
    if not freqs:
        return np.zeros(int(round(duration_s * sr)), dtype=np.float32)
    waves = [_sine(f, duration_s, sr) for f in freqs]
    mix = np.sum(np.stack(waves), axis=0) / len(waves)
    return mix.astype(np.float32)


def _midi_to_hz(midi_number: int) -> float:
    return float(440.0 * 2.0 ** ((midi_number - 69) / 12.0))


class TestRegistration:
    """The module registers its descriptor under the expected name."""

    def test_descriptor_registered(self):
        descriptor, fn = get_generator("tis_tension")
        assert descriptor.name == "tis_tension"
        assert descriptor.type is TargetType.CURVE
        assert descriptor.category == "curves"
        assert descriptor.curve_spec == DEFAULT_CURVE_SPEC
        assert callable(fn)


class TestShape:
    """generate() returns a float curve of the schema-declared shape."""

    def test_sine_wave_shape(self, tmp_path):
        audio = _sine(440.0, duration_s=DEFAULT_CURVE_SPEC.duration_s)
        wav = _write_wav(tmp_path / "sine.wav", audio)

        curve = _tis_tension.generate(wav)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert curve.dtype == np.float32
        assert np.all(np.isfinite(curve))


class TestDynamicRange:
    """Chord progression with varying tension yields non-zero variance."""

    def test_chord_progression_has_variance(self, tmp_path):
        # Piano-register triads: C major -> C diminished -> C augmented.
        # Splice them end-to-end so the curve sees three distinct chroma
        # states; their TIS projections differ, so tension must vary.
        chord_s = DEFAULT_CURVE_SPEC.duration_s / 3.0
        major = _mix_pitches(
            [_midi_to_hz(60), _midi_to_hz(64), _midi_to_hz(67)], chord_s
        )
        diminished = _mix_pitches(
            [_midi_to_hz(60), _midi_to_hz(63), _midi_to_hz(66)], chord_s
        )
        augmented = _mix_pitches(
            [_midi_to_hz(60), _midi_to_hz(64), _midi_to_hz(68)], chord_s
        )
        audio = np.concatenate([major, diminished, augmented])
        wav = _write_wav(tmp_path / "progression.wav", audio)

        curve = _tis_tension.generate(wav)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert curve.std() > 0.05, (
            "curve should respond to harmonic changes between chords; "
            f"got std={curve.std():.4f}"
        )


class TestSilence:
    """Silent audio produces a constant (zero) curve."""

    def test_silence_is_constant(self, tmp_path):
        audio = np.zeros(int(DEFAULT_CURVE_SPEC.duration_s * _SR), dtype=np.float32)
        wav = _write_wav(tmp_path / "silence.wav", audio)

        curve = _tis_tension.generate(wav)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert curve.std() == pytest.approx(0.0)


class TestShortAudio:
    """Audio shorter than duration_s is padded to the full frame grid."""

    def test_five_second_clip_pads_without_crashing(self, tmp_path):
        audio = _sine(440.0, duration_s=5.0)
        wav = _write_wav(tmp_path / "short.wav", audio)

        curve = _tis_tension.generate(wav)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert np.all(np.isfinite(curve))


class TestStereoCollapse:
    """Stereo input is collapsed to mono before analysis."""

    def test_stereo_input_is_handled(self, tmp_path):
        mono = _sine(440.0, duration_s=DEFAULT_CURVE_SPEC.duration_s)
        stereo = np.stack([mono, mono * 0.5], axis=-1)  # (samples, 2)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, _SR)

        curve = _tis_tension.generate(path)

        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert np.all(np.isfinite(curve))
