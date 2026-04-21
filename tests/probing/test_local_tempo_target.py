"""Unit tests for the T3 local-tempo curve target.

All tests use synthetic click tracks so the suite stays hermetic (no real
audio fixtures) and fast.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from mess.probing._schema import DEFAULT_CURVE_SPEC, CurveSpec, TargetType
from mess.probing.targets._local_tempo import DESCRIPTOR, generate
from mess.probing.targets._registry import get_generator

pytestmark = pytest.mark.unit


_SR = 22_050


def _click_track(bpm: float, duration_s: float, path, sr: int = _SR) -> None:
    """Write a simple click-track WAV at ``bpm`` to ``path``."""
    n = int(round(sr * duration_s))
    audio = np.zeros(n, dtype=np.float32)
    interval_samples = int(round(sr * 60.0 / bpm))
    click_len = 32
    click = np.hanning(click_len).astype(np.float32)
    for start in range(0, n - click_len, interval_samples):
        audio[start : start + click_len] += click
    # Normalize, headroom.
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = 0.8 * audio / peak
    sf.write(str(path), audio, sr)


def _log_bpm(bpm: float) -> float:
    return float(np.log2(bpm / 60.0))


class TestRegistration:
    def test_descriptor_fields(self):
        assert DESCRIPTOR.name == "local_tempo"
        assert DESCRIPTOR.type is TargetType.CURVE
        assert DESCRIPTOR.category == "curves"
        assert DESCRIPTOR.curve_spec == DEFAULT_CURVE_SPEC

    def test_registered_with_registry(self):
        desc, fn = get_generator("local_tempo")
        assert desc is DESCRIPTOR
        assert fn is generate


class TestShape:
    def test_returns_default_curve_length(self, tmp_path):
        wav = tmp_path / "click120.wav"
        _click_track(120.0, duration_s=30.0, path=wav)
        curve = generate(wav)
        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert curve.dtype == np.float32
        assert np.all(np.isfinite(curve))

    def test_respects_custom_curve_spec(self, tmp_path):
        wav = tmp_path / "click120.wav"
        _click_track(120.0, duration_s=10.0, path=wav)
        spec = CurveSpec(frame_rate_hz=4.0, duration_s=10.0)
        curve = generate(wav, curve_spec=spec)
        assert curve.shape == (spec.n_frames,)


class TestTracksTempo:
    """Curve mean should sit near log2(bpm/60) for each click track.

    Note on 180 BPM: a perfectly periodic click has equal autocorrelation
    energy at every integer multiple of the true lag, so a log-normal prior
    centered at 120 BPM (the librosa default) resolves 180 BPM to its half,
    90 BPM. That is the expected, documented behavior — the prior is what
    keeps 60 BPM from collapsing to 30 or 20. Real classical performances
    have enough micro-timing noise to break this symmetry.
    """

    @pytest.mark.parametrize(
        "bpm,expected_log,tolerance",
        [
            (60.0, 0.0, 0.2),
            (120.0, 1.0, 0.2),
            # 180 BPM click aliases to 90 BPM (~0.585) under the tempo prior.
            (180.0, np.log2(90.0 / 60.0), 0.2),
        ],
    )
    def test_click_track_mean_matches_tempo(self, tmp_path, bpm, expected_log, tolerance):
        wav = tmp_path / f"click{int(bpm)}.wav"
        _click_track(bpm, duration_s=30.0, path=wav)
        curve = generate(wav)
        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert abs(curve.mean() - expected_log) < tolerance, (
            f"bpm={bpm}: mean={curve.mean():.3f} expected={expected_log:.3f}"
        )


class TestEdgeCases:
    def test_silent_audio_returns_default_curve(self, tmp_path):
        wav = tmp_path / "silence.wav"
        silent = np.zeros(int(_SR * 30.0), dtype=np.float32)
        sf.write(str(wav), silent, _SR)
        curve = generate(wav)
        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert np.allclose(curve, _log_bpm(120.0))

    def test_short_audio_pads_without_crash(self, tmp_path):
        wav = tmp_path / "short.wav"
        _click_track(120.0, duration_s=2.0, path=wav)
        curve = generate(wav)
        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert np.all(np.isfinite(curve))

    def test_stereo_input_is_downmixed(self, tmp_path):
        wav = tmp_path / "stereo.wav"
        mono = np.zeros(int(_SR * 5.0), dtype=np.float32)
        interval = int(round(_SR * 0.5))  # 120 BPM
        click_len = 32
        click = np.hanning(click_len).astype(np.float32)
        for start in range(0, mono.size - click_len, interval):
            mono[start : start + click_len] += click
        mono = 0.8 * mono / max(np.max(np.abs(mono)), 1e-9)
        stereo = np.stack([mono, mono], axis=-1)
        sf.write(str(wav), stereo, _SR)

        curve = generate(wav)
        assert curve.shape == (DEFAULT_CURVE_SPEC.n_frames,)
        assert np.all(np.isfinite(curve))
