"""Local tempo curve target (T3).

Replaces the scalar ``tempo`` target, which had negative mean R^2 across all
MERT layers because librosa's ``beat_track`` is brittle on classical rubato.
A local tempo *curve* captures expressive timing variation — the performer's
rhythmic shaping that scalar BPM throws away.

Pipeline
--------

1. Load audio, mix to mono, truncate/pad to ``curve_spec.duration_s``.
2. Onset envelope -> Fourier tempogram via librosa.
3. Apply a log-normal tempo prior (mean 120 BPM, std 1 octave) to the
   log-magnitude tempogram and argmax the weighted result per frame. The
   prior resolves octave aliasing — without it, periodic signals have equal
   autocorrelation energy at every integer multiple of the true lag and
   argmax collapses to the slowest in-range tempo. This matches
   ``librosa.feature.rhythm.tempo``'s default behavior.
4. Convert lag -> BPM and clip to [40, 240] BPM.
5. Median-filter (length 5) to suppress residual octave jumps on rubato.
6. Pool down to ``curve_spec.n_frames`` frames over ``duration_s``.
7. Log-scale: ``log2(bpm / 60)``. Reasons:
   * Tempo doubling = +1, halving = -1 (octave-symmetric, matches how
     musicians perceive tempo and how the tempogram aliases octaves).
   * Compresses the [40, 240] range into ~[-0.58, +2.0], keeping the Ridge
     regression target well-conditioned (large BPM numbers would otherwise
     dominate multi-output Ridge's per-frame scale).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._schema import DEFAULT_CURVE_SPEC, CurveSpec, TargetDescriptor, TargetType
from ._registry import register

DESCRIPTOR = TargetDescriptor(
    name="local_tempo",
    type=TargetType.CURVE,
    category="curves",
    curve_spec=DEFAULT_CURVE_SPEC,
)

# Implementation constants.
_SR = 22_050
_HOP_LENGTH = 512
_TEMPOGRAM_WIN_LENGTH = 384
_BPM_MIN = 40.0
_BPM_MAX = 240.0
_DEFAULT_BPM = 120.0  # log2(120/60) = 1.0 — the neutral fallback tempo.
_MEDIAN_KERNEL = 5
# Log-normal tempo prior (mean 120 BPM, 1-octave std) — matches the default
# in ``librosa.feature.rhythm.tempo`` and resolves octave aliasing.
_PRIOR_MEAN_BPM = 120.0
_PRIOR_STD_OCTAVES = 1.0


def _log_bpm(bpm: float) -> float:
    return float(np.log2(bpm / 60.0))


def _load_mono(audio_path: str | Path, duration_s: float) -> np.ndarray:
    """Load audio at the fixed sample rate, force mono, pad/truncate."""
    import librosa

    audio, _ = librosa.load(str(audio_path), sr=_SR, mono=True, duration=duration_s)
    target_len = int(round(_SR * duration_s))
    if audio.size < target_len:
        audio = np.pad(audio, (0, target_len - audio.size))
    else:
        audio = audio[:target_len]
    return audio.astype(np.float32)


def _median_filter(curve: np.ndarray, kernel: int) -> np.ndarray:
    """Centered 1-D median filter with edge reflection."""
    if kernel <= 1 or curve.size == 0:
        return curve
    pad = kernel // 2
    padded = np.pad(curve, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
    return np.asarray(np.median(windows, axis=-1))


def _pool_to_frames(curve: np.ndarray, n_frames: int) -> np.ndarray:
    """Mean-pool ``curve`` into exactly ``n_frames`` equal-time buckets.

    When the source is shorter than ``n_frames`` (very short audio clips),
    the unreachable tail is filled with the neutral default tempo.
    """
    if curve.size == 0 or n_frames <= 0:
        return np.full(max(n_frames, 0), _log_bpm(_DEFAULT_BPM), dtype=np.float32)
    if curve.size == n_frames:
        return curve.astype(np.float32)

    if curve.size < n_frames:
        out = np.full(n_frames, _log_bpm(_DEFAULT_BPM), dtype=np.float32)
        out[: curve.size] = curve.astype(np.float32)
        return out

    buckets = np.minimum(
        (np.arange(curve.size) * n_frames) // curve.size,
        n_frames - 1,
    )
    sums = np.bincount(buckets, weights=curve, minlength=n_frames)
    counts = np.maximum(np.bincount(buckets, minlength=n_frames), 1)
    return (sums / counts).astype(np.float32)


def generate(
    audio_path: str | Path,
    curve_spec: CurveSpec = DEFAULT_CURVE_SPEC,
) -> np.ndarray:
    """Produce a ``(n_frames,)`` local-tempo curve in log2(BPM/60).

    Returns a flat curve at ``log2(120/60) = 1.0`` when the audio is silent,
    no onsets are detected, or the tempogram is degenerate. Never raises on
    edge-case audio — callers batch-process many tracks and should get a
    well-shaped array for every input.
    """
    import librosa

    n_frames = curve_spec.n_frames
    default_curve = np.full(n_frames, _log_bpm(_DEFAULT_BPM), dtype=np.float32)

    audio = _load_mono(audio_path, curve_spec.duration_s)
    if not np.any(audio):
        return default_curve

    onset_env = librosa.onset.onset_strength(y=audio, sr=_SR, hop_length=_HOP_LENGTH)
    if onset_env.size == 0 or not np.any(onset_env):
        return default_curve

    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env,
        sr=_SR,
        hop_length=_HOP_LENGTH,
        win_length=_TEMPOGRAM_WIN_LENGTH,
    )
    if tempogram.size == 0 or tempogram.shape[1] == 0:
        return default_curve

    n_lags = tempogram.shape[0]
    lag_indices = np.arange(n_lags, dtype=float)
    # BPM of each lag; lag 0 is infinite tempo and gets zero prior mass.
    bpm_of_lag = np.zeros(n_lags, dtype=float)
    bpm_of_lag[1:] = 60.0 * _SR / (_HOP_LENGTH * lag_indices[1:])

    # Log-normal prior: 0 at lag 0, tapered Gaussian in log-BPM space.
    prior = np.zeros(n_lags, dtype=float)
    with np.errstate(divide="ignore"):
        log_bpm = np.log2(np.where(bpm_of_lag > 0, bpm_of_lag, 1.0))
    z = (log_bpm - np.log2(_PRIOR_MEAN_BPM)) / _PRIOR_STD_OCTAVES
    prior[1:] = np.exp(-0.5 * z[1:] ** 2)
    # Also zero-out lags whose BPM is outside the plausible [40, 240] band.
    prior[(bpm_of_lag < _BPM_MIN) | (bpm_of_lag > _BPM_MAX)] = 0.0

    # Per-frame weighted argmax.
    log_mag = np.log1p(np.maximum(tempogram, 0.0))
    weighted = log_mag * prior[:, None]
    if not np.any(weighted > 0):
        return default_curve
    # Prior mass is 0 at lag 0 and outside [_BPM_MIN, _BPM_MAX], so argmax
    # is guaranteed to land on a plausible-tempo lag.
    lags = np.argmax(weighted, axis=0).astype(int)
    bpm = np.clip(bpm_of_lag[lags], _BPM_MIN, _BPM_MAX)

    # Median-filter in BPM space to suppress octave flips before log-scaling.
    bpm = _median_filter(bpm, _MEDIAN_KERNEL)

    log_curve = np.log2(bpm / 60.0)
    pooled = _pool_to_frames(log_curve, n_frames)

    if not np.all(np.isfinite(pooled)):
        return default_curve
    return pooled


register("local_tempo", DESCRIPTOR, generate)
