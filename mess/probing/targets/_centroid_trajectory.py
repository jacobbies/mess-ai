"""T4 — Spectral centroid trajectory (log-scaled, per-track z-scored).

Per-frame spectral centroid (Hz) captures how "bright" a passage sounds. The
scalar summary of centroid is flat-spread across MERT layers (R² layer-spread
≈ 0.12), so picking a "best layer" for brightness is statistical noise. A
trajectory, in contrast, exposes *how* brightness evolves — rise/fall,
inflections, local variance — which is exactly what a multi-output Ridge probe
can learn to align with specific MERT layers.

Key design choice — per-track z-scoring
---------------------------------------
Absolute brightness is trivially encoded everywhere (register-dependent,
instrument-dependent, recording-dependent). We care about the *shape* of
brightness evolution, not the baseline. After log-scaling to compress the
wide Hz range and bring the curve into a perceptually linear domain, we
z-score per track:

    curve = log2(centroid + 1)         # +1 avoids log(0) on silent frames
    curve = (curve - mean) / (std + ε) # per-track standardization

This forces the probe to learn "which layer encodes *relative* brightness
variation" rather than "which layer encodes absolute brightness" — the
former is discriminative between layers; the latter is not.

Replaces the scalar ``spectral_centroid`` and ``spectral_rolloff`` targets
(both flat-spread across layers).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...extraction.audio import load_audio
from .._schema import (
    DEFAULT_CURVE_SPEC,
    CurveSpec,
    TargetDescriptor,
    TargetType,
)
from ._registry import register

# Audio loading sample rate. 22.05 kHz gives a Nyquist of ~11 kHz which
# amply covers brightness-relevant energy for classical repertoire.
_AUDIO_SR: int = 22050

# STFT parameters. hop=512 at 22.05 kHz => ~43 frames/sec, so even a 1 s
# pooled window averages ~43 STFT frames — plenty of statistics.
_N_FFT: int = 2048
_HOP_LENGTH: int = 512

# Numerical stability for the per-track z-score denominator.
_ZSCORE_EPS: float = 1e-6

# Absolute-amplitude threshold below which we treat the clip as pure
# silence (returns a zero curve without running FFTs).
_SILENCE_AMP: float = 1e-8


DESCRIPTOR = TargetDescriptor(
    name="centroid_trajectory",
    type=TargetType.CURVE,
    category="curves",
    curve_spec=DEFAULT_CURVE_SPEC,
)


def _spectral_centroid_hz(audio: np.ndarray, sr: int) -> np.ndarray:
    """Compute per-frame spectral centroid in Hz; returns shape ``(n_hops,)``."""
    import librosa  # lazy — heavy dep

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=_N_FFT,
        hop_length=_HOP_LENGTH,
    )
    return np.asarray(centroid[0], dtype=np.float64)


def _mean_pool_to_frames(values: np.ndarray, n_frames: int) -> np.ndarray:
    """Mean-pool ``(n_hops,)`` into ``(n_frames,)`` over equal-time windows.

    Right-pads with zeros when ``n_hops <= n_frames`` so the frame grid is
    always fully populated; otherwise each frame averages its contiguous
    bucket of STFT hops.
    """
    if values.size == 0 or n_frames <= 0:
        return np.zeros(max(n_frames, 0), dtype=np.float64)

    n_hops = values.shape[0]
    if n_hops <= n_frames:
        pooled = np.zeros(n_frames, dtype=np.float64)
        pooled[:n_hops] = values
        return pooled

    bucket = np.minimum(
        (np.arange(n_hops) * n_frames) // n_hops,
        n_frames - 1,
    )
    sums = np.bincount(bucket, weights=values, minlength=n_frames)
    counts = np.maximum(np.bincount(bucket, minlength=n_frames), 1)
    return np.asarray(sums / counts, dtype=np.float64)


def _zscore(curve: np.ndarray) -> np.ndarray:
    """Per-track z-score; a constant curve maps to all zeros."""
    mean = float(curve.mean())
    std = float(curve.std())
    if std < _ZSCORE_EPS:
        return np.zeros_like(curve, dtype=np.float32)
    return ((curve - mean) / (std + _ZSCORE_EPS)).astype(np.float32)


def generate(
    audio_path: str | Path,
    curve_spec: CurveSpec = DEFAULT_CURVE_SPEC,
) -> np.ndarray:
    """Produce a ``(n_frames,)`` spectral-centroid trajectory for ``audio_path``.

    Steps:

    1. Load audio (mono, 22.05 kHz) via ``mess.extraction.audio.load_audio``
       and truncate/pad to ``curve_spec.duration_s``.
    2. Compute per-frame spectral centroid in Hz via ``librosa``
       (``n_fft=2048``, ``hop_length=512`` => ~43 fps).
    3. Mean-pool the STFT-rate centroid onto ``curve_spec.n_frames`` bins.
    4. Log-scale (``log2(x + 1)``) to compress the Hz range.
    5. Per-track z-score so the probe learns relative (not absolute) brightness.

    Silent audio yields an all-zero curve. Returns ``float32``.
    """
    n_frames = curve_spec.n_frames
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    audio = np.asarray(load_audio(audio_path, target_sr=_AUDIO_SR), dtype=np.float32)

    max_samples = int(round(curve_spec.duration_s * _AUDIO_SR))
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]
    elif audio.shape[0] < max_samples:
        audio = np.pad(audio, (0, max_samples - audio.shape[0]))

    # Short-circuit on fully-silent audio; spectral_centroid on zeros is
    # ill-defined and the probe should see a flat (zero) curve.
    if not np.any(np.abs(audio) > _SILENCE_AMP):
        return np.zeros(n_frames, dtype=np.float32)

    centroid_hz = _spectral_centroid_hz(audio, _AUDIO_SR)
    pooled = _mean_pool_to_frames(centroid_hz, n_frames)
    log_scaled = np.log2(pooled + 1.0)
    return _zscore(log_scaled)


register("centroid_trajectory", DESCRIPTOR, generate)
