"""
Dynamic arc envelope target (T2).

Produces a per-track-normalized RMS-energy envelope of shape
``(curve_spec.n_frames,)`` that preserves the expressive *shape* of a
track's loudness trajectory. This replaces the scalar targets
``crescendo_strength``, ``diminuendo_strength``, and ``dynamic_variance``
— classical dynamics are curves, not scalars, and collapsing a crescendo
to a single number destroys the shape.

Pipeline:
    wav -> mono @ sr -> truncate/pad to duration_s -> short-time RMS
    -> dB with floor -> light Gaussian smoothing -> per-track min/max
    normalization to [0, 1].

Per-track normalization is intentional: we want the *shape* of the arc,
not absolute loudness (different tracks have different mastering).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._schema import DEFAULT_CURVE_SPEC, CurveSpec, TargetDescriptor, TargetType
from ._registry import register

DESCRIPTOR = TargetDescriptor(
    name="dynamic_arc",
    type=TargetType.CURVE,
    category="curves",
    curve_spec=DEFAULT_CURVE_SPEC,
)

_TARGET_SR: int = 22050
_DB_FLOOR: float = -60.0
_SMOOTHING_SIGMA_FRAMES: float = 3.0
_OVERSAMPLE: int = 4
_EPS: float = 1e-10


def _fit_duration(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Truncate or right-pad ``audio`` with silence to exactly ``n_samples``."""
    current = audio.shape[0]
    if current >= n_samples:
        return audio[:n_samples]
    return np.pad(audio, (0, n_samples - current), mode="constant")


def _mean_pool(values: np.ndarray, n_frames: int) -> np.ndarray:
    """Mean-pool ``values`` into exactly ``n_frames`` buckets.

    Trailing samples that don't evenly fit are dropped so each output
    frame is backed by the same number of input samples — keeps later
    normalization from being biased by an uneven last bin.
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    per_bucket = max(1, values.shape[0] // n_frames)
    trimmed = values[: per_bucket * n_frames]
    pooled: np.ndarray = trimmed.reshape(n_frames, per_bucket).mean(axis=1)
    return pooled


def _normalize_unit_range(curve: np.ndarray) -> np.ndarray:
    """Per-track min-max normalize to ``[0, 1]``; flat curves -> zeros."""
    lo = float(curve.min())
    hi = float(curve.max())
    span = hi - lo
    if span <= _EPS:
        return np.zeros_like(curve, dtype=np.float32)
    return ((curve - lo) / span).astype(np.float32)


def generate(
    audio_path: str | Path,
    curve_spec: CurveSpec = DEFAULT_CURVE_SPEC,
) -> np.ndarray:
    """Produce the per-track-normalized dynamic-arc curve for a track.

    Args:
        audio_path: Path to a readable audio file.
        curve_spec: Fixed frame-rate / duration contract. Must match
            across tracks so probing can stack targets into a matrix.

    Returns:
        Float32 array of shape ``(curve_spec.n_frames,)`` in ``[0, 1]``.
        Silent audio produces an all-zero curve (no crash).
    """
    import librosa  # type: ignore[import-untyped]
    from scipy.ndimage import gaussian_filter1d  # type: ignore[import-untyped]

    from ...extraction.audio import load_audio

    audio = np.asarray(
        load_audio(audio_path, target_sr=_TARGET_SR), dtype=np.float32
    ).reshape(-1)

    n_samples = int(round(curve_spec.duration_s * _TARGET_SR))
    audio = _fit_duration(audio, n_samples)

    n_frames = curve_spec.n_frames
    # 4x oversample so mean-pooling to n_frames averages over a handful
    # of RMS samples rather than landing on single noisy ones.
    hop_length = max(1, n_samples // (n_frames * _OVERSAMPLE))
    frame_length = max(hop_length * 2, 2048)

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]

    pooled = _mean_pool(np.asarray(rms, dtype=np.float64), n_frames)
    db = np.maximum(20.0 * np.log10(pooled + _EPS), _DB_FLOOR)
    smoothed = gaussian_filter1d(db, sigma=_SMOOTHING_SIGMA_FRAMES, mode="nearest")

    return _normalize_unit_range(smoothed)


register("dynamic_arc", DESCRIPTOR, generate)
