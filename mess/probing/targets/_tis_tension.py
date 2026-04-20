"""T1 — Tonal tension curve via Tonal Interval Space (TIS).

Projects chroma onto the 6D Tonal Interval Space of Bernardes et al.
("Conceptual framework for analysis of harmonic content in music", MDPI
Entropy 2020) and reports per-frame tonal tension as the L2 norm of that
projection, normalized within the track.

The TIS basis is the first six DFT bins of a 12-point chroma vector, each
weighted by a perceptually-derived constant ``w_k`` that emphasizes the
harmonic intervals most relevant to Western tonal music (fifths, thirds,
whole-tones, tritones, etc.). In this basis, Euclidean distance between two
chroma vectors approximates their perceived tonal distance, so the norm of
a projected frame is a consistent indicator of harmonic "pull" or tension
relative to a flat/uniform chroma (the silence-like origin).

Reference:
    Bernardes, G., Cocharro, D., Guedes, C., Davies, M. E. P. (2020).
    "Perceptually-motivated Audio Indexing and Classification". MDPI Entropy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._schema import (
    DEFAULT_CURVE_SPEC,
    CurveSpec,
    TargetDescriptor,
    TargetType,
)
from ._registry import register

# Per-harmonic weights from Bernardes et al. for DFT bins 1..6 of the
# 12-point chroma DFT. Bins map to interval classes: fifths, major thirds,
# minor thirds / major sixths, whole tones, semitones / major sevenths,
# tritones.
_TIS_WEIGHTS: np.ndarray = np.array(
    [3.0, 8.0, 11.5, 15.0, 14.5, 7.5], dtype=np.float64
)

# Sample rate for chroma analysis. CQT chroma is SR-agnostic beyond
# Nyquist for pitched music, so we match librosa's default.
_AUDIO_SR: int = 22050
_HOP_LENGTH: int = 512  # ~23 ms at 22.05 kHz.
_EPS: float = 1e-12

DESCRIPTOR = TargetDescriptor(
    name="tis_tension",
    type=TargetType.CURVE,
    category="curves",
    curve_spec=DEFAULT_CURVE_SPEC,
)


def _pool_chroma_to_frames(chroma: np.ndarray, n_frames: int) -> np.ndarray:
    """Average-pool a ``(12, n_hops)`` chroma matrix to ``(12, n_frames)``.

    When the chroma is shorter than ``n_frames`` hops we right-pad with
    zeros (silent tail). When it is longer we pool contiguous windows.
    """
    if chroma.size == 0 or n_frames <= 0:
        return np.zeros((12, max(n_frames, 0)), dtype=np.float64)

    n_hops = chroma.shape[1]
    if n_hops <= n_frames:
        pooled = np.zeros((12, n_frames), dtype=np.float64)
        pooled[:, :n_hops] = chroma
        return pooled

    # Even split: assign each hop to a frame bucket via integer indices.
    bucket = np.minimum(
        (np.arange(n_hops) * n_frames) // n_hops,
        n_frames - 1,
    )
    pooled = np.zeros((12, n_frames), dtype=np.float64)
    counts = np.zeros(n_frames, dtype=np.int64)
    for i in range(n_hops):
        pooled[:, bucket[i]] += chroma[:, i]
        counts[bucket[i]] += 1
    counts = np.maximum(counts, 1)
    pooled /= counts
    return pooled


def _tis_project(chroma_frames: np.ndarray) -> np.ndarray:
    """Project ``(12, n_frames)`` chroma onto the 6D TIS basis.

    Returns ``(6, n_frames)`` real magnitudes of DFT bins 1..6 scaled by
    ``_TIS_WEIGHTS``. Each frame is L1-normalized first so the projection
    is independent of absolute loudness.
    """
    col_sums = chroma_frames.sum(axis=0, keepdims=True)
    safe_sums = np.where(col_sums > 0, col_sums, 1.0)
    normalized = chroma_frames / safe_sums

    # DFT along the 12-point chroma axis. Bins 1..6 are the harmonic
    # components used by TIS; bin 0 is the DC offset (ignored) and bins
    # 7..11 are conjugates of 5..1.
    spectrum = np.fft.fft(normalized, axis=0)[1:7, :]  # (6, n_frames)
    magnitudes = np.abs(spectrum)
    return _TIS_WEIGHTS[:, None] * magnitudes


def _normalize_curve(curve: np.ndarray) -> np.ndarray:
    """Per-track min-max normalize to ``[0, 1]``; constant curves -> zeros."""
    lo = float(curve.min())
    hi = float(curve.max())
    span = hi - lo
    if span < _EPS:
        return np.zeros_like(curve, dtype=np.float32)
    return ((curve - lo) / span).astype(np.float32)


def generate(
    audio_path: str | Path,
    curve_spec: CurveSpec = DEFAULT_CURVE_SPEC,
) -> np.ndarray:
    """Produce a ``(n_frames,)`` tonal-tension curve for ``audio_path``.

    Steps:

    1. Load audio (mono) and truncate/pad to ``curve_spec.duration_s``.
    2. Extract CQT chroma at ~23 ms hops.
    3. Pool to ``curve_spec.n_frames`` columns.
    4. Project each column onto the 6D TIS basis and take the L2 norm.
    5. Min-max normalize the resulting curve within the track.

    Silent audio yields an all-zero curve.
    """
    import librosa  # lazy — heavy dep

    from ...extraction.audio import load_audio

    n_frames = curve_spec.n_frames
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    audio = np.asarray(load_audio(audio_path, target_sr=_AUDIO_SR), dtype=np.float32)
    # Defensive: load_audio returns 1-D, but collapse any stray channel dim.
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)

    max_samples = int(round(curve_spec.duration_s * _AUDIO_SR))
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]
    elif audio.shape[0] < max_samples:
        audio = np.pad(audio, (0, max_samples - audio.shape[0]))

    # Skip CQT work on fully-silent audio — chroma_cqt on pure zeros both
    # wastes compute and logs a librosa warning.
    if not np.any(audio):
        return np.zeros(n_frames, dtype=np.float32)

    chroma = np.asarray(
        librosa.feature.chroma_cqt(
            y=audio, sr=_AUDIO_SR, hop_length=_HOP_LENGTH, n_chroma=12
        ),
        dtype=np.float64,
    )
    pooled = _pool_chroma_to_frames(chroma, n_frames)
    tis = _tis_project(pooled)  # (6, n_frames)
    tension = np.linalg.norm(tis, axis=0)  # (n_frames,)
    return _normalize_curve(tension)


register("tis_tension", DESCRIPTOR, generate)
