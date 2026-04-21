"""T5 — Foote novelty curve target.

Produces a per-track-normalized structural novelty curve of shape
``(curve_spec.n_frames,)`` that peaks where the timbral/harmonic content
of the audio changes abruptly — i.e. at likely phrase, section, or
movement boundaries. This is a new structural aspect (replaces nothing)
and supports a future ``structure`` aspect enabling retrieval like
"find passages with similar structural contour".

Pipeline:
    wav -> mono @ sr -> truncate/pad to duration_s
        -> MFCC (13, T)   -> SSM -> checkerboard novelty -> [0,1]
        -> chroma_cqt (12, T) -> SSM -> checkerboard novelty -> [0,1]
        -> mean of the two novelty curves -> mean-pool to n_frames.

The checkerboard kernel (Foote 1999) is the classical detector for
"corners" along the SSM diagonal: a 2x2 block with ``+, -, -, +`` signs
weighted by a Gaussian radial taper. Sliding it along the diagonal
scores each frame by how self-similar the immediately-past window is
with itself, *and* how self-similar the immediately-future window is
with itself, *while* simultaneously being dissimilar across the divide.
Local peaks correspond to structural transitions.

Reference:
    Foote, J. (1999). "Visualizing Music and Audio using Self-Similarity".
    Proc. ACM Multimedia, 77-80.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._schema import DEFAULT_CURVE_SPEC, CurveSpec, TargetDescriptor, TargetType
from ._registry import register

DESCRIPTOR = TargetDescriptor(
    name="novelty",
    type=TargetType.CURVE,
    category="curves",
    curve_spec=DEFAULT_CURVE_SPEC,
)

# Audio loading sample rate — matches the other curve targets (T1, T2).
_AUDIO_SR: int = 22050

# Hop length for MFCC / chroma frames (~46 ms at 22.05 kHz). Coarser than
# T1's 512-hop chroma: structural novelty is a slow-timescale phenomenon
# and the SSM scales as O(T^2) in both memory and convolution cost.
_HOP_LENGTH: int = 1024

# Checkerboard kernel half-size in frames. ``2 * _KERNEL_RADIUS`` must be
# small enough that the kernel fits inside the SSM; ``8`` gives a 16x16
# kernel, matching Foote's classical size.
_KERNEL_RADIUS: int = 8

# Minimum SSM side length (frames) for the novelty computation to be
# well-defined. Below this the kernel cannot slide, so we return zeros.
_MIN_SSM_FRAMES: int = 2 * _KERNEL_RADIUS

_EPS: float = 1e-10


def _load_mono_audio(audio_path: str | Path, target_sr: int) -> np.ndarray:
    """Load audio as mono float32 at ``target_sr``."""
    from ...extraction.audio import load_audio

    audio = load_audio(audio_path, target_sr=target_sr)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _fit_duration(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Truncate or right-pad ``audio`` with silence to exactly ``n_samples``."""
    current = audio.shape[0]
    if current == n_samples:
        return audio
    if current > n_samples:
        return audio[:n_samples]
    return np.pad(audio, (0, n_samples - current), mode="constant")


def _checkerboard_kernel(radius: int) -> np.ndarray:
    """Build a Gaussian-tapered checkerboard kernel of size ``2*radius``.

    The kernel at position ``(i, j)`` (with ``i, j`` in ``[-radius, radius)``)
    has sign ``sign(i) * sign(j)`` and magnitude ``exp(-(i^2 + j^2) / sigma^2)``.
    This gives the classical "+, -, -, +" 2x2 block pattern that responds to
    corners in the SSM along its diagonal (Foote 1999).
    """
    # Cell-centered offsets (no zero axis) so every cell has a definite sign.
    axis = np.concatenate(
        [
            np.arange(-radius, 0, dtype=np.float64),
            np.arange(1, radius + 1, dtype=np.float64),
        ]
    )
    ii, jj = np.meshgrid(axis, axis, indexing="ij")
    # sigma = radius gives ~37% weight at the corners.
    sigma = float(radius)
    taper = np.exp(-(ii * ii + jj * jj) / (sigma * sigma))
    signs = np.sign(ii) * np.sign(jj)
    return np.asarray(signs * taper, dtype=np.float64)


def _self_similarity(features: np.ndarray) -> np.ndarray:
    """Build a cosine self-similarity matrix from ``(D, T)`` features.

    Each column is L2-normalized (with an eps floor so silent columns are
    zero-similarity rather than NaN), and the SSM is the column-wise Gram
    matrix. Result is ``(T, T)`` with values in ``[-1, 1]``.
    """
    norms = np.linalg.norm(features, axis=0, keepdims=True)
    normalized = features / np.maximum(norms, _EPS)
    ssm = normalized.T @ normalized
    return np.asarray(ssm, dtype=np.float64)


def _novelty_from_ssm(ssm: np.ndarray, radius: int) -> np.ndarray:
    """Slide a checkerboard kernel along the SSM diagonal -> novelty curve.

    Returns a length-``T`` curve. Positions where the kernel cannot fit
    (the first and last ``radius`` frames) are left at zero; the middle
    positions contain the sum of ``kernel * ssm_window``.
    """
    t = ssm.shape[0]
    curve = np.zeros(t, dtype=np.float64)
    size = 2 * radius
    if t < size:
        return curve
    kernel = _checkerboard_kernel(radius)
    # Vectorized Foote 1999: each diagonal window (2R, 2R) centred on
    # (n, n) becomes one row of ``windows``; the kernel is broadcast and
    # summed over the spatial axes in a single tensor reduction.
    windows = np.lib.stride_tricks.sliding_window_view(ssm, (size, size))
    # sliding_window_view gives shape (T-size+1, T-size+1, size, size);
    # the diagonal windows are the ones where both leading indices agree.
    n_diag = t - size + 1
    diag_idx = np.arange(n_diag)
    diag_windows = windows[diag_idx, diag_idx]  # (n_diag, size, size)
    scores = np.einsum("nij,ij->n", diag_windows, kernel)
    curve[radius : radius + n_diag] = scores
    return curve


def _normalize_unit_range(curve: np.ndarray) -> np.ndarray:
    """Divide by ``max(|curve|)`` so peaks sit in ``[0, 1]``; flat -> zeros.

    Foote novelty is a signed score, but for fusion we only care about
    relative peak heights. Rescaling by the max absolute value keeps the
    sign structure while making the two feature-novelty curves comparable.
    """
    peak = float(np.max(curve))
    if peak <= _EPS:
        return np.zeros_like(curve, dtype=np.float64)
    return curve / peak


def _mean_pool(values: np.ndarray, n_frames: int) -> np.ndarray:
    """Mean-pool ``values`` into exactly ``n_frames`` buckets.

    If ``values`` is shorter than ``n_frames``, right-pad with zeros.
    Otherwise each bucket averages a contiguous slice of equal size
    (any trailing samples that don't fit evenly are dropped — keeps
    every output frame backed by the same input-sample count).
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if values.shape[0] == 0:
        return np.zeros(n_frames, dtype=np.float32)
    if values.shape[0] < n_frames:
        padded = np.zeros(n_frames, dtype=np.float64)
        padded[: values.shape[0]] = values
        return padded.astype(np.float32)
    per_bucket = values.shape[0] // n_frames
    trimmed = values[: per_bucket * n_frames]
    pooled = trimmed.reshape(n_frames, per_bucket).mean(axis=1)
    return np.asarray(pooled, dtype=np.float32)


def generate(
    audio_path: str | Path,
    curve_spec: CurveSpec = DEFAULT_CURVE_SPEC,
) -> np.ndarray:
    """Produce the Foote structural-novelty curve for a track.

    Args:
        audio_path: Path to a readable audio file.
        curve_spec: Fixed frame-rate / duration contract. Must match
            across tracks so probing can stack targets into a matrix.

    Returns:
        Float32 array of shape ``(curve_spec.n_frames,)`` in ``[0, 1]``.
        Silent audio (or audio too short for the kernel) produces an
        all-zero curve (no crash).
    """
    import librosa  # type: ignore[import-untyped]  # lazy — heavy dep

    n_frames = curve_spec.n_frames
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    audio = _load_mono_audio(audio_path, target_sr=_AUDIO_SR)
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)

    n_samples = int(round(curve_spec.duration_s * _AUDIO_SR))
    audio = _fit_duration(audio, n_samples)

    # Short-circuit on fully-silent audio: chroma_cqt of silence produces
    # a NaN-y CQT that we'd rather not propagate.
    if not np.any(np.abs(audio) > 1e-8):
        return np.zeros(n_frames, dtype=np.float32)

    mfcc = np.asarray(
        librosa.feature.mfcc(
            y=audio,
            sr=_AUDIO_SR,
            n_mfcc=13,
            hop_length=_HOP_LENGTH,
        ),
        dtype=np.float64,
    )
    chroma = np.asarray(
        librosa.feature.chroma_cqt(
            y=audio,
            sr=_AUDIO_SR,
            hop_length=_HOP_LENGTH,
        ),
        dtype=np.float64,
    )

    # Align the two feature sequences to a common length (librosa can
    # return off-by-one frame counts between MFCC and chroma_cqt).
    t = min(mfcc.shape[1], chroma.shape[1])
    if t < _MIN_SSM_FRAMES:
        return np.zeros(n_frames, dtype=np.float32)
    mfcc = mfcc[:, :t]
    chroma = chroma[:, :t]

    mfcc_nov = _normalize_unit_range(
        _novelty_from_ssm(_self_similarity(mfcc), _KERNEL_RADIUS)
    )
    chroma_nov = _normalize_unit_range(
        _novelty_from_ssm(_self_similarity(chroma), _KERNEL_RADIUS)
    )

    fused = 0.5 * mfcc_nov + 0.5 * chroma_nov
    return _mean_pool(fused, n_frames)


register("novelty", DESCRIPTOR, generate)
