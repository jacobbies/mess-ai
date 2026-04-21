"""T8 — MIDI performance-tempo curve target.

A sliding-window local-BPM curve computed from MIDI onset
inter-onset-intervals (IOIs). For each frame at time ``t_i``, we take the
median IOI within a 2 s window centered at ``t_i``, convert to BPM, and
log-scale as ``log2(bpm / 60)``. The log-scale convention matches T3
(``local_tempo``) so the audio-derived and MIDI-derived tempo curves are
directly comparable (tempo doubling = +1, halving = -1, 120 BPM = +1.0).

Where ``local_tempo`` probes audio surface periodicity via a tempogram,
``midi_perf_tempo`` reads the ground-truth note timing. Together they let
layer discovery separate layers that encode *perceived* tempo (from audio)
from layers that encode *performed* tempo (the pianist's actual onsets).

Edge-case policy: when a frame has fewer than 3 onsets in its window, we
carry the previous frame's value forward rather than emit a NaN — the
probe-time mask already drops tracks missing MIDI, and frame-level
dropouts would need per-frame masking that the current Ridge path does
not support. The curve starts at the neutral default (log2(120/60) = 1.0)
and rides from there.
"""

from __future__ import annotations

import numpy as np

from .._midi import align_midi_to_audio, load_midi_for_track
from .._schema import DEFAULT_CURVE_SPEC, CurveSpec, TargetDescriptor, TargetType
from ._registry import register

DESCRIPTOR = TargetDescriptor(
    name="midi_perf_tempo",
    type=TargetType.MIDI_CURVE,
    category="midi",
    curve_spec=DEFAULT_CURVE_SPEC,
)

# 2 s sliding window around each frame centre. Wide enough to include
# several onsets at slow tempi (40 BPM = 1.5 s IOI) but narrow enough to
# track rubato within a phrase.
_WINDOW_S = 2.0
_BPM_MIN = 40.0
_BPM_MAX = 240.0
_DEFAULT_LOG_BPM = float(np.log2(120.0 / 60.0))  # 1.0 — neutral fallback.
# Length-3 median filter smooths single-frame octave jumps (e.g. when a
# window briefly sees only one subdivision's worth of onsets and halves
# the BPM estimate).
_MEDIAN_KERNEL = 3


def generate(
    track_id: str,
    dataset: str,
    curve_spec: CurveSpec = DEFAULT_CURVE_SPEC,
) -> np.ndarray | None:
    """Produce a ``(n_frames,)`` MIDI performance-tempo curve in log2(BPM/60).

    Parameters
    ----------
    track_id:
        Audio stem; the MIDI file is resolved by :func:`load_midi_for_track`.
    dataset:
        ``"smd"`` or ``"maestro"``; forwarded to the MIDI loader.
    curve_spec:
        Shape contract. Defaults to the 60-frame / 30 s grid shared by all
        curve targets so discovery can stack them.

    Returns
    -------
    np.ndarray | None
        Shape ``(curve_spec.n_frames,)`` float32 in log2(BPM/60), or
        ``None`` when MIDI is missing or fewer than 4 onsets fall in the
        30 s window (insufficient data to estimate a local IOI).
    """
    midi = load_midi_for_track(track_id, dataset)
    if midi is None:
        return None

    aligned = align_midi_to_audio(midi, curve_spec.duration_s, curve_spec)
    onsets = aligned["onsets"]
    if onsets.size < 4:
        return None

    n_frames = curve_spec.n_frames
    times = (np.arange(n_frames) + 0.5) / curve_spec.frame_rate_hz
    half = _WINDOW_S / 2.0

    curve = np.empty(n_frames, dtype=float)
    prev = _DEFAULT_LOG_BPM
    for i, t in enumerate(times):
        window = onsets[(onsets >= t - half) & (onsets <= t + half)]
        # Need >=3 onsets in the window to form >=2 IOIs and take a stable
        # median. Below that, carry the previous frame forward.
        if window.size >= 3:
            iois = np.diff(window)
            median_ioi = float(np.median(iois))
            if median_ioi > 0:
                bpm = np.clip(60.0 / median_ioi, _BPM_MIN, _BPM_MAX)
                prev = float(np.log2(bpm / 60.0))
        curve[i] = prev

    # Length-3 median filter with edge padding to smooth residual octave
    # jumps (matches the T3 local_tempo post-filter convention).
    padded = np.pad(curve, _MEDIAN_KERNEL // 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, _MEDIAN_KERNEL)
    smoothed = np.asarray(np.median(windows, axis=1))
    return smoothed.astype(np.float32)


register("midi_perf_tempo", DESCRIPTOR, generate)
