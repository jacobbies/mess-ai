"""
T6 — MIDI articulation histogram target generator.

For each note i in a MIDI performance, the *articulation ratio* is defined as::

    articulation_i = (offset_i - onset_i) / (onset_{i+1} - onset_i)

i.e. note duration divided by the inter-onset interval (IOI) to the next note
(see Widmer & Goebl 2004; Cancino-Chacón 2018). Values near 0 indicate
staccato (short note, long gap), values near 1 indicate legato (note holds
until the next onset), and values above 1 indicate overlap/pedaling.

Rather than collapsing the ratios to a single scalar, we bin them into a
16-bin histogram over ``[0, 1.5]`` and normalize to a probability
distribution. This preserves the *shape* of the articulation mix within a
passage (e.g. mostly-staccato vs mostly-legato vs uniformly-mixed).

Shape contract — we reuse :class:`CurveSpec` to carry the 16-dim shape
through the existing multi-output Ridge path. The spec's ``n_frames`` is
interpreted as "histogram bins" here; that is a slight semantic stretch but
it keeps the probing dispatch uniform across all ``MIDI_CURVE`` targets.
"""

from __future__ import annotations

import numpy as np

from .._midi import align_midi_to_audio, load_midi_for_track
from .._schema import CurveSpec, TargetDescriptor, TargetType
from ._registry import register

# 16 "frames" (= histogram bins) over a nominal 1 s span — ``duration_s`` is
# unused for histograms but must be set so that ``n_frames`` == 16.
ARTICULATION_SPEC = CurveSpec(frame_rate_hz=16.0, duration_s=1.0)
N_BINS = ARTICULATION_SPEC.n_frames
ARTICULATION_RANGE = (0.0, 1.5)
BIN_EDGES = np.linspace(ARTICULATION_RANGE[0], ARTICULATION_RANGE[1], N_BINS + 1)

# Guards against division by ~zero for simultaneous onsets (chord tones)
# without silently folding them into the last bin.
_MIN_IOI_S = 1e-3

# Only onsets/offsets are used from ``align_midi_to_audio``; keep the
# pedal-curve resolution low so we don't pay for an unused 128-row piano roll.
_ALIGN_FRAME_RATE_HZ = 2.0


DESCRIPTOR = TargetDescriptor(
    name="midi_articulation_hist",
    type=TargetType.MIDI_CURVE,
    category="midi",
    curve_spec=ARTICULATION_SPEC,
)


def _uniform() -> np.ndarray:
    """Return a 16-bin uniform probability distribution as float32."""
    return np.full(N_BINS, 1.0 / N_BINS, dtype=np.float32)


def generate(
    track_id: str,
    dataset: str,
    audio_duration_s: float = 30.0,
) -> np.ndarray | None:
    """Return a 16-bin articulation histogram for ``track_id``.

    Parameters
    ----------
    track_id:
        Audio stem (the MIDI file is resolved by :func:`load_midi_for_track`).
    dataset:
        ``"smd"`` or ``"maestro"``; forwarded to the MIDI loader.
    audio_duration_s:
        Window (seconds from the start) over which to compute the histogram.
        Notes after this window are dropped so the target aligns with the
        30 s segment used for MERT embedding aggregation.

    Returns
    -------
    np.ndarray | None
        Shape ``(16,)`` float32 summing to ~1.0, or ``None`` when no MIDI is
        available for ``track_id`` (caller handles via ``midi_available``).
        Falls back to a uniform distribution when fewer than 2 notes fall in
        the window (insufficient data to compute IOIs).
    """
    midi = load_midi_for_track(track_id, dataset)
    if midi is None:
        return None

    align_spec = CurveSpec(
        frame_rate_hz=_ALIGN_FRAME_RATE_HZ, duration_s=audio_duration_s
    )
    aligned = align_midi_to_audio(midi, audio_duration_s, align_spec)
    onsets = np.asarray(aligned["onsets"], dtype=float)
    offsets = np.asarray(aligned["offsets"], dtype=float)

    # <2 notes: no IOI available. Return uniform as a neutral prior.
    if onsets.size < 2:
        return _uniform()

    durations = offsets[:-1] - onsets[:-1]
    iois = np.diff(onsets)
    ratios = durations / np.maximum(iois, _MIN_IOI_S)
    ratios = np.clip(ratios, ARTICULATION_RANGE[0], ARTICULATION_RANGE[1])

    counts, _ = np.histogram(ratios, bins=BIN_EDGES)
    total = counts.sum()
    if total == 0:
        return _uniform()
    return (counts / float(total)).astype(np.float32)


register("midi_articulation_hist", DESCRIPTOR, generate)
