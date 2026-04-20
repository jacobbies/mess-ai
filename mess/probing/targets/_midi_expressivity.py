"""MIDI velocity + IOI + pedal expressivity scalars (T7).

Fulfils the long-standing ``rubato`` / ``expressiveness`` promise from
``ASPECT_REGISTRY`` — those aspects advertised MIDI-derived features but were
never wired up. This module registers three MIDI-backed scalar targets:

* ``midi_velocity_std`` — stddev of note velocities in ``[0, 30s]``. Larger
  values indicate more dynamic expression (wider touch range).
* ``midi_ioi_std`` — stddev of ``log(inter-onset-interval)``. Larger values
  indicate more rubato / tempo flexibility.
* ``midi_pedal_ratio`` — mean of the per-frame pedal-on curve (fraction of
  the window where the sustain pedal is engaged).

The features are extracted from the same ``align_midi_to_audio`` call so the
generator does one MIDI decode per target triplet when batch-driven.
Generators return ``None`` when MIDI is missing or the note population is
too sparse (<2 notes). Missing MIDI is not an error — many tracks lack a
ground-truth MIDI file.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .._midi import align_midi_to_audio, load_midi_for_track
from .._schema import CurveSpec, TargetDescriptor, TargetType
from ._registry import register

# Matches the audio-side curve-target window (first 30 s at 2 Hz) so MIDI
# scalars align with what the curve probes are seeing.
_ALIGN_SPEC = CurveSpec(frame_rate_hz=2.0, duration_s=30.0)

# Small floor so ``log(ioi)`` stays finite if two onsets land within the
# same microsecond (the ``iois > 0`` filter should already remove true
# zeros; this is a numerical belt-and-braces).
_LOG_IOI_EPS: float = 1e-6

_FEATURE_NAMES: tuple[str, ...] = (
    "midi_velocity_std",
    "midi_ioi_std",
    "midi_pedal_ratio",
)


def _features(track_id: str, dataset: str) -> dict[str, float] | None:
    """Extract all three MIDI expressivity scalars in a single MIDI decode.

    Returns ``None`` if MIDI is unavailable or fewer than two notes fall in
    the 30-second window (stddev is undefined with <2 samples).
    """
    midi = load_midi_for_track(track_id, dataset)
    if midi is None:
        return None

    aligned = align_midi_to_audio(midi, _ALIGN_SPEC.duration_s, _ALIGN_SPEC)
    velocities = aligned["velocities"]
    onsets = aligned["onsets"]
    pedal = aligned["pedal_curve"]

    if velocities.size < 2:
        return None

    # Drop non-positive IOIs — simultaneous onsets would yield log(0).
    iois = np.diff(onsets)
    iois = iois[iois > 0]
    ioi_std = float(np.std(np.log(iois + _LOG_IOI_EPS))) if iois.size else 0.0

    return {
        "midi_velocity_std": float(np.std(velocities)),
        "midi_ioi_std": ioi_std,
        "midi_pedal_ratio": float(np.mean(pedal)) if pedal.size else 0.0,
    }


def _make_generator(
    feature_name: str,
) -> Callable[[str, str], np.ndarray | None]:
    """Build a single-feature generator closure.

    Each T7 target shares the same underlying ``_features`` call; the
    closure picks out one field and wraps it in a 1-element array so
    downstream probing code can treat every scalar target uniformly.
    """

    def generate(track_id: str, dataset: str) -> np.ndarray | None:
        feats = _features(track_id, dataset)
        if feats is None:
            return None
        return np.asarray([feats[feature_name]], dtype=float)

    generate.__name__ = f"generate_{feature_name}"
    return generate


DESCRIPTORS: dict[str, TargetDescriptor] = {
    name: TargetDescriptor(
        name=name,
        type=TargetType.MIDI_SCALAR,
        category="midi",
        curve_spec=None,
    )
    for name in _FEATURE_NAMES
}


for _name in _FEATURE_NAMES:
    register(_name, DESCRIPTORS[_name], _make_generator(_name))
