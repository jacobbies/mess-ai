"""
Target schema for probing: SCALAR, CURVE, MIDI_SCALAR, MIDI_CURVE.

This module is pure-Python and free of heavy ML/audio imports so it can be
consumed by any part of the probing package (including target generator
registries in T1-T8 units) without pulling in sklearn/librosa/pretty_midi.

Extended .npz layout (backwards compatible with existing files):

    * Existing scalar categories: ``rhythm``, ``harmony``, ``timbre``,
      ``dynamics``, ``articulation``, ``phrasing``, ``expression`` — each is
      a pickled dict ``{field: np.ndarray}`` (no change).
    * New ``curves`` top-level key: ``{target_name: np.ndarray[T]}`` where
      ``T = frame_rate_hz * duration_s``. Frame rate and duration are fixed
      across tracks so probing can stack into a matrix.
    * New ``midi`` top-level key: ``{target_name: np.ndarray}`` (scalar or
      curve). Accompany with a top-level boolean key ``midi_available`` so
      probing can mask tracks without MIDI.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class TargetType(Enum):
    """Kind of target value encoded in a .npz file."""

    SCALAR = "scalar"
    CURVE = "curve"
    MIDI_SCALAR = "midi_scalar"
    MIDI_CURVE = "midi_curve"


@dataclass(frozen=True)
class CurveSpec:
    """Shape contract for curve-valued targets.

    ``n_frames`` is fixed across tracks so multi-output Ridge can stack
    targets into a matrix.
    """

    frame_rate_hz: float = 2.0
    duration_s: float = 30.0

    @property
    def n_frames(self) -> int:
        return int(round(self.frame_rate_hz * self.duration_s))


DEFAULT_CURVE_SPEC = CurveSpec()


@dataclass(frozen=True)
class TargetDescriptor:
    """Declarative entry for a probing target.

    Each T1-T8 unit registers its target through this descriptor; probing
    dispatches on ``type`` to pick the right loader and probe.
    """

    name: str
    type: TargetType
    category: str
    curve_spec: CurveSpec | None = None


def _is_midi(target_type: TargetType) -> bool:
    return target_type in (TargetType.MIDI_SCALAR, TargetType.MIDI_CURVE)


def _is_curve(target_type: TargetType) -> bool:
    return target_type in (TargetType.CURVE, TargetType.MIDI_CURVE)


def _midi_available(data: Any) -> bool:
    """Return whether the .npz declares midi_available=True.

    Treats missing key as True (back-compat: if a writer stored MIDI-derived
    values it is expected to flag availability, but older files predating the
    flag should not be silently discarded).
    """
    try:
        flag = data["midi_available"]
    except (KeyError, ValueError):
        return True
    arr = np.asarray(flag)
    try:
        return bool(arr.item())
    except (ValueError, AttributeError):
        return bool(arr.all()) if arr.size else True


def _unpack_category(data: Any, key: str) -> dict | None:
    """Return ``data[key].item()`` as a dict, or ``None`` if the key is
    missing or the payload is not a dict."""
    try:
        raw = data[key].item()
    except (KeyError, ValueError, AttributeError):
        return None
    return raw if isinstance(raw, dict) else None


def load_target_field(
    npz_path: str | Path,
    descriptor: TargetDescriptor,
) -> np.ndarray | None:
    """Load a single target field from a proxy-target ``.npz``.

    Dispatches on ``descriptor.type``:

    * ``SCALAR``: returns ``data[category].item()[name]`` coerced to a
      1-d float array.
    * ``CURVE``: returns ``data['curves'].item()[name]`` as a 1-d array of
      shape ``(curve_spec.n_frames,)``.
    * ``MIDI_SCALAR`` / ``MIDI_CURVE``: returns the value from
      ``data['midi'].item()[name]``. Returns ``None`` if the track's
      ``midi_available`` flag is ``False`` or the key is missing.

    Returns ``None`` if the file or field is missing (missing MIDI for a
    track is not an error — many tracks legitimately lack MIDI).
    """
    path = Path(npz_path)
    if not path.exists():
        return None

    with np.load(path, allow_pickle=True) as data:
        if _is_midi(descriptor.type):
            if not _midi_available(data):
                return None
            payload_key = "midi"
        elif descriptor.type is TargetType.CURVE:
            payload_key = "curves"
        else:
            payload_key = descriptor.category

        payload = _unpack_category(data, payload_key)
        value = payload.get(descriptor.name) if payload is not None else None

    if value is None:
        return None

    arr = np.asarray(value, dtype=float)
    if _is_curve(descriptor.type) and descriptor.curve_spec is not None:
        if arr.shape != (descriptor.curve_spec.n_frames,):
            return None
    return arr
