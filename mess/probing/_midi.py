"""
MIDI ingestion utilities for probing.

Pure helpers — no probing logic. T1-T8 target generators consume these to
extract ground-truth note events, velocities, and pedal activity from MIDI
files that accompany the audio.

``pretty_midi`` is lazy-imported so ``from mess.probing import ...`` works
without the ``ml`` extra. Callers that actually touch MIDI need the extra
installed (it's bundled with ``mess-ai[ml]``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import mess_config
from ._schema import CurveSpec

PEDAL_CC = 64  # MIDI Control Change for sustain pedal.
PEDAL_ON_THRESHOLD = 64  # Standard threshold for "pedal down" (>= 64).


def _import_pretty_midi() -> Any:
    """Lazy-import pretty_midi with a helpful error if the extra is missing."""
    try:
        import pretty_midi  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised when extra missing
        raise ModuleNotFoundError(
            "pretty_midi is required for MIDI-backed probing targets. "
            "Install with `mess-ai[ml]`."
        ) from exc
    return pretty_midi


def _smd_midi_path(track_id: str) -> Path:
    return mess_config.data_root / "audio" / "smd" / "midi" / f"{track_id}.mid"


def _maestro_midi_path(track_id: str) -> Path | None:
    """Resolve a MAESTRO MIDI path given the audio stem.

    MAESTRO lays out files as ``data/audio/maestro/<year>/<id>.wav`` paired
    with ``<id>.midi``. We search across year subdirectories because the year
    is not encoded in the track id itself.
    """
    maestro_root = mess_config.data_root / "audio" / "maestro"
    if not maestro_root.exists():
        return None
    # Scan year subdirectories (cheap: a few dozen dirs, not recursive).
    for year_dir in sorted(maestro_root.iterdir()):
        if not year_dir.is_dir():
            continue
        candidate = year_dir / f"{track_id}.midi"
        if candidate.exists():
            return candidate
    return None


def load_midi_for_track(
    track_id: str,
    dataset: str,
) -> Any | None:  # returns pretty_midi.PrettyMIDI | None
    """Return the ``PrettyMIDI`` for ``track_id`` or ``None`` if not found.

    Supports ``dataset`` values ``"smd"`` and ``"maestro"``. A missing file
    is NOT an error — many tracks legitimately lack MIDI ground truth.
    """
    dataset = dataset.lower()
    if dataset == "smd":
        path = _smd_midi_path(track_id)
    elif dataset == "maestro":
        resolved = _maestro_midi_path(track_id)
        if resolved is None:
            return None
        path = resolved
    else:
        raise ValueError(f"Unknown dataset {dataset!r}; expected 'smd' or 'maestro'.")

    if not path.exists():
        return None

    pretty_midi = _import_pretty_midi()
    return pretty_midi.PrettyMIDI(str(path))


def _pedal_curve_from_instruments(
    instruments: list[Any],
    audio_duration_s: float,
    curve_spec: CurveSpec,
) -> np.ndarray:
    """Compute per-frame pedal-on fraction from CC64 events.

    Each frame holds the mean "pedal is down" indicator over the frame's
    time span, in ``[0, 1]``. When multiple instruments contribute pedal
    events (rare but valid) we OR their states before averaging.
    """
    n_frames = curve_spec.n_frames
    if n_frames <= 0:
        return np.zeros(0, dtype=float)

    frame_dt = 1.0 / curve_spec.frame_rate_hz
    # Sub-sample each frame on a fine grid then average — cheap and avoids
    # pitfalls from integrating CC-edge step functions.
    n_subframes = max(4, int(round(frame_dt * 100)))  # 10 ms resolution.
    subframe_dt = frame_dt / n_subframes
    times = (
        np.arange(n_frames * n_subframes, dtype=float) + 0.5
    ) * subframe_dt

    combined = np.zeros_like(times, dtype=bool)
    for instrument in instruments:
        events = sorted(
            (cc for cc in getattr(instrument, "control_changes", []) if cc.number == PEDAL_CC),
            key=lambda cc: cc.time,
        )
        if not events:
            continue
        # Build a step function: pedal state at time t = most recent CC <= t.
        event_times = np.array([cc.time for cc in events], dtype=float)
        event_states = np.array(
            [cc.value >= PEDAL_ON_THRESHOLD for cc in events], dtype=bool
        )
        idx = np.searchsorted(event_times, times, side="right") - 1
        state = np.zeros_like(times, dtype=bool)
        valid = idx >= 0
        state[valid] = event_states[idx[valid]]
        combined |= state

    # Clamp to [0, audio_duration_s]: frames past the audio are 0.
    if audio_duration_s < n_frames * frame_dt:
        combined[times >= audio_duration_s] = False

    curve = combined.reshape(n_frames, n_subframes).mean(axis=1).astype(float)
    return np.asarray(curve)


def align_midi_to_audio(
    midi: Any,  # pretty_midi.PrettyMIDI
    audio_duration_s: float,
    curve_spec: CurveSpec,
) -> dict[str, np.ndarray]:
    """Align a ``PrettyMIDI`` object to a fixed-length frame grid.

    Returns a dict with keys:

    * ``piano_roll``: ``(128, n_frames)`` — ``pretty_midi.get_piano_roll``
      sampled at ``curve_spec.frame_rate_hz``, truncated/padded to
      ``n_frames``.
    * ``onsets``: onset times (s) within ``[0, audio_duration_s]``.
    * ``offsets``: matching offset times (clipped to audio duration).
    * ``velocities``: matching velocities (``0..127``).
    * ``pedal_curve``: ``(n_frames,)`` — pedal-on fraction per frame from
      CC64 events across all instruments.
    """
    n_frames = curve_spec.n_frames
    frame_rate = curve_spec.frame_rate_hz

    # Piano roll: pretty_midi returns shape (128, T) where T depends on fs
    # and the MIDI's internal duration. We crop/pad to our fixed grid.
    raw_roll = np.asarray(midi.get_piano_roll(fs=frame_rate))  # (128, T_raw)
    piano_roll = np.zeros((128, n_frames), dtype=float)
    t_keep = min(raw_roll.shape[1], n_frames)
    if t_keep > 0:
        piano_roll[:, :t_keep] = raw_roll[:, :t_keep]

    onsets: list[float] = []
    offsets: list[float] = []
    velocities: list[int] = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            if note.start > audio_duration_s:
                continue
            onsets.append(float(note.start))
            offsets.append(float(min(note.end, audio_duration_s)))
            velocities.append(int(note.velocity))

    onset_arr = np.asarray(onsets, dtype=float)
    offset_arr = np.asarray(offsets, dtype=float)
    velocity_arr = np.asarray(velocities, dtype=int)

    # Sort by onset so downstream consumers can assume monotonic ordering.
    if onset_arr.size:
        order = np.argsort(onset_arr, kind="stable")
        onset_arr = onset_arr[order]
        offset_arr = offset_arr[order]
        velocity_arr = velocity_arr[order]

    pedal_curve = _pedal_curve_from_instruments(
        list(midi.instruments), audio_duration_s, curve_spec
    )

    return {
        "piano_roll": piano_roll,
        "onsets": onset_arr,
        "offsets": offset_arr,
        "velocities": velocity_arr,
        "pedal_curve": pedal_curve,
    }
