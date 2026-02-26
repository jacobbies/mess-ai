"""
MIDI-derived expressive proxy targets for layer discovery.

Computes expression-related features from MIDI note data that capture
*how* notes are played (rubato, velocity dynamics, articulation) rather
than *what* notes are played. These complement audio-derived proxy targets
by providing ground truth from the symbolic score layer.

Data Structure Contract:
    Returns {'expression': {field: np.ndarray}} compatible with the NPZ
    format consumed by discovery.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class MidiExpressionTargets:
    """Generate expressive proxy targets from MIDI performance data.

    Extracts timing, velocity, and articulation features from MIDI note events
    that reflect performance expression rather than compositional content.
    """

    def __init__(self, min_notes: int = 10):
        """
        Args:
            min_notes: Minimum number of notes required to compute targets.
                       Files with fewer notes raise ValueError.
        """
        self.min_notes = min_notes

    def generate_expression_targets(
        self, midi_path: str | Path,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Generate all expression targets from a MIDI file.

        Args:
            midi_path: Path to a MIDI file (.mid or .midi).

        Returns:
            ``{'expression': {target_name: np.ndarray}}`` nested dict.

        Raises:
            ValueError: If MIDI has fewer than *min_notes* notes.
            FileNotFoundError: If *midi_path* does not exist.
        """
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(str(midi_path))
        notes = self._collect_notes(pm)

        if len(notes) < self.min_notes:
            raise ValueError(
                f"MIDI file has {len(notes)} notes, need >= {self.min_notes}"
            )

        onsets = np.array([n.start for n in notes])
        offsets = np.array([n.end for n in notes])
        velocities = np.array([n.velocity for n in notes], dtype=np.float64)
        durations = offsets - onsets

        # Inter-onset intervals
        ioi = np.diff(onsets)
        ioi_positive = ioi[ioi > 0]  # filter simultaneous notes (chords)

        return {
            "expression": {
                "rubato": self._compute_rubato(ioi_positive),
                "velocity_mean": np.array([np.mean(velocities)]),
                "velocity_std": np.array([np.std(velocities)]),
                "velocity_range": np.array(
                    [float(np.max(velocities) - np.min(velocities))]
                ),
                "articulation_ratio": self._compute_articulation_ratio(durations, ioi),
                "tempo_variability": self._compute_tempo_variability(ioi_positive),
                "onset_timing_std": self._compute_onset_timing_std(onsets),
            }
        }

    @staticmethod
    def _collect_notes(pm: object) -> list:
        """Collect all notes across all instruments, sorted by onset time."""
        all_notes = []
        for instrument in pm.instruments:  # type: ignore[attr-defined]
            if not instrument.is_drum:
                all_notes.extend(instrument.notes)
        all_notes.sort(key=lambda n: n.start)
        return all_notes

    @staticmethod
    def _compute_rubato(ioi_positive: np.ndarray) -> np.ndarray:
        """Std of consecutive IOI ratios — measures micro-timing deviation."""
        if len(ioi_positive) < 3:
            return np.array([0.0])
        ratios = ioi_positive[:-1] / ioi_positive[1:]
        return np.array([float(np.std(ratios))])

    @staticmethod
    def _compute_articulation_ratio(
        durations: np.ndarray, ioi: np.ndarray,
    ) -> np.ndarray:
        """Mean of (note_duration / IOI).  <1 = staccato, >1 = legato."""
        paired_durations = durations[:-1]  # all but last note
        mask = ioi > 0
        if not np.any(mask):
            return np.array([1.0])
        ratios = paired_durations[mask] / ioi[mask]
        return np.array([float(np.mean(ratios))])

    @staticmethod
    def _compute_tempo_variability(ioi_positive: np.ndarray) -> np.ndarray:
        """Std of local tempo estimates from IOI (BPM)."""
        if len(ioi_positive) < 2:
            return np.array([0.0])
        local_tempos = 60.0 / ioi_positive
        return np.array([float(np.std(local_tempos))])

    @staticmethod
    def _compute_onset_timing_std(onsets: np.ndarray) -> np.ndarray:
        """Std of onset deviations from a median-IOI quantized grid."""
        if len(onsets) < 4:
            return np.array([0.0])
        ioi = np.diff(onsets)
        ioi_positive = ioi[ioi > 0]
        if len(ioi_positive) < 2:
            return np.array([0.0])

        grid_spacing = float(np.median(ioi_positive))
        if grid_spacing <= 0:
            return np.array([0.0])

        grid_positions = np.round(onsets / grid_spacing) * grid_spacing
        deviations = onsets - grid_positions
        return np.array([float(np.std(deviations))])


def resolve_midi_path(audio_path: str | Path, dataset_id: str) -> Path | None:
    """Find the MIDI file corresponding to an audio file.

    Args:
        audio_path: Path to the .wav audio file.
        dataset_id: Dataset identifier ('smd' or 'maestro').

    Returns:
        Path to the matching MIDI file, or None if not found.
    """
    audio_path = Path(audio_path)
    stem = audio_path.stem

    if dataset_id == "smd":
        # SMD: MIDI files in sibling midi/ directory
        midi_dir = audio_path.parent.parent / "midi"
        for ext in (".mid", ".midi"):
            candidate = midi_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate

    elif dataset_id == "maestro":
        # MAESTRO: MIDI co-located with audio, same stem
        for ext in (".midi", ".mid"):
            candidate = audio_path.with_suffix(ext)
            if candidate.exists():
                return candidate

    else:
        # Generic fallback: check same directory
        for ext in (".mid", ".midi"):
            candidate = audio_path.with_suffix(ext)
            if candidate.exists():
                return candidate

    return None
