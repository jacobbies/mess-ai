"""Tests for mess.probing.midi_targets — MIDI expression target generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mess.probing.midi_targets import MidiExpressionTargets, resolve_midi_path

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_note(start: float, end: float, velocity: int = 80) -> MagicMock:
    note = MagicMock()
    note.start = start
    note.end = end
    note.velocity = velocity
    return note


def _make_pm(notes: list, is_drum: bool = False) -> MagicMock:
    """Create a mock PrettyMIDI with one instrument containing *notes*."""
    instrument = MagicMock()
    instrument.is_drum = is_drum
    instrument.notes = notes
    pm = MagicMock()
    pm.instruments = [instrument]
    return pm


# ---------------------------------------------------------------------------
# MidiExpressionTargets
# ---------------------------------------------------------------------------

class TestGenerateExpressionTargets:

    def test_returns_correct_structure(self):
        notes = [_make_note(i * 0.5, i * 0.5 + 0.4, 60 + i) for i in range(20)]
        pm = _make_pm(notes)
        gen = MidiExpressionTargets(min_notes=5)

        with patch("pretty_midi.PrettyMIDI", return_value=pm):
            result = gen.generate_expression_targets("/fake/file.mid")

        assert "expression" in result
        expected_keys = {
            "rubato", "velocity_mean", "velocity_std", "velocity_range",
            "articulation_ratio", "tempo_variability", "onset_timing_std",
        }
        assert set(result["expression"].keys()) == expected_keys

    def test_all_values_are_single_element_arrays(self):
        notes = [_make_note(i * 0.5, i * 0.5 + 0.4, 64) for i in range(20)]
        pm = _make_pm(notes)
        gen = MidiExpressionTargets(min_notes=5)

        with patch("pretty_midi.PrettyMIDI", return_value=pm):
            result = gen.generate_expression_targets("/fake/file.mid")

        for key, val in result["expression"].items():
            assert isinstance(val, np.ndarray), f"{key} is not ndarray"
            assert val.shape == (1,), f"{key} shape is {val.shape}, expected (1,)"

    def test_raises_on_too_few_notes(self):
        notes = [_make_note(0.0, 0.5, 64)]
        pm = _make_pm(notes)
        gen = MidiExpressionTargets(min_notes=10)

        with patch("pretty_midi.PrettyMIDI", return_value=pm):
            with pytest.raises(ValueError, match="need >= 10"):
                gen.generate_expression_targets("/fake/file.mid")

    def test_skips_drum_instruments(self):
        drum_notes = [_make_note(i * 0.5, i * 0.5 + 0.1) for i in range(30)]
        piano_notes = [_make_note(i * 0.5, i * 0.5 + 0.4) for i in range(15)]

        drum_inst = MagicMock()
        drum_inst.is_drum = True
        drum_inst.notes = drum_notes

        piano_inst = MagicMock()
        piano_inst.is_drum = False
        piano_inst.notes = piano_notes

        pm = MagicMock()
        pm.instruments = [drum_inst, piano_inst]

        gen = MidiExpressionTargets(min_notes=5)
        with patch("pretty_midi.PrettyMIDI", return_value=pm):
            result = gen.generate_expression_targets("/fake/file.mid")

        assert "expression" in result

    def test_velocity_stats_correct(self):
        velocities = [40, 60, 80, 100, 120]
        notes = [_make_note(i * 0.5, i * 0.5 + 0.4, v) for i, v in enumerate(velocities)]
        all_notes = notes * 3
        pm = _make_pm(all_notes)

        gen = MidiExpressionTargets(min_notes=5)
        with patch("pretty_midi.PrettyMIDI", return_value=pm):
            result = gen.generate_expression_targets("/fake/file.mid")

        expr = result["expression"]
        all_vels = np.array(velocities * 3, dtype=np.float64)
        np.testing.assert_allclose(expr["velocity_mean"][0], np.mean(all_vels), rtol=1e-5)
        np.testing.assert_allclose(expr["velocity_std"][0], np.std(all_vels), rtol=1e-5)
        np.testing.assert_allclose(expr["velocity_range"][0], 80.0)

    def test_uniform_timing_low_rubato(self):
        """Perfectly uniform IOIs should produce near-zero rubato."""
        notes = [_make_note(i * 0.5, i * 0.5 + 0.4, 64) for i in range(20)]
        pm = _make_pm(notes)

        gen = MidiExpressionTargets(min_notes=5)
        with patch("pretty_midi.PrettyMIDI", return_value=pm):
            result = gen.generate_expression_targets("/fake/file.mid")

        assert result["expression"]["rubato"][0] < 0.01

    def test_varied_timing_higher_rubato(self):
        """Non-uniform IOIs should produce measurable rubato."""
        times = []
        t = 0.0
        for i in range(20):
            times.append(t)
            t += 0.3 if i % 2 == 0 else 0.7
        notes = [_make_note(t, t + 0.2, 64) for t in times]
        pm = _make_pm(notes)

        gen = MidiExpressionTargets(min_notes=5)
        with patch("pretty_midi.PrettyMIDI", return_value=pm):
            result = gen.generate_expression_targets("/fake/file.mid")

        assert result["expression"]["rubato"][0] > 0.1


# ---------------------------------------------------------------------------
# resolve_midi_path
# ---------------------------------------------------------------------------

class TestResolveMidiPath:

    def test_smd_finds_mid_in_sibling_midi_dir(self, tmp_path):
        wav_dir = tmp_path / "audio" / "smd" / "wav-44"
        midi_dir = tmp_path / "audio" / "smd" / "midi"
        wav_dir.mkdir(parents=True)
        midi_dir.mkdir(parents=True)

        wav_file = wav_dir / "Bach_BWV849.wav"
        midi_file = midi_dir / "Bach_BWV849.mid"
        wav_file.touch()
        midi_file.touch()

        result = resolve_midi_path(wav_file, "smd")
        assert result == midi_file

    def test_maestro_finds_midi_collocated(self, tmp_path):
        year_dir = tmp_path / "audio" / "maestro" / "2018"
        year_dir.mkdir(parents=True)

        wav_file = year_dir / "track.wav"
        midi_file = year_dir / "track.midi"
        wav_file.touch()
        midi_file.touch()

        result = resolve_midi_path(wav_file, "maestro")
        assert result == midi_file

    def test_returns_none_when_no_midi(self, tmp_path):
        wav_file = tmp_path / "track.wav"
        wav_file.touch()

        assert resolve_midi_path(wav_file, "smd") is None
        assert resolve_midi_path(wav_file, "maestro") is None

    def test_unknown_dataset_checks_same_directory(self, tmp_path):
        wav_file = tmp_path / "track.wav"
        midi_file = tmp_path / "track.mid"
        wav_file.touch()
        midi_file.touch()

        result = resolve_midi_path(wav_file, "custom")
        assert result == midi_file
