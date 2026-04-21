"""Unit tests for T6 — MIDI articulation histogram target generator."""

from __future__ import annotations

import numpy as np
import pretty_midi
import pytest

from mess.probing.targets import _midi_articulation
from mess.probing.targets._midi_articulation import (
    ARTICULATION_SPEC,
    BIN_EDGES,
    DESCRIPTOR,
    N_BINS,
    generate,
)
from mess.probing.targets._registry import get_generator

pytestmark = pytest.mark.unit


def _synth_midi(
    *,
    n_notes: int = 20,
    ioi: float = 0.5,
    duration_ratio: float = 0.5,
    start: float = 0.0,
    pitch: int = 60,
    velocity: int = 80,
) -> pretty_midi.PrettyMIDI:
    """Build a single-instrument MIDI with notes of fixed IOI and duration ratio.

    ``duration_ratio`` is ``note_duration / ioi``. ``0.1`` is staccato, ``1.0``
    is legato. The final note shares ``duration_ratio * ioi`` so it doesn't
    perturb tests that peek at the last note.
    """
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    note_dur = duration_ratio * ioi
    for i in range(n_notes):
        onset = start + i * ioi
        inst.notes.append(
            pretty_midi.Note(
                velocity=velocity, pitch=pitch, start=onset, end=onset + note_dur
            )
        )
    midi.instruments.append(inst)
    return midi


class TestRegistration:
    def test_descriptor_registered(self):
        desc, fn = get_generator("midi_articulation_hist")
        assert desc is DESCRIPTOR
        assert fn is generate

    def test_descriptor_shape_contract(self):
        assert DESCRIPTOR.name == "midi_articulation_hist"
        assert DESCRIPTOR.category == "midi"
        assert DESCRIPTOR.curve_spec is ARTICULATION_SPEC
        assert ARTICULATION_SPEC.n_frames == 16
        assert N_BINS == 16
        assert BIN_EDGES.shape == (17,)
        assert BIN_EDGES[0] == pytest.approx(0.0)
        assert BIN_EDGES[-1] == pytest.approx(1.5)


class TestShape:
    def test_generate_returns_16_dim_probability(self, monkeypatch):
        midi = _synth_midi(n_notes=20, ioi=0.5, duration_ratio=0.5)
        monkeypatch.setattr(
            _midi_articulation, "load_midi_for_track", lambda t, d: midi
        )

        hist = generate("fake_track", "smd", audio_duration_s=30.0)

        assert hist is not None
        assert hist.shape == (16,)
        assert hist.dtype == np.float32
        assert hist.sum() == pytest.approx(1.0, abs=1e-5)
        assert np.all(hist >= 0.0)


class TestStaccato:
    def test_staccato_mass_in_low_bins(self, monkeypatch):
        # duration_ratio = 0.1 -> articulation ≈ 0.1 -> bin index
        # floor(0.1 / (1.5/16)) = floor(1.066...) = 1
        midi = _synth_midi(n_notes=30, ioi=0.4, duration_ratio=0.1)
        monkeypatch.setattr(
            _midi_articulation, "load_midi_for_track", lambda t, d: midi
        )

        hist = generate("fake_track", "smd", audio_duration_s=30.0)

        assert hist is not None
        # Almost all mass in the first three bins (0..~0.28).
        low_mass = hist[:3].sum()
        assert low_mass > 0.9, f"expected staccato mass in low bins; got {hist!r}"
        # Peak should be in bin 0 or 1.
        assert hist.argmax() in (0, 1)


class TestLegato:
    def test_legato_mass_near_bin_10(self, monkeypatch):
        # duration_ratio = 1.0 -> articulation ≈ 1.0 -> bin index
        # floor(1.0 / (1.5/16)) = floor(10.666...) = 10
        midi = _synth_midi(n_notes=30, ioi=0.4, duration_ratio=1.0)
        monkeypatch.setattr(
            _midi_articulation, "load_midi_for_track", lambda t, d: midi
        )

        hist = generate("fake_track", "smd", audio_duration_s=30.0)

        assert hist is not None
        # Mass concentrates in bin 10 (articulation ≈ 1.0).
        assert hist.argmax() == 10, f"expected legato peak at bin 10; got {hist!r}"
        assert hist[10] > 0.9


class TestNonePropagation:
    def test_returns_none_when_midi_missing(self, monkeypatch):
        monkeypatch.setattr(
            _midi_articulation, "load_midi_for_track", lambda t, d: None
        )
        assert generate("missing_track", "smd", audio_duration_s=30.0) is None


class TestInsufficientNotes:
    def test_single_note_returns_uniform(self, monkeypatch):
        midi = _synth_midi(n_notes=1, ioi=0.5, duration_ratio=0.5)
        monkeypatch.setattr(
            _midi_articulation, "load_midi_for_track", lambda t, d: midi
        )

        hist = generate("fake_track", "smd", audio_duration_s=30.0)

        assert hist is not None
        assert hist.shape == (16,)
        assert hist.sum() == pytest.approx(1.0, abs=1e-5)
        # Uniform: every bin has the same probability.
        assert np.allclose(hist, 1.0 / 16.0)

    def test_empty_midi_returns_uniform(self, monkeypatch):
        midi = pretty_midi.PrettyMIDI()
        midi.instruments.append(pretty_midi.Instrument(program=0))
        monkeypatch.setattr(
            _midi_articulation, "load_midi_for_track", lambda t, d: midi
        )

        hist = generate("fake_track", "smd", audio_duration_s=30.0)

        assert hist is not None
        assert np.allclose(hist, 1.0 / 16.0)
