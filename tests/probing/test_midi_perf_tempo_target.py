"""Unit tests for T8 — MIDI performance-tempo curve target generator."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pretty_midi
import pytest

from mess.probing._schema import DEFAULT_CURVE_SPEC
from mess.probing.targets import _midi_perf_tempo
from mess.probing.targets._midi_perf_tempo import DESCRIPTOR, generate
from mess.probing.targets._registry import get_generator

pytestmark = pytest.mark.unit


def _midi_from_onsets(onsets: Iterable[float]) -> pretty_midi.PrettyMIDI:
    """Build a single-instrument MIDI with a C4 note at each onset."""
    midi = pretty_midi.PrettyMIDI(resolution=220)
    inst = pretty_midi.Instrument(program=0)
    for onset in onsets:
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=onset, end=onset + 0.2)
        )
    midi.instruments.append(inst)
    return midi


class TestRegistration:
    def test_descriptor_registered(self):
        desc, fn = get_generator("midi_perf_tempo")
        assert desc is DESCRIPTOR
        assert fn is generate

    def test_descriptor_shape_contract(self):
        assert DESCRIPTOR.name == "midi_perf_tempo"
        assert DESCRIPTOR.category == "midi"
        assert DESCRIPTOR.curve_spec is DEFAULT_CURVE_SPEC


class TestConstantTempo:
    def test_120_bpm_click_track(self, monkeypatch):
        # 120 BPM -> 0.5 s IOI. Cover the full 30 s window.
        onsets = np.arange(0.0, 30.0, 0.5)
        midi = _midi_from_onsets(onsets)
        monkeypatch.setattr(
            _midi_perf_tempo, "load_midi_for_track", lambda t, d: midi
        )

        curve = generate("fake_track", "smd")

        assert curve is not None
        assert curve.shape == (60,)
        assert curve.dtype == np.float32
        # log2(120/60) = 1.0. Allow tiny slack for window-edge effects.
        assert curve.mean() == pytest.approx(1.0, abs=0.02)
        assert curve.std() < 0.05


class TestTempoStep:
    def test_step_slow_to_fast(self, monkeypatch):
        # 90 BPM (0.667 s IOI) for first 15 s, 180 BPM (0.333 s IOI) for last
        # 15 s. Chose IOIs << _WINDOW_S/2 so every window reliably contains
        # >=3 onsets (60 BPM with 1 s IOI was borderline).
        slow = list(np.arange(0.0, 15.0, 60.0 / 90.0))
        fast = list(np.arange(15.0, 30.0, 60.0 / 180.0))
        onsets = slow + fast
        midi = _midi_from_onsets(onsets)
        monkeypatch.setattr(
            _midi_perf_tempo, "load_midi_for_track", lambda t, d: midi
        )

        curve = generate("fake_track", "smd")

        expected_slow = float(np.log2(90.0 / 60.0))  # ≈ 0.585
        expected_fast = float(np.log2(180.0 / 60.0))  # ≈ 1.585

        assert curve is not None
        assert curve.shape == (60,)
        # Early frames (well inside the slow section) sit near log2(90/60).
        assert curve[5] == pytest.approx(expected_slow, abs=0.1)
        # Late frames sit near log2(180/60).
        assert curve[-3] == pytest.approx(expected_fast, abs=0.1)
        # Second half is clearly above 0.7 (i.e. well into "fast" territory).
        assert (curve[len(curve) // 2 + 5 :] > 0.7).all()
        # And the curve moves in the expected direction across the step.
        assert curve[-3] > curve[5] + 0.5


class TestNonePropagation:
    def test_returns_none_when_midi_missing(self, monkeypatch):
        monkeypatch.setattr(
            _midi_perf_tempo, "load_midi_for_track", lambda t, d: None
        )
        assert generate("missing_track", "smd") is None


class TestInsufficientOnsets:
    def test_fewer_than_four_onsets_returns_none(self, monkeypatch):
        # Only 3 onsets — below the min-4 floor the generator requires
        # before it will attempt any local-BPM estimation.
        midi = _midi_from_onsets([0.5, 1.0, 1.5])
        monkeypatch.setattr(
            _midi_perf_tempo, "load_midi_for_track", lambda t, d: midi
        )
        assert generate("fake_track", "smd") is None
