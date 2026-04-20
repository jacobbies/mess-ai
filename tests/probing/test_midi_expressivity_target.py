"""Unit tests for the T7 MIDI velocity / IOI / pedal expressivity scalars.

All tests are hermetic: they synthesize tiny ``PrettyMIDI`` objects and
monkeypatch ``load_midi_for_track`` so the suite doesn't depend on the
SMD / MAESTRO corpora.
"""

from __future__ import annotations

import numpy as np
import pretty_midi
import pytest

from mess.probing._schema import TargetType
from mess.probing.targets import _midi_expressivity
from mess.probing.targets._midi_expressivity import (
    DESCRIPTORS,
    _features,
    _make_generator,
)
from mess.probing.targets._registry import get_generator

pytestmark = pytest.mark.unit


_FEATURE_NAMES = ("midi_velocity_std", "midi_ioi_std", "midi_pedal_ratio")


def _make_midi(
    velocities: list[int],
    onsets: list[float],
    durations: list[float] | None = None,
    pedal_events: list[tuple[float, int]] | None = None,
) -> pretty_midi.PrettyMIDI:
    """Build a one-instrument ``PrettyMIDI`` from raw event data.

    ``pedal_events`` is a list of ``(time_s, cc_value)`` tuples for sustain
    (CC64). ``cc_value >= 64`` means pedal down.
    """
    if durations is None:
        durations = [0.1] * len(onsets)
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    piano = pretty_midi.Instrument(program=0)
    for vel, start, dur in zip(velocities, onsets, durations, strict=True):
        piano.notes.append(
            pretty_midi.Note(velocity=vel, pitch=60, start=start, end=start + dur)
        )
    if pedal_events:
        for time_s, value in pedal_events:
            piano.control_changes.append(
                pretty_midi.ControlChange(number=64, value=value, time=time_s)
            )
    midi.instruments.append(piano)
    return midi


class TestRegistration:
    """Each of the three targets registers itself under the expected name."""

    @pytest.mark.parametrize("name", _FEATURE_NAMES)
    def test_descriptor_metadata(self, name):
        desc = DESCRIPTORS[name]
        assert desc.name == name
        assert desc.type is TargetType.MIDI_SCALAR
        assert desc.category == "midi"
        assert desc.curve_spec is None

    @pytest.mark.parametrize("name", _FEATURE_NAMES)
    def test_get_generator_returns_pair(self, name):
        descriptor, fn = get_generator(name)
        assert descriptor is DESCRIPTORS[name]
        assert callable(fn)


class TestSyntheticVelocity:
    """``midi_velocity_std`` matches ``np.std(velocities)``."""

    def test_matches_numpy_std(self, monkeypatch):
        velocities = list(range(40, 131, 10))  # [40, 50, ..., 130]
        onsets = [i * 1.0 for i in range(len(velocities))]
        midi = _make_midi(velocities, onsets)
        monkeypatch.setattr(
            _midi_expressivity, "load_midi_for_track", lambda *_: midi
        )

        descriptor, fn = get_generator("midi_velocity_std")
        out = fn("fake", "smd")
        assert out is not None
        assert out.shape == (1,)
        assert out[0] == pytest.approx(float(np.std(velocities)), abs=1e-6)


class TestSyntheticPedal:
    """``midi_pedal_ratio`` reflects fraction of time CC64 is engaged."""

    def test_half_engaged(self, monkeypatch):
        # Pedal down 5-15s and 20-30s: 20 s of 30 s window -> 0.667.
        pedal_events = [(5.0, 127), (15.0, 0), (20.0, 127), (30.0, 0)]
        velocities = list(range(40, 131, 10))
        onsets = [i * 1.0 for i in range(len(velocities))]
        midi = _make_midi(velocities, onsets, pedal_events=pedal_events)
        monkeypatch.setattr(
            _midi_expressivity, "load_midi_for_track", lambda *_: midi
        )

        descriptor, fn = get_generator("midi_pedal_ratio")
        out = fn("fake", "smd")
        assert out is not None
        assert out.shape == (1,)
        assert out[0] == pytest.approx(20.0 / 30.0, abs=0.1)

    def test_no_pedal_returns_zero(self, monkeypatch):
        # No CC64 events at all — valid: some pianists don't pedal.
        velocities = [60, 70, 80]
        onsets = [0.0, 1.0, 2.0]
        midi = _make_midi(velocities, onsets)
        monkeypatch.setattr(
            _midi_expressivity, "load_midi_for_track", lambda *_: midi
        )

        _, fn = get_generator("midi_pedal_ratio")
        out = fn("fake", "smd")
        assert out is not None
        assert out[0] == pytest.approx(0.0, abs=1e-6)


class TestSyntheticIOI:
    """Variable IOIs yield larger ``midi_ioi_std`` than uniform IOIs."""

    def _run(self, monkeypatch, onsets):
        velocities = [80] * len(onsets)
        midi = _make_midi(velocities, onsets)
        monkeypatch.setattr(
            _midi_expressivity, "load_midi_for_track", lambda *_: midi
        )
        _, fn = get_generator("midi_ioi_std")
        out = fn("fake", "smd")
        assert out is not None
        return float(out[0])

    def test_variable_greater_than_uniform(self, monkeypatch):
        # Uniform: 0.5 s between each onset. Variable: a mix of 0.2 and 1.0 s.
        uniform = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        variable = [0.0, 0.2, 1.2, 1.4, 2.4, 2.6, 3.6, 3.8]

        std_uniform = self._run(monkeypatch, uniform)
        std_variable = self._run(monkeypatch, variable)
        assert std_variable > std_uniform
        # Uniform IOIs -> log-stddev ~ 0 (all equal).
        assert std_uniform == pytest.approx(0.0, abs=1e-3)


class TestNoneHandling:
    """Missing MIDI / sparse notes propagate ``None`` through all three."""

    def test_missing_midi_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            _midi_expressivity, "load_midi_for_track", lambda *_: None
        )
        for name in _FEATURE_NAMES:
            _, fn = get_generator(name)
            assert fn("fake", "smd") is None

    def test_single_note_returns_none(self, monkeypatch):
        midi = _make_midi(velocities=[80], onsets=[0.0])
        monkeypatch.setattr(
            _midi_expressivity, "load_midi_for_track", lambda *_: midi
        )
        for name in _FEATURE_NAMES:
            _, fn = get_generator(name)
            assert fn("fake", "smd") is None


class TestFeaturesHelper:
    """The shared ``_features`` helper produces all three keys in one call."""

    def test_returns_all_keys(self, monkeypatch):
        velocities = [60, 70, 80, 90]
        onsets = [0.0, 0.5, 1.0, 1.5]
        midi = _make_midi(velocities, onsets, pedal_events=[(0.0, 127), (30.0, 0)])
        monkeypatch.setattr(
            _midi_expressivity, "load_midi_for_track", lambda *_: midi
        )

        feats = _features("fake", "smd")
        assert feats is not None
        assert set(feats) == set(_FEATURE_NAMES)
        # Pedal held for entire window.
        assert feats["midi_pedal_ratio"] == pytest.approx(1.0, abs=0.05)

    def test_zero_ioi_after_filter_returns_zero(self, monkeypatch):
        # Two simultaneous onsets at t=0 collapse to a single positive IOI
        # after filtering; the filter leaves one element so stddev is 0.
        velocities = [80, 80]
        onsets = [0.0, 0.0]
        midi = _make_midi(velocities, onsets)
        monkeypatch.setattr(
            _midi_expressivity, "load_midi_for_track", lambda *_: midi
        )
        feats = _features("fake", "smd")
        assert feats is not None
        assert feats["midi_ioi_std"] == pytest.approx(0.0, abs=1e-6)


class TestMakeGeneratorMetadata:
    """Closure names are descriptive (helps mlflow / tracebacks)."""

    @pytest.mark.parametrize("name", _FEATURE_NAMES)
    def test_generator_name_is_descriptive(self, name):
        fn = _make_generator(name)
        assert fn.__name__ == f"generate_{name}"
