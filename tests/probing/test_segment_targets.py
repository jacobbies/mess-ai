"""Tests for segment-level target generation and segment probing."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mess.extraction.audio import segment_audio
from mess.probing.discovery import SEGMENT_TARGETS, LayerDiscoverySystem
from mess.probing.segment_targets import (
    SEGMENT_TARGET_NAMES,
    generate_segment_expression_targets,
    generate_segment_targets,
    get_segment_boundaries,
)

pytestmark = pytest.mark.unit


# =========================================================================
# TestGenerateSegmentTargets
# =========================================================================


class TestGenerateSegmentTargets:
    """Test segment-level audio target generation."""

    @pytest.fixture
    def synthetic_audio(self, tmp_path):
        """Create a synthetic 10s .wav file and return its path."""
        import torch
        import torchaudio

        sr = 24000
        duration = 10.0
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, dtype=np.float32)
        # Mix of frequencies to give meaningful spectral features
        audio = (
            0.5 * np.sin(2 * np.pi * 440 * t)
            + 0.3 * np.sin(2 * np.pi * 880 * t)
            + 0.1 * np.random.default_rng(42).standard_normal(n_samples).astype(np.float32)
        )
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        path = tmp_path / "test_audio.wav"
        torchaudio.save(str(path), audio_tensor, sr)
        return path, sr, n_samples

    def test_output_shape_matches_segment_count(self, synthetic_audio):
        path, sr, n_samples = synthetic_audio
        targets = generate_segment_targets(
            path, segment_duration=5.0, overlap_ratio=0.5, sample_rate=sr,
        )

        # Determine expected segment count
        audio_np = np.zeros(n_samples)  # dummy for counting
        expected_segments = len(
            segment_audio(audio_np, segment_duration=5.0, overlap_ratio=0.5, sample_rate=sr)
        )

        # Every target array should have length == num_segments
        for category, fields in targets.items():
            for field_name, arr in fields.items():
                assert isinstance(arr, np.ndarray), f"{category}/{field_name} not ndarray"
                assert arr.ndim == 1, f"{category}/{field_name} not 1D"
                assert len(arr) == expected_segments, (
                    f"{category}/{field_name}: got {len(arr)}, expected {expected_segments}"
                )

    def test_only_segment_viable_targets_present(self, synthetic_audio):
        path, sr, _ = synthetic_audio
        targets = generate_segment_targets(path, sample_rate=sr)

        # Collect all field names
        all_fields = set()
        for fields in targets.values():
            all_fields.update(fields.keys())

        # Should not contain track-level-only targets
        assert 'tempo' not in all_fields
        assert 'phrase_regularity' not in all_fields
        assert 'num_phrases' not in all_fields
        assert 'onset_density' not in all_fields

        # Should contain segment-viable targets
        assert 'spectral_centroid' in all_fields
        assert 'dynamic_range' in all_fields
        assert 'harmonic_complexity' in all_fields

    def test_target_values_are_finite(self, synthetic_audio):
        path, sr, _ = synthetic_audio
        targets = generate_segment_targets(path, sample_rate=sr)

        for category, fields in targets.items():
            for field_name, arr in fields.items():
                assert np.all(np.isfinite(arr)), (
                    f"{category}/{field_name} contains non-finite values"
                )


# =========================================================================
# TestSegmentExpressionTargets
# =========================================================================


class TestSegmentExpressionTargets:
    """Test per-segment MIDI expression target generation."""

    @pytest.fixture
    def mock_midi(self):
        """Create a mock MIDI with notes at known positions."""
        notes = []
        # Segment 0: t=[0, 5) — 20 notes
        for i in range(20):
            note = MagicMock()
            note.start = 0.1 * i  # 0.0, 0.1, ..., 1.9
            note.end = note.start + 0.08
            note.velocity = 60 + i
            notes.append(note)

        # Segment 1: t=[2.5, 7.5) — 15 notes
        for i in range(15):
            note = MagicMock()
            note.start = 3.0 + 0.1 * i
            note.end = note.start + 0.05
            note.velocity = 80 + i
            notes.append(note)

        # Segment 2: t=[5, 10) — only 2 notes (below min_notes)
        for i in range(2):
            note = MagicMock()
            note.start = 8.0 + 0.5 * i
            note.end = note.start + 0.1
            note.velocity = 50
            notes.append(note)

        return notes

    def test_per_segment_slicing(self, tmp_path, mock_midi):
        boundaries = [(0.0, 5.0), (2.5, 7.5), (5.0, 10.0)]

        instrument = MagicMock()
        instrument.is_drum = False
        instrument.notes = mock_midi

        pm = MagicMock()
        pm.instruments = [instrument]

        mock_pretty_midi = MagicMock()
        mock_pretty_midi.PrettyMIDI.return_value = pm

        with patch.dict("sys.modules", {"pretty_midi": mock_pretty_midi}):
            result = generate_segment_expression_targets(
                tmp_path / "test.mid",
                segment_boundaries=boundaries,
                min_notes=5,
            )

        expr = result['expression']
        assert len(expr['velocity_mean']) == 3

        # Segment 0 should have valid values
        assert not np.isnan(expr['velocity_mean'][0])
        # Segment 1 should also have valid values
        assert not np.isnan(expr['velocity_mean'][1])
        # Segment 2 has only 2 notes < min_notes=5, should be NaN
        assert np.isnan(expr['velocity_mean'][2])

    def test_segment_isolation(self, tmp_path, mock_midi):
        """Notes in segment 0 should not affect segment 1."""
        boundaries = [(0.0, 2.0), (3.0, 5.0)]

        instrument = MagicMock()
        instrument.is_drum = False
        instrument.notes = mock_midi

        pm = MagicMock()
        pm.instruments = [instrument]

        mock_pretty_midi = MagicMock()
        mock_pretty_midi.PrettyMIDI.return_value = pm

        with patch.dict("sys.modules", {"pretty_midi": mock_pretty_midi}):
            result = generate_segment_expression_targets(
                tmp_path / "test.mid",
                segment_boundaries=boundaries,
                min_notes=5,
            )

        expr = result['expression']
        # Seg 0: [0, 2) has notes at 0.0..1.9 → 20 notes with vel 60-79
        # Seg 1: [3, 5) has notes at 3.0..4.4 → 15 notes with vel 80-94
        # Velocities should differ between segments
        assert expr['velocity_mean'][0] != expr['velocity_mean'][1]


# =========================================================================
# TestGetSegmentBoundaries
# =========================================================================


class TestGetSegmentBoundaries:
    def test_matches_segment_audio_count(self):
        sr = 24000
        duration = 15.0
        audio = np.zeros(int(sr * duration))
        segments = segment_audio(audio, segment_duration=5.0, overlap_ratio=0.5, sample_rate=sr)
        boundaries = get_segment_boundaries(
            len(audio), segment_duration=5.0, overlap_ratio=0.5, sample_rate=sr,
        )
        assert len(boundaries) == len(segments)

    def test_boundaries_cover_correct_times(self):
        boundaries = get_segment_boundaries(
            audio_length_samples=240000,  # 10s at 24kHz
            segment_duration=5.0,
            overlap_ratio=0.5,
            sample_rate=24000,
        )
        assert boundaries[0] == pytest.approx((0.0, 5.0))
        assert boundaries[1] == pytest.approx((2.5, 7.5))


# =========================================================================
# TestDiscoverSegments
# =========================================================================


class TestDiscoverSegments:
    """Test segment-level probing pipeline."""

    def test_discover_segments_runs_with_segment_data(self, monkeypatch):
        system = object.__new__(LayerDiscoverySystem)
        system.alpha = 1.0
        system.n_folds = 2
        system.features_dir = None  # not used (mocked)
        system.targets_dir = None   # not used (mocked)

        class DummyDataset:
            name = "SMD"

            @staticmethod
            def get_audio_files():
                return ["a.wav", "b.wav"]

        system.dataset = DummyDataset()

        n_seg_a, n_seg_b = 10, 12
        total = n_seg_a + n_seg_b

        def fake_load_segment_features(_audio_files):
            per_layer = {
                layer: np.random.default_rng(layer).standard_normal((total, 768)).astype(np.float32)
                for layer in range(13)
            }
            return per_layer, ["a.wav", "b.wav"], np.array([n_seg_a, n_seg_b])

        def fake_load_segment_targets(_audio_files):
            targets = {
                "spectral_centroid": np.random.default_rng(0).standard_normal(total).astype(np.float32),
                "dynamic_range": np.random.default_rng(1).standard_normal(total).astype(np.float32),
            }
            return targets, ["a.wav", "b.wav"], np.array([n_seg_a, n_seg_b])

        monkeypatch.setattr(system, "load_segment_features", fake_load_segment_features)
        monkeypatch.setattr(system, "load_segment_targets", fake_load_segment_targets)

        results = system.discover_segments(n_samples=10)

        assert results
        assert len(results) == 13  # 13 layers
        assert "spectral_centroid" in results[0]
        assert "dynamic_range" in results[0]
        assert set(results[0]["spectral_centroid"].keys()) >= {"r2_score", "correlation", "rmse"}

    def test_discover_segments_output_format_matches_discover(self, monkeypatch):
        """Output structure should be identical to discover()."""
        system = object.__new__(LayerDiscoverySystem)
        system.alpha = 1.0
        system.n_folds = 2

        class DummyDataset:
            name = "test"

            @staticmethod
            def get_audio_files():
                return ["t.wav"]

        system.dataset = DummyDataset()

        rng = np.random.default_rng(42)

        def fake_seg_features(_):
            return (
                {l: rng.standard_normal((20, 768)).astype(np.float32) for l in range(13)},
                ["t.wav"],
                np.array([20]),
            )

        def fake_seg_targets(_):
            return (
                {"spectral_centroid": rng.standard_normal(20).astype(np.float32)},
                ["t.wav"],
                np.array([20]),
            )

        monkeypatch.setattr(system, "load_segment_features", fake_seg_features)
        monkeypatch.setattr(system, "load_segment_targets", fake_seg_targets)

        results = system.discover_segments(n_samples=10)

        # Should pass best_layers without error
        best = LayerDiscoverySystem.best_layers(results)
        assert "spectral_centroid" in best
        assert "layer" in best["spectral_centroid"]
        assert "r2_score" in best["spectral_centroid"]

    def test_segment_count_mismatch_skips_track(self, monkeypatch):
        system = object.__new__(LayerDiscoverySystem)
        system.alpha = 1.0
        system.n_folds = 2

        class DummyDataset:
            name = "test"

            @staticmethod
            def get_audio_files():
                return ["a.wav", "b.wav"]

        system.dataset = DummyDataset()

        rng = np.random.default_rng(42)

        def fake_seg_features(_):
            # a.wav: 10 segments, b.wav: 15 segments
            return (
                {l: rng.standard_normal((25, 768)).astype(np.float32) for l in range(13)},
                ["a.wav", "b.wav"],
                np.array([10, 15]),
            )

        def fake_seg_targets(_):
            # a.wav: 10 segments (matches), b.wav: 12 segments (MISMATCH)
            return (
                {"spectral_centroid": rng.standard_normal(22).astype(np.float32)},
                ["a.wav", "b.wav"],
                np.array([10, 12]),
            )

        monkeypatch.setattr(system, "load_segment_features", fake_seg_features)
        monkeypatch.setattr(system, "load_segment_targets", fake_seg_targets)

        results = system.discover_segments(n_samples=10)

        # b.wav should be skipped due to mismatch, only a.wav used (10 segments)
        assert results
        assert results[0]["spectral_centroid"]["n_valid"] == 10.0


# =========================================================================
# TestSegmentTargetsConstant
# =========================================================================


class TestSegmentTargetsConstant:
    """Verify SEGMENT_TARGETS is a proper subset of SCALAR_TARGETS."""

    def test_segment_targets_subset_of_scalar_targets(self):
        scalar_names = set(LayerDiscoverySystem.SCALAR_TARGETS.keys())
        assert SEGMENT_TARGETS.issubset(scalar_names), (
            f"SEGMENT_TARGETS has entries not in SCALAR_TARGETS: "
            f"{SEGMENT_TARGETS - scalar_names}"
        )

    def test_segment_target_names_matches_discovery_constant(self):
        """segment_targets.SEGMENT_TARGET_NAMES should match discovery.SEGMENT_TARGETS."""
        assert SEGMENT_TARGET_NAMES == SEGMENT_TARGETS

    def test_excludes_track_level_targets(self):
        assert 'tempo' not in SEGMENT_TARGETS
        assert 'phrase_regularity' not in SEGMENT_TARGETS
        assert 'num_phrases' not in SEGMENT_TARGETS
        assert 'onset_density' not in SEGMENT_TARGETS

    def test_excludes_expression_targets(self):
        """Expression targets are handled separately by segment MIDI slicing."""
        for name in ('rubato', 'velocity_mean', 'velocity_std'):
            assert name not in SEGMENT_TARGETS
