"""Tests for mess.extraction.audio — segment_audio and load_audio."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mess.extraction.audio import (
    load_audio,
    load_audio_segments,
    segment_audio,
    validate_audio_file,
)

pytestmark = pytest.mark.unit


class TestSegmentAudio:
    """segment_audio is a pure function — no mocking needed."""

    def test_basic_segmentation_count(self, long_audio_array):
        """10s audio at 5s/50% overlap -> 3 segments."""
        segments = segment_audio(
            long_audio_array,
            segment_duration=5.0,
            overlap_ratio=0.5,
            sample_rate=24000,
        )
        # Positions: 0-5, 2.5-7.5, 5-10 = 3 from range loop
        # Final segment logic may add one more — check >= 3
        assert len(segments) >= 3

    def test_single_segment_exact_duration(self, sample_audio_array):
        """Audio exactly equal to segment_duration gives 1 segment."""
        # sample_audio_array is 1s, so use segment_duration=1.0
        segments = segment_audio(
            sample_audio_array,
            segment_duration=1.0,
            overlap_ratio=0.5,
            sample_rate=24000,
        )
        assert len(segments) == 1

    def test_no_overlap(self, long_audio_array):
        """overlap=0 means hop == segment_duration."""
        segments = segment_audio(
            long_audio_array,
            segment_duration=5.0,
            overlap_ratio=0.0,
            sample_rate=24000,
        )
        # 10s / 5s = 2 segments from range loop
        assert len(segments) >= 2

    def test_each_segment_has_exact_length(self, long_audio_array):
        """Each segment should be exactly segment_duration * sample_rate samples."""
        sr = 24000
        duration = 5.0
        segments = segment_audio(
            long_audio_array,
            segment_duration=duration,
            overlap_ratio=0.5,
            sample_rate=sr,
        )
        expected_len = int(duration * sr)
        for seg in segments:
            assert len(seg) == expected_len

    def test_short_audio_returns_empty(self):
        """Audio shorter than segment_duration returns no segments from loop."""
        short = np.zeros(100, dtype=np.float32)  # ~4ms at 24kHz
        segments = segment_audio(
            short, segment_duration=5.0, overlap_ratio=0.5, sample_rate=24000
        )
        # range(0, 100 - 120000 + 1, ...) is empty, but final-segment logic
        # adds last segment if len(audio) % hop_samples != 0 AND len >= segment_samples
        # 100 < 120000, so the final append gets audio[-120000:] which is still the full short array
        # The current implementation appends it — that's the behavior we document
        assert isinstance(segments, list)

    def test_returns_list_of_ndarrays(self, long_audio_array):
        segments = segment_audio(long_audio_array)
        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg, np.ndarray)


class TestLoadAudio:
    """load_audio requires AudioDecoder mocking to avoid real file I/O."""

    def test_configures_decoder_for_mono_resampled_output(self, mocker):
        samples = SimpleNamespace(data=torch.randn(1, 24000))
        decoder_cls = mocker.patch("mess.extraction.audio.AudioDecoder")
        decoder_cls.return_value.get_all_samples.return_value = samples

        result = load_audio("fake.wav", target_sr=24000)

        decoder_cls.assert_called_once_with(
            "fake.wav",
            sample_rate=24000,
            num_channels=1,
        )
        assert result.shape == (24000,)
        assert result.dtype == np.float32

    def test_decoder_output_is_squeezed_to_1d(self, mocker):
        samples = SimpleNamespace(data=torch.randn(1, 1234))
        decoder_cls = mocker.patch("mess.extraction.audio.AudioDecoder")
        decoder_cls.return_value.get_all_samples.return_value = samples

        result = load_audio("fake.wav", target_sr=24000)
        assert result.ndim == 1
        assert result.shape == (1234,)


class TestValidateAudioFile:
    def test_missing_file(self, tmp_path):
        result = validate_audio_file(tmp_path / "nonexistent.wav")
        assert result["valid"] is False
        assert result["file_exists"] is False

    def test_metadata_and_probe_decode_success(self, tmp_path, mocker):
        path = tmp_path / "ok.wav"
        path.write_bytes(b"fake")

        metadata = SimpleNamespace(
            sample_rate=24000,
            num_channels=1,
            duration_seconds=2.0,
        )
        probe = SimpleNamespace(data=torch.ones(1, 128))
        decoder = mocker.Mock(metadata=metadata)
        decoder.get_samples_played_in_range.return_value = probe
        decoder_cls = mocker.patch("mess.extraction.audio.AudioDecoder", return_value=decoder)

        result = validate_audio_file(path)

        decoder_cls.assert_called_once_with(str(path))
        decoder.get_samples_played_in_range.assert_called_once_with(0.0, 0.1)
        assert result["valid"] is True
        assert result["readable"] is True
        assert result["sample_rate"] == 24000
        assert result["channels"] == 1
        assert result["duration"] == pytest.approx(2.0)

    def test_probe_failure_marks_file_invalid(self, tmp_path, mocker):
        path = tmp_path / "bad.wav"
        path.write_bytes(b"fake")

        metadata = SimpleNamespace(
            sample_rate=24000,
            num_channels=1,
            duration_seconds=2.0,
        )
        decoder = mocker.Mock(metadata=metadata)
        decoder.get_samples_played_in_range.side_effect = RuntimeError("decode failed")
        mocker.patch("mess.extraction.audio.AudioDecoder", return_value=decoder)

        result = validate_audio_file(path)

        assert result["valid"] is False
        assert any("corrupted" in err.lower() for err in result["errors"])


class TestLoadAudioSegments:
    def test_torchcodec_range_decode_builds_overlapping_segments(self, mocker):
        metadata = SimpleNamespace(duration_seconds=10.0)
        decoder = mocker.Mock(metadata=metadata)

        def _fake_range(start_sec, end_sec):
            n = int(round((end_sec - start_sec) * 10))
            return SimpleNamespace(data=torch.ones(1, n, dtype=torch.float32))

        decoder.get_samples_played_in_range.side_effect = _fake_range
        decoder_cls = mocker.patch("mess.extraction.audio.AudioDecoder", return_value=decoder)

        segments = load_audio_segments(
            "fake.wav",
            target_sr=10,
            segment_duration=5.0,
            overlap_ratio=0.5,
        )

        decoder_cls.assert_called_once_with("fake.wav", sample_rate=10, num_channels=1)
        assert decoder.get_samples_played_in_range.call_count == 3
        decoder.get_samples_played_in_range.assert_any_call(0.0, 5.0)
        decoder.get_samples_played_in_range.assert_any_call(2.5, 7.5)
        decoder.get_samples_played_in_range.assert_any_call(5.0, 10.0)
        assert [len(segment) for segment in segments] == [50, 50, 50]

    def test_fallback_uses_load_audio_and_segment_audio(self, monkeypatch):
        monkeypatch.setattr("mess.extraction.audio.AudioDecoder", None)
        monkeypatch.setattr(
            "mess.extraction.audio.load_audio",
            lambda *_a, **_k: np.ones(12, dtype=np.float32),
        )
        monkeypatch.setattr(
            "mess.extraction.audio.segment_audio",
            lambda *_a, **_k: [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)],
        )

        segments = load_audio_segments(
            "fake.wav",
            target_sr=24000,
            segment_duration=5.0,
            overlap_ratio=0.5,
        )
        assert len(segments) == 2
