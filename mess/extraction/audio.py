"""
Audio preprocessing utilities for MERT feature extraction.

Standalone functions (no model dependency) for loading, segmenting,
and validating audio files. Can be imported independently for data
exploration without loading the MERT model.

Usage:
    from mess.extraction.audio import load_audio, segment_audio, validate_audio_file

    audio = load_audio("track.wav", target_sr=24000)
    segments = segment_audio(audio, segment_duration=5.0, overlap_ratio=0.5, sample_rate=24000)
    validation = validate_audio_file("track.wav")
"""

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

try:
    from torchcodec.decoders import AudioDecoder  # type: ignore[import-untyped]
except Exception:
    AudioDecoder = None  # type: ignore[assignment,misc]


def load_audio(audio_path: str | Path, target_sr: int = 24000) -> np.ndarray:
    """
    Load audio and preprocess for MERT (mono, resampled).

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 24000 for MERT)

    Returns:
        Audio array (1D, target sample rate)
    """
    try:
        if AudioDecoder is not None:
            decoder = AudioDecoder(
                str(audio_path),
                sample_rate=target_sr,
                num_channels=1,
            )
            samples = decoder.get_all_samples()
            waveform = samples.data
            return np.asarray(waveform.squeeze(0).cpu().numpy(), dtype=np.float32)

        audio, orig_sr = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=True,
        )
        mono = audio.mean(axis=1)
        if orig_sr != target_sr:
            gcd = math.gcd(orig_sr, target_sr)
            up = target_sr // gcd
            down = orig_sr // gcd
            mono = signal.resample_poly(mono, up=up, down=down)
        return np.asarray(mono, dtype=np.float32)

    except Exception as e:
        logging.error(f"Error preprocessing audio {audio_path}: {e}")
        raise


def segment_audio(
    audio: np.ndarray,
    segment_duration: float = 5.0,
    overlap_ratio: float = 0.5,
    sample_rate: int = 24000
) -> list[np.ndarray]:
    """
    Segment audio into overlapping windows.

    Args:
        audio: Audio array (1D)
        segment_duration: Duration of each segment in seconds (default: 5.0)
        overlap_ratio: Overlap between segments (default: 0.5 = 50%)
        sample_rate: Audio sample rate (default: 24000)

    Returns:
        List of audio segments
    """
    segment_samples = int(segment_duration * sample_rate)
    hop_samples = int(segment_samples * (1 - overlap_ratio))

    segments = []
    for start in range(0, len(audio) - segment_samples + 1, hop_samples):
        end = start + segment_samples
        segments.append(audio[start:end])

    # Add final segment if remaining audio
    if len(audio) % hop_samples != 0:
        segments.append(audio[-segment_samples:])

    return segments


def _segment_bounds(
    total_samples: int,
    segment_samples: int,
    hop_samples: int,
) -> list[tuple[int, int]]:
    """Compute sample bounds matching segment_audio overlap behavior."""
    bounds: list[tuple[int, int]] = []
    for start in range(0, total_samples - segment_samples + 1, hop_samples):
        bounds.append((start, start + segment_samples))

    if total_samples % hop_samples != 0:
        if total_samples >= segment_samples:
            start = total_samples - segment_samples
            end = total_samples
        else:
            start = 0
            end = total_samples
        if not bounds or bounds[-1] != (start, end):
            bounds.append((start, end))

    return bounds


def _match_expected_length(samples: np.ndarray, expected_len: int) -> np.ndarray:
    """Pad/trim decoded samples so segment lengths are deterministic."""
    if expected_len <= 0:
        return samples
    current_len = len(samples)
    if current_len == expected_len:
        return samples
    if current_len > expected_len:
        return samples[:expected_len]
    return np.pad(samples, (0, expected_len - current_len), mode="constant")


def load_audio_segments(
    audio_path: str | Path,
    target_sr: int = 24000,
    segment_duration: float = 5.0,
    overlap_ratio: float = 0.5,
) -> list[np.ndarray]:
    """
    Decode overlapping audio segments.

    Uses TorchCodec range decoding when available to avoid full-track arrays.
    Falls back to full decode + in-memory segmentation otherwise.
    """
    segment_samples = int(segment_duration * target_sr)
    hop_samples = int(segment_samples * (1 - overlap_ratio))
    if segment_samples <= 0 or hop_samples <= 0:
        raise ValueError("segment_duration and overlap_ratio must produce positive sample counts")

    if AudioDecoder is None:
        full_audio = load_audio(audio_path, target_sr=target_sr)
        return segment_audio(
            full_audio,
            segment_duration=segment_duration,
            overlap_ratio=overlap_ratio,
            sample_rate=target_sr,
        )

    decoder = AudioDecoder(
        str(audio_path),
        sample_rate=target_sr,
        num_channels=1,
    )
    metadata = decoder.metadata
    duration = getattr(metadata, "duration_seconds", None)
    if duration is None:
        duration = getattr(metadata, "duration_seconds_from_header", None)
    if duration is None or duration <= 0:
        return []

    total_samples = int(round(float(duration) * target_sr))
    if total_samples <= 0:
        return []

    bounds = _segment_bounds(total_samples, segment_samples, hop_samples)
    segments: list[np.ndarray] = []

    for start_sample, end_sample in bounds:
        start_sec = start_sample / target_sr
        end_sec = end_sample / target_sr
        decoded = decoder.get_samples_played_in_range(start_sec, end_sec).data
        segment = np.asarray(decoded.squeeze(0).cpu().numpy(), dtype=np.float32)
        expected_len = end_sample - start_sample
        segment = _match_expected_length(segment, expected_len)
        segments.append(segment)

    return segments


def validate_audio_file(
    audio_path: str | Path,
    check_corruption: bool = True,
    min_duration: float = 1.0
) -> dict[str, Any]:
    """
    Validate audio file before extraction.

    Checks:
      - File exists and readable
      - Valid audio format (torchcodec compatible)
      - Not corrupted (can decode a small sample window)
      - Sufficient duration (>= min_duration)

    Args:
        audio_path: Path to audio file
        check_corruption: Attempt to load audio to detect corruption
        min_duration: Minimum required duration in seconds (default: 1.0)

    Returns:
        Dict with validation results:
          - valid: bool
          - file_exists: bool
          - readable: bool
          - sample_rate: Optional[int]
          - duration: Optional[float]
          - channels: Optional[int]
          - errors: List[str]
    """
    audio_path = Path(audio_path)
    errors = []
    result: dict[str, Any] = {
        'valid': True,
        'file_exists': False,
        'readable': False,
        'sample_rate': None,
        'duration': None,
        'channels': None,
        'errors': []
    }

    # Check file exists
    if not audio_path.exists():
        errors.append(f"File does not exist: {audio_path}")
        result['valid'] = False
    else:
        result['file_exists'] = True

        if AudioDecoder is not None:
            decoder: Any | None = None
            # Try to load metadata
            try:
                decoder = AudioDecoder(str(audio_path))
                metadata = decoder.metadata
                result['sample_rate'] = metadata.sample_rate
                duration = getattr(metadata, "duration_seconds", None)
                if duration is None:
                    duration = getattr(metadata, "duration_seconds_from_header", None)
                result['duration'] = float(duration) if duration is not None else None
                result['channels'] = metadata.num_channels
                result['readable'] = True

                # Check minimum duration
                if result['duration'] is None:
                    errors.append("Audio metadata missing duration")
                    result['valid'] = False
                elif result['duration'] < min_duration:
                    errors.append(
                        f"Duration too short: {result['duration']:.2f}s < {min_duration}s"
                    )
                    result['valid'] = False

            except Exception as e:
                errors.append(f"Failed to read audio metadata: {str(e)}")
                result['valid'] = False
                result['readable'] = False

            # Check for corruption by decoding a small range
            if check_corruption and result['readable']:
                try:
                    if decoder is None or result['duration'] is None:
                        raise RuntimeError("Cannot probe corruption without metadata")
                    probe_stop = min(float(result['duration']), 0.1)
                    if probe_stop <= 0:
                        errors.append("Audio file is empty (zero samples)")
                        result['valid'] = False
                    else:
                        probe = decoder.get_samples_played_in_range(0.0, probe_stop)
                        if probe.data.shape[1] == 0:
                            errors.append("Audio file is empty (zero samples)")
                            result['valid'] = False
                except Exception as e:
                    errors.append(f"Audio file corrupted: {str(e)}")
                    result['valid'] = False
        else:
            try:
                metadata = sf.info(str(audio_path))
                result['sample_rate'] = int(metadata.samplerate)
                result['channels'] = int(metadata.channels)
                result['duration'] = float(metadata.duration)
                result['readable'] = True

                if result['duration'] < min_duration:
                    errors.append(
                        f"Duration too short: {result['duration']:.2f}s < {min_duration}s"
                    )
                    result['valid'] = False
            except Exception as e:
                errors.append(f"Failed to read audio metadata: {str(e)}")
                result['valid'] = False
                result['readable'] = False

            if check_corruption and result['readable']:
                try:
                    num_probe_frames = max(1, int(float(result['sample_rate']) * 0.1))
                    probe, _ = sf.read(
                        str(audio_path),
                        dtype="float32",
                        always_2d=True,
                        start=0,
                        frames=num_probe_frames,
                    )
                    if probe.shape[0] == 0:
                        errors.append("Audio file is empty (zero samples)")
                        result['valid'] = False
                except Exception as e:
                    errors.append(f"Audio file corrupted: {str(e)}")
                    result['valid'] = False

    result['errors'] = errors
    return result
