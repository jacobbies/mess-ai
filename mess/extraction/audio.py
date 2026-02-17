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

import torch
import torchaudio
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Any


def load_audio(audio_path: Union[str, Path], target_sr: int = 24000) -> np.ndarray:
    """
    Load audio and preprocess for MERT (mono, resampled).

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 24000 for MERT)

    Returns:
        Audio array (1D, target sample rate)
    """
    try:
        audio, orig_sr = torchaudio.load(str(audio_path))

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Resample if needed
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=target_sr
            )
            audio = resampler(audio)

        # Convert to numpy and squeeze
        audio = audio.squeeze().numpy()

        return audio

    except Exception as e:
        logging.error(f"Error preprocessing audio {audio_path}: {e}")
        raise


def segment_audio(
    audio: np.ndarray,
    segment_duration: float = 5.0,
    overlap_ratio: float = 0.5,
    sample_rate: int = 24000
) -> List[np.ndarray]:
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


def validate_audio_file(
    audio_path: Union[str, Path],
    check_corruption: bool = True,
    min_duration: float = 1.0
) -> Dict[str, Any]:
    """
    Validate audio file before extraction.

    Checks:
      - File exists and readable
      - Valid audio format (torchaudio compatible)
      - Not corrupted (can load audio data)
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
    result = {
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

        # Try to load metadata
        try:
            metadata = torchaudio.info(str(audio_path))
            result['sample_rate'] = metadata.sample_rate
            result['duration'] = metadata.num_frames / metadata.sample_rate
            result['channels'] = metadata.num_channels
            result['readable'] = True

            # Check minimum duration
            if result['duration'] < min_duration:
                errors.append(
                    f"Duration too short: {result['duration']:.2f}s < {min_duration}s"
                )
                result['valid'] = False

        except Exception as e:
            errors.append(f"Failed to read audio metadata: {str(e)}")
            result['valid'] = False
            result['readable'] = False

        # Check for corruption by trying to load audio
        if check_corruption and result['readable']:
            try:
                audio, sr = torchaudio.load(str(audio_path))
                if audio.shape[1] == 0:
                    errors.append("Audio file is empty (zero samples)")
                    result['valid'] = False
            except Exception as e:
                errors.append(f"Audio file corrupted: {str(e)}")
                result['valid'] = False

    result['errors'] = errors
    return result
