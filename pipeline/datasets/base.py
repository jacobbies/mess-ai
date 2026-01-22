"""
Base dataset interface for ML workflows.
Simplified for local development - focuses on audio files and feature paths.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class BaseDataset(ABC):
    """Abstract base class for music datasets (ML-focused)."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    @abstractmethod
    def get_audio_files(self) -> List[Path]:
        """
        Get list of audio file paths for the dataset.

        Returns:
            List of Path objects to .wav files
        """
        pass

    def get_feature_path(self, track_id: str, feature_type: str = "aggregated") -> Path:
        """
        Get path where features should be saved/loaded for a track.

        Args:
            track_id: Track identifier (usually filename stem)
            feature_type: Type of features (raw, segments, aggregated)

        Returns:
            Path to .npy feature file
        """
        features_dir = self.data_dir / "processed" / "features" / feature_type
        return features_dir / f"{track_id}.npy"

    def exists(self) -> bool:
        """Check if dataset directory exists."""
        audio_files = self.get_audio_files()
        return len(audio_files) > 0

    def __len__(self) -> int:
        """Number of audio files in dataset."""
        return len(self.get_audio_files())

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Dataset description."""
        pass