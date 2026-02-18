"""
Base dataset interface for ML workflows.
Simplified for local development - focuses on audio files and feature paths.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional


class BaseDataset(ABC):
    """
    Abstract base class for music datasets (ML-focused).

    Hybrid approach: Datasets are self-contained but use config.data_root as default.
    This allows both:
    - Easy default usage: dataset = SMDDataset() uses config.data_root
    - Flexible override: dataset = SMDDataset(data_root="/custom/path")
    """

    def __init__(self, data_root: Optional[Path] = None):
        """
        Initialize dataset with optional data root.

        Args:
            data_root: Root data directory. If None, uses config.data_root.
        """
        if data_root is None:
            from ..config import mess_config
            data_root = mess_config.data_root

        self.data_root = Path(data_root)

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """Stable dataset identifier used in configs and artifact naming."""
        ...

    @property
    @abstractmethod
    def audio_dir(self) -> Path:
        """Directory containing audio files for this dataset."""
        ...

    @property
    @abstractmethod
    def embeddings_dir(self) -> Path:
        """Directory for storing MERT embeddings for this dataset."""
        ...

    @property
    def aggregated_dir(self) -> Path:
        """Directory for aggregated [13, 768] embeddings."""
        return self.embeddings_dir / "aggregated"

    def get_audio_files(self) -> List[Path]:
        """
        Get list of audio file paths for the dataset.

        Returns:
            List of Path objects to .wav files
        """
        if not self.audio_dir.exists():
            return []
        return sorted(self.audio_dir.glob("*.wav"))

    def get_feature_path(self, track_id: str, feature_type: str = "aggregated") -> Path:
        """
        Get path where features should be saved/loaded for a track.

        Args:
            track_id: Track identifier (usually filename stem)
            feature_type: Type of features (raw, segments, aggregated)

        Returns:
            Path to .npy feature file
        """
        if feature_type == "aggregated":
            return self.aggregated_dir / f"{track_id}.npy"
        else:
            return self.embeddings_dir / feature_type / f"{track_id}.npy"

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
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Dataset description."""
        ...
