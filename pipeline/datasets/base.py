"""
Base dataset interface for music similarity backend
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
from ..models.metadata import TrackMetadata


class BaseDataset(ABC):
    """Abstract base class for music datasets."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    @abstractmethod
    def get_wav_dir(self) -> Path:
        """Get the directory containing audio files."""
        pass
    
    @abstractmethod
    def get_features_dir(self) -> Path:
        """Get the directory containing processed features."""
        pass
    
    @abstractmethod
    def get_metadata_dir(self) -> Path:
        """Get the directory containing metadata files."""
        pass
    
    @abstractmethod
    def load_metadata(self) -> Dict[str, TrackMetadata]:
        """Load track metadata for the dataset."""
        pass
    
    @abstractmethod
    def get_track_ids(self) -> List[str]:
        """Get list of available track IDs."""
        pass
    
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