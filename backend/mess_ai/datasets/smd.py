"""
Saarland Music Dataset (SMD) implementation
"""
import json
from pathlib import Path
from typing import Dict, List
from .base import BaseDataset
from ..models.metadata import TrackMetadata


class SMDDataset(BaseDataset):
    """Saarland Music Dataset implementation."""
    
    def get_wav_dir(self) -> Path:
        """Get SMD audio directory."""
        return self.data_dir / "smd" / "wav-44"
    
    def get_features_dir(self) -> Path:
        """Get SMD features directory."""
        return self.data_dir / "processed" / "features"
    
    def get_metadata_dir(self) -> Path:
        """Get SMD metadata directory."""
        return self.data_dir / "metadata"
    
    def load_metadata(self) -> Dict[str, TrackMetadata]:
        """Load SMD track metadata."""
        from ..metadata.processor import MetadataProcessor
        processor = MetadataProcessor(
            data_dir=str(self.data_dir),
            metadata_dir=str(self.get_metadata_dir())
        )
        
        try:
            return processor.load_metadata_dict()
        except FileNotFoundError:
            # Generate metadata if not exists
            print("Generating SMD metadata...")
            wav_dir = self.get_wav_dir()
            if wav_dir.exists():
                metadata_list = processor.process_smd_directory(wav_dir)
                processor.save_to_csv(metadata_list)
                return processor.load_metadata_dict()
            else:
                print(f"Warning: SMD wav directory not found at {wav_dir}")
                return {}
    
    def get_track_ids(self) -> List[str]:
        """Get SMD track IDs from audio files."""
        wav_dir = self.get_wav_dir()
        if not wav_dir.exists():
            return []
        
        track_ids = []
        for wav_file in wav_dir.glob("*.wav"):
            track_ids.append(wav_file.stem)
        
        return sorted(track_ids)
    
    @property
    def name(self) -> str:
        return "Saarland Music Dataset (SMD)"
    
    @property
    def description(self) -> str:
        return "Classical music dataset with 50 recordings for similarity search"