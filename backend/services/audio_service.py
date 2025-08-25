"""
Audio Service

Simple audio file serving for the API.
"""
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AudioService:
    """Simple audio file serving service."""
    
    def __init__(self, audio_dir: Path):
        """Initialize audio service.
        
        Args:
            audio_dir: Directory containing audio files
        """
        self.audio_dir = Path(audio_dir)
    
    def get_audio_path(self, track_id: str) -> Optional[Path]:
        """Get the path to an audio file.
        
        Args:
            track_id: Track identifier
            
        Returns:
            Path to audio file if exists
        """
        # Try with .wav extension
        audio_path = self.audio_dir / f"{track_id}.wav"
        if audio_path.exists():
            return audio_path
        
        # Try with .mp3 extension
        audio_path = self.audio_dir / f"{track_id}.mp3"
        if audio_path.exists():
            return audio_path
        
        logger.warning(f"Audio file not found for track: {track_id}")
        return None
    
    def validate_track_exists(self, track_id: str) -> bool:
        """Check if audio file exists for track.
        
        Args:
            track_id: Track identifier
            
        Returns:
            True if audio file exists
        """
        return self.get_audio_path(track_id) is not None