"""
Audio file serving service.
Handles all audio and media file operations.
"""
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AudioService:
    """Service for audio file management and serving."""
    
    def __init__(self, library, wav_dir: Path, waveforms_dir: Path):
        """Initialize with audio library and file paths."""
        self.library = library
        self.wav_dir = wav_dir
        self.waveforms_dir = waveforms_dir
        logger.info(f"AudioService initialized with wav_dir: {wav_dir}")
    
    def get_audio_file_path(self, filename: str) -> Optional[Path]:
        """
        Get the full path to an audio file.
        
        Args:
            filename: Name of the audio file
            
        Returns:
            Path to the file if it exists, None otherwise
        """
        file_path = self.wav_dir / filename
        return file_path if file_path.exists() else None
    
    def get_waveform_file_path(self, filename: str) -> Optional[Path]:
        """
        Get the full path to a waveform image file.
        
        Args:
            filename: Name of the audio file (with or without .wav extension)
            
        Returns:
            Path to the waveform PNG if it exists, None otherwise
        """
        # Remove .wav extension if present
        track_id = filename.replace('.wav', '')
        waveform_path = self.waveforms_dir / f"{track_id}.png"
        return waveform_path if waveform_path.exists() else None
    
    def audio_file_exists(self, filename: str) -> bool:
        """Check if an audio file exists."""
        return self.get_audio_file_path(filename) is not None
    
    def waveform_file_exists(self, filename: str) -> bool:
        """Check if a waveform file exists."""
        return self.get_waveform_file_path(filename) is not None
    
    def list_audio_files(self) -> list:
        """Get list of all available audio files."""
        if self.library:
            return list(self.library.list_files())
        return []
    
    def get_library_stats(self) -> dict:
        """Get statistics about the audio library."""
        audio_files = self.list_audio_files()
        
        # Count waveform files
        waveform_count = 0
        if self.waveforms_dir.exists():
            waveform_count = len(list(self.waveforms_dir.glob("*.png")))
        
        return {
            "total_audio_files": len(audio_files),
            "total_waveforms": waveform_count,
            "library_loaded": self.library is not None,
            "wav_directory": str(self.wav_dir),
            "waveforms_directory": str(self.waveforms_dir)
        }
    
    def validate_file_access(self, filename: str) -> dict:
        """
        Validate that a file can be accessed.
        
        Args:
            filename: Name of the file to validate
            
        Returns:
            Dictionary with validation results
        """
        audio_path = self.get_audio_file_path(filename)
        waveform_path = self.get_waveform_file_path(filename)
        
        return {
            "filename": filename,
            "audio_exists": audio_path is not None,
            "audio_path": str(audio_path) if audio_path else None,
            "waveform_exists": waveform_path is not None,
            "waveform_path": str(waveform_path) if waveform_path else None,
            "audio_readable": audio_path.is_file() if audio_path else False,
            "waveform_readable": waveform_path.is_file() if waveform_path else False
        }