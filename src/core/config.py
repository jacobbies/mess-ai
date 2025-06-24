"""
Application configuration settings.
"""
import os
from pathlib import Path
from typing import Optional


class Settings:
    """Application settings and configuration."""
    
    # API Configuration
    API_TITLE: str = "Music Similarity API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "AI-powered classical music discovery with MERT embeddings and FAISS similarity search"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # External URLs (for production S3/CDN)
    WAVEFORM_BASE_URL: Optional[str] = os.getenv('WAVEFORM_BASE_URL', None)
    AUDIO_BASE_URL: Optional[str] = os.getenv('AUDIO_BASE_URL', None)
    
    # File Paths
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        return self.project_root / "data"
    
    @property
    def wav_dir(self) -> Path:
        """Get the WAV files directory."""
        return self.data_dir / "smd" / "wav-44"
    
    @property
    def features_dir(self) -> Path:
        """Get the features directory."""
        return self.data_dir / "processed" / "features"
    
    @property
    def waveforms_dir(self) -> Path:
        """Get the waveforms directory."""
        return self.data_dir / "processed" / "waveforms"
    
    @property
    def metadata_dir(self) -> Path:
        """Get the metadata directory."""
        return self.data_dir / "metadata"
    
    # Thread Pool Configuration
    THREAD_POOL_MAX_WORKERS: int = 4
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    # Dataset Information
    DATASET_NAME: str = "Saarland Music Dataset (SMD)"


# Global settings instance
settings = Settings()