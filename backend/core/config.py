"""
Application configuration settings.
"""
import os
from pathlib import Path
from typing import Optional
from functools import lru_cache


class Settings:
    """Application settings and configuration."""
    
    # API Configuration
    API_TITLE: str = "Music Similarity API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "AI-powered classical music discovery with MERT embeddings and FAISS similarity search"
    
    # Server Configuration
    HOST: str = os.getenv('HOST') or '0.0.0.0'
    PORT: int = int(os.getenv('PORT') or '8000')
    
    # Environment
    ENVIRONMENT: str = os.getenv('ENVIRONMENT') or 'development'  # development, staging, production
    DEBUG: bool = (os.getenv('DEBUG') or 'false').lower() == 'true'
    
    # Security
    SECRET_KEY: str = os.getenv('SECRET_KEY') or 'dev-secret-key-change-in-production'
    ALLOWED_ORIGINS: list = (os.getenv('ALLOWED_ORIGINS') or 'http://localhost:3000').split(',')
    
    # Database Configuration
    DATABASE_URL: Optional[str] = os.getenv('DATABASE_URL')
    DB_HOST: str = os.getenv('DB_HOST') or 'localhost'
    DB_PORT: int = int(os.getenv('DB_PORT') or '5432')
    DB_NAME: str = os.getenv('DB_NAME') or 'mess_ai'
    DB_USER: str = os.getenv('DB_USER') or 'postgres'
    DB_PASSWORD: str = os.getenv('DB_PASSWORD') or 'password'
    
    # External URLs (for production S3/CDN)
    AUDIO_BASE_URL: Optional[str] = os.getenv('AUDIO_BASE_URL')
    
    # Storage Configuration (flexible for different deployment strategies)
    STORAGE_TYPE: str = os.getenv('STORAGE_TYPE') or 'local'  # local, s3, hybrid
    
    # AWS Configuration (optional)
    AWS_REGION: str = os.getenv('AWS_REGION') or 'us-west-2'
    AWS_S3_BUCKET: Optional[str] = os.getenv('AWS_S3_BUCKET')
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    # File Paths (Environment configurable)
    DATA_ROOT_DIR: str = os.getenv('DATA_ROOT_DIR') or ''
    
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        if self.DATA_ROOT_DIR:
            return Path(self.DATA_ROOT_DIR)
        return self.project_root / "data"
    
    @property
    def wav_dir(self) -> Path:
        """Get the WAV files directory."""
        wav_path = os.getenv('WAV_DIR', '')
        if wav_path:
            return Path(wav_path)
        return self.data_dir / "smd" / "wav-44"
    
    @property
    def features_dir(self) -> Path:
        """Get the features directory."""
        features_path = os.getenv('FEATURES_DIR', '')
        if features_path:
            return Path(features_path)
        return self.data_dir / "processed" / "features"
    
    
    @property
    def metadata_dir(self) -> Path:
        """Get the metadata directory."""
        metadata_path = os.getenv('METADATA_DIR', '')
        if metadata_path:
            return Path(metadata_path)
        return self.data_dir / "metadata"
    
    @property
    def maestro_dir(self) -> Path:
        """Get the Maestro dataset directory."""
        maestro_path = os.getenv('MAESTRO_DIR', '')
        if maestro_path:
            return Path(maestro_path)
        return self.data_dir / "maestro"
    
    # Thread Pool Configuration
    THREAD_POOL_MAX_WORKERS: int = int(os.getenv('THREAD_POOL_MAX_WORKERS') or '4')
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL') or 'INFO'
    LOG_FORMAT: str = os.getenv('LOG_FORMAT') or ('json' if os.getenv('ENVIRONMENT') == 'production' else 'console')
    
    # Dataset Configuration
    DATASET_TYPE: str = os.getenv('DATASET_TYPE') or 'smd'  # smd, maestro, custom
    DATASET_NAME: str = os.getenv('DATASET_NAME') or 'Saarland Music Dataset (SMD)'
    
    # Performance Configuration
    MAX_MEMORY_GB: int = int(os.getenv('MAX_MEMORY_GB') or '8')
    CACHE_SIZE_MB: int = int(os.getenv('CACHE_SIZE_MB') or '1024')
    
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == 'production'
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == 'development'
    
    def validate_config(self) -> None:
        """Validate configuration settings."""
        if self.is_production and self.SECRET_KEY == 'dev-secret-key-change-in-production':
            raise ValueError("Must set SECRET_KEY in production")
        
        if self.is_production and self.DEBUG:
            raise ValueError("DEBUG must be False in production")
        
        # Validate required directories exist
        required_dirs = [self.data_dir, self.features_dir, self.metadata_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"Warning: Required directory does not exist: {dir_path}")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()