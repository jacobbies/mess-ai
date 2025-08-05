"""
Pipeline configuration using shared backend settings.
"""

from core.config import settings
from typing import Dict, Any, Optional
from pathlib import Path


class PipelineConfig:
    """
    Pipeline configuration that extends the main backend settings.
    
    This provides a pipeline-specific interface while using the shared
    backend configuration for consistency.
    """
    
    def __init__(self):
        self._settings = settings
    
    # Model Configuration
    @property
    def model_name(self) -> str:
        return self._settings.MERT_MODEL_NAME
    
    @property
    def device(self) -> str:
        return self._settings.mert_device
    
    @property
    def cache_dir(self) -> Optional[Path]:
        return self._settings.mert_cache_dir
    
    # Audio Processing Configuration
    @property
    def target_sample_rate(self) -> int:
        return self._settings.MERT_TARGET_SAMPLE_RATE
    
    @property
    def segment_duration(self) -> float:
        return self._settings.MERT_SEGMENT_DURATION
    
    @property
    def overlap_ratio(self) -> float:
        return self._settings.MERT_OVERLAP_RATIO
    
    # Processing Configuration
    @property
    def batch_size(self) -> int:
        return self._settings.MERT_BATCH_SIZE
    
    @property
    def max_workers(self) -> int:
        return self._settings.MERT_MAX_WORKERS
    
    @property
    def memory_efficient(self) -> bool:
        return self._settings.MERT_MEMORY_EFFICIENT
    
    @property
    def checkpoint_interval(self) -> int:
        return self._settings.MERT_CHECKPOINT_INTERVAL
    
    @property
    def verbose(self) -> bool:
        return self._settings.MERT_VERBOSE
    
    # Path Configuration (from backend settings)
    @property
    def project_root(self) -> Path:
        return self._settings.project_root
    
    @property
    def data_dir(self) -> Path:
        return self._settings.data_dir
    
    @property
    def output_dir(self) -> Path:
        return self._settings.features_dir
    
    @property
    def audio_dir(self) -> Path:
        return self._settings.wav_dir
    
    @property
    def dataset_type(self) -> str:
        return self._settings.DATASET_TYPE
    
    # Validation and utilities
    def validate_config(self) -> None:
        """Validate pipeline configuration."""
        # Validate audio parameters
        if self.segment_duration <= 0:
            raise ValueError("MERT_SEGMENT_DURATION must be positive")
        if not 0 <= self.overlap_ratio < 1:
            raise ValueError("MERT_OVERLAP_RATIO must be between 0 and 1")
        if self.target_sample_rate <= 0:
            raise ValueError("MERT_SAMPLE_RATE must be positive")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Warn if audio directory doesn't exist (not critical for some use cases)
        if not self.audio_dir.exists():
            print(f"Warning: Audio directory does not exist: {self.audio_dir}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information for debugging."""
        import torch
        device = self.device
        info = {
            'device': device,
            'device_auto_detected': not bool(self._settings.MERT_CACHE_DIR)
        }
        
        if device == 'cuda' and torch.cuda.is_available():
            info.update({
                'cuda_version': getattr(torch.version, 'cuda', None),
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0)
            })
        elif device == 'mps' and torch.backends.mps.is_available():
            info.update({
                'mps_available': True,
                'mps_built': torch.backends.mps.is_built()
            })
        
        return info
    
    def get_path_info(self) -> Dict[str, Optional[str]]:
        """Get path information for debugging."""
        return {
            'project_root': str(self.project_root),
            'data_dir': str(self.data_dir),
            'audio_dir': str(self.audio_dir),
            'output_dir': str(self.output_dir),
            'cache_dir': str(self.cache_dir) if self.cache_dir else None
        }
    
    def print_config(self) -> None:
        """Print current configuration for debugging."""
        print("=== MERT Pipeline Configuration ===")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Sample Rate: {self.target_sample_rate}Hz")
        print(f"Segment Duration: {self.segment_duration}s")
        print(f"Overlap Ratio: {self.overlap_ratio}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Dataset: {self.dataset_type}")
        print(f"Audio Dir: {self.audio_dir}")
        print(f"Output Dir: {self.output_dir}")
        print("=====================================")


# Global pipeline config instance
pipeline_config = PipelineConfig()

# Backward compatibility exports
MODEL_NAME = pipeline_config.model_name
DEVICE = pipeline_config.device
BASE_DIR = pipeline_config.project_root