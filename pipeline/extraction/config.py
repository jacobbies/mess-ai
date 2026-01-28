import os
from pathlib import Path
from typing import Dict, Any, Optional


class PipelineConfig:
    """
    Pipeline configuration that extends the main backend settings.
    
    This provides a pipeline-specific interface while using the shared
    backend configuration for consistency.
    """
    
    def __init__(self):
        # Pipeline is independent - no backend dependency
        self.project_root = Path(__file__).parent.parent.parent
        
        # MERT Pipeline Configuration - hardcoded defaults
        self.MERT_MODEL_NAME: str = 'm-a-p/MERT-v1-95M'
        self.MERT_CACHE_DIR: Optional[str] = None
        self.MERT_TARGET_SAMPLE_RATE: int = 24000
        self.MERT_SEGMENT_DURATION: float = 5.0
        self.MERT_OVERLAP_RATIO: float = 0.5
        self.MERT_BATCH_SIZE: int = 8
        self.MERT_MAX_WORKERS: int = 4
        self.MERT_MEMORY_EFFICIENT: bool = False
        self.MERT_CHECKPOINT_INTERVAL: int = 10
        self.MERT_VERBOSE: bool = True
    
    # Model Configuration
    @property
    def model_name(self) -> str:
        return self.MERT_MODEL_NAME
    
    @property
    def device(self) -> str:
        """Get optimal device for MERT processing with auto-detection."""
        import torch
        # Auto-detect best available device
        if torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    @property
    def cache_dir(self) -> Optional[Path]:
        if self.MERT_CACHE_DIR:
            return Path(self.MERT_CACHE_DIR)
        return None
    
    # Audio Processing Configuration
    @property
    def target_sample_rate(self) -> int:
        return self.MERT_TARGET_SAMPLE_RATE
    
    @property
    def segment_duration(self) -> float:
        return self.MERT_SEGMENT_DURATION
    
    @property
    def overlap_ratio(self) -> float:
        return self.MERT_OVERLAP_RATIO
    
    # Processing Configuration
    @property
    def batch_size(self) -> int:
        return self.MERT_BATCH_SIZE
    
    @property
    def max_workers(self) -> int:
        return self.MERT_MAX_WORKERS
    
    @property
    def memory_efficient(self) -> bool:
        return self.MERT_MEMORY_EFFICIENT
    
    @property
    def checkpoint_interval(self) -> int:
        return self.MERT_CHECKPOINT_INTERVAL
    
    @property
    def verbose(self) -> bool:
        return self.MERT_VERBOSE
    
    # Path Configuration (from backend settings)
    # project_root is already set in __init__
    
    # Path Configuration
    @property
    def data_root(self) -> Path:
        """
        Root data directory - single source of truth for data location.

        Datasets use this as their default base directory.
        """
        return self.project_root / "data"

    @property
    def data_dir(self) -> Path:
        """Alias for data_root (backward compatibility)."""
        return self.data_root

    @property
    def output_dir(self) -> Path:
        """Legacy output directory for processed features."""
        return self.data_root / "processed" / "features"

    @property
    def probing_results_file(self) -> Path:
        """Layer discovery validation results JSON."""
        return self.project_root / "pipeline" / "probing" / "layer_discovery_results.json"

    @property
    def proxy_targets_dir(self) -> Path:
        """Proxy target storage for validation."""
        return self.data_root / "proxy_targets"

    # ============================================================================
    # DEPRECATED METHODS - Use Dataset classes instead
    # ============================================================================
    # These methods are kept for backward compatibility with existing code.
    # New code should instantiate dataset classes directly:
    #   from pipeline.datasets.factory import DatasetFactory
    #   dataset = DatasetFactory.get_dataset("maestro")
    #   audio_dir = dataset.audio_dir
    # ============================================================================

    def get_audio_dir(self, dataset: str = "smd") -> Path:
        """
        DEPRECATED: Use dataset.audio_dir instead.

        Get audio directory for a specific dataset.
        """
        if dataset == "smd":
            return self.data_root / "audio" / "smd" / "wav-44"
        elif dataset == "maestro":
            return self.data_root / "audio" / "maestro"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def get_embeddings_dir(self, dataset: str = "smd") -> Path:
        """
        DEPRECATED: Use dataset.embeddings_dir instead.

        Get embeddings directory for a specific dataset.
        """
        if dataset == "smd":
            return self.data_root / "embeddings" / "smd-emb"
        elif dataset == "maestro":
            return self.data_root / "embeddings" / "maestro-emb"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def get_aggregated_features_dir(self, dataset: str = "smd") -> Path:
        """
        DEPRECATED: Use dataset.aggregated_dir instead.

        Get aggregated features directory for a specific dataset.
        """
        return self.get_embeddings_dir(dataset) / "aggregated"

    @property
    def audio_dir(self) -> Path:
        """DEPRECATED: Use dataset.audio_dir instead. Defaults to SMD."""
        return self.get_audio_dir("smd")

    @property
    def aggregated_features_dir(self) -> Path:
        """DEPRECATED: Use dataset.aggregated_dir instead. Defaults to SMD."""
        return self.get_aggregated_features_dir("smd")

    @property
    def smd_audio_dir(self) -> Path:
        """DEPRECATED: Use SMDDataset().audio_dir instead."""
        return self.get_audio_dir("smd")

    @property
    def smd_embeddings_dir(self) -> Path:
        """DEPRECATED: Use SMDDataset().embeddings_dir instead."""
        return self.get_embeddings_dir("smd")

    @property
    def dataset_type(self) -> str:
        return os.getenv('DATASET_TYPE', 'smd')
    
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
            'device_auto_detected': True,
            'pytorch_version': torch.__version__,
            'available_devices': []
        }
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