from pathlib import Path
from typing import Dict, Any, Optional


class MESSConfig:
    """
    Global MESS configuration for MERT feature extraction.

    Provides MERT model settings and global paths (project_root, data_root).
    Dataset-specific paths (audio, embeddings) should use Dataset classes.
    """

    def __init__(self):
        # Project structure
        self.project_root = Path(__file__).parent.parent

        # MERT Model Configuration
        self.MERT_MODEL_NAME: str = 'm-a-p/MERT-v1-95M'
        self.MERT_CACHE_DIR: Optional[str] = None
        self.MERT_TARGET_SAMPLE_RATE: int = 24000
        self.MERT_SEGMENT_DURATION: float = 5.0
        self.MERT_OVERLAP_RATIO: float = 0.5

        # Processing Configuration
        self.MERT_BATCH_SIZE: int = 8
        self.MERT_MAX_WORKERS: int = 4
        self.MERT_MEMORY_EFFICIENT: bool = False
        self.MERT_CHECKPOINT_INTERVAL: int = 10
        self.MERT_VERBOSE: bool = True
    
    # =========================================================================
    # Model Configuration
    # =========================================================================

    @property
    def model_name(self) -> str:
        """MERT model identifier from Hugging Face."""
        return self.MERT_MODEL_NAME

    @property
    def device(self) -> str:
        """Auto-detect optimal device (CUDA → MPS → CPU)."""
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    @property
    def cache_dir(self) -> Optional[Path]:
        """Optional cache directory for Hugging Face models."""
        if self.MERT_CACHE_DIR:
            return Path(self.MERT_CACHE_DIR)
        return None

    # =========================================================================
    # Audio Processing Configuration
    # =========================================================================

    @property
    def target_sample_rate(self) -> int:
        """Target sample rate for MERT (24kHz)."""
        return self.MERT_TARGET_SAMPLE_RATE

    @property
    def segment_duration(self) -> float:
        """Duration of audio segments in seconds."""
        return self.MERT_SEGMENT_DURATION

    @property
    def overlap_ratio(self) -> float:
        """Overlap ratio between consecutive segments (0.0-1.0)."""
        return self.MERT_OVERLAP_RATIO

    # =========================================================================
    # Processing Configuration
    # =========================================================================

    @property
    def batch_size(self) -> int:
        """Batch size for MERT inference."""
        return self.MERT_BATCH_SIZE

    @property
    def max_workers(self) -> int:
        """Number of worker threads for parallel processing."""
        return self.MERT_MAX_WORKERS

    @property
    def memory_efficient(self) -> bool:
        """Enable memory-efficient mode (slower but uses less RAM)."""
        return self.MERT_MEMORY_EFFICIENT

    @property
    def checkpoint_interval(self) -> int:
        """Save checkpoint every N tracks during extraction."""
        return self.MERT_CHECKPOINT_INTERVAL

    @property
    def verbose(self) -> bool:
        """Enable verbose logging during extraction."""
        return self.MERT_VERBOSE
    
    # =========================================================================
    # Global Path Configuration
    # =========================================================================
    # For dataset-specific paths (audio, embeddings), use Dataset classes:
    #   from mess.datasets.factory import DatasetFactory
    #   dataset = DatasetFactory.get_dataset("smd")
    #   audio_dir = dataset.audio_dir
    #   embeddings_dir = dataset.embeddings_dir
    # =========================================================================

    @property
    def data_root(self) -> Path:
        """
        Root data directory - base path for all datasets.

        Structure:
            data/
            ├── audio/          # Raw audio files (managed by Dataset classes)
            │   ├── smd/
            │   └── maestro/
            └── embeddings/     # MERT features (managed by Dataset classes)
                ├── smd-emb/
                └── maestro-emb/
        """
        return self.project_root / "data"

    @property
    def probing_results_file(self) -> Path:
        """Layer discovery validation results JSON."""
        return self.project_root / "mess" / "probing" / "layer_discovery_results.json"

    @property
    def proxy_targets_dir(self) -> Path:
        """Proxy target storage for layer validation experiments."""
        return self.data_root / "proxy_targets"
    
    # =========================================================================
    # Validation and Utilities
    # =========================================================================

    def validate_config(self) -> None:
        """Validate MESS configuration parameters."""
        # Validate audio processing parameters
        if self.segment_duration <= 0:
            raise ValueError("segment_duration must be positive")
        if not 0 <= self.overlap_ratio < 1:
            raise ValueError("overlap_ratio must be in range [0, 1)")
        if self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        # Ensure data root exists
        self.data_root.mkdir(parents=True, exist_ok=True)
        print(f"✓ Config validated. Data root: {self.data_root}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information for debugging."""
        import torch
        return {
            'device': self.device,
            'pytorch_version': torch.__version__,
            'mps_available': torch.backends.mps.is_available(),
            'cuda_available': torch.cuda.is_available()
        }

    def get_path_info(self) -> Dict[str, str]:
        """Get path information for debugging."""
        return {
            'project_root': str(self.project_root),
            'data_root': str(self.data_root),
            'probing_results': str(self.probing_results_file),
            'proxy_targets': str(self.proxy_targets_dir),
            'cache_dir': str(self.cache_dir) if self.cache_dir else 'None'
        }
    
    def print_config(self) -> None:
        """Print current configuration for debugging."""
        print("\n" + "=" * 60)
        print("MESS Configuration".center(60))
        print("=" * 60)
        print(f"  Model:            {self.model_name}")
        print(f"  Device:           {self.device}")
        print(f"  Sample Rate:      {self.target_sample_rate} Hz")
        print(f"  Segment Duration: {self.segment_duration}s")
        print(f"  Overlap Ratio:    {self.overlap_ratio}")
        print(f"  Batch Size:       {self.batch_size}")
        print(f"  Max Workers:      {self.max_workers}")
        print(f"  Memory Efficient: {self.memory_efficient}")
        print(f"  Verbose:          {self.verbose}")
        print("-" * 60)
        print(f"  Project Root:     {self.project_root}")
        print(f"  Data Root:        {self.data_root}")
        print("=" * 60 + "\n")


# ============================================================================
# Global Configuration Instance
# ============================================================================

mess_config = MESSConfig()
