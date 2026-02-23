import os
from pathlib import Path
from typing import Any


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
        self.MERT_CACHE_DIR: str | None = None
        self.MERT_TARGET_SAMPLE_RATE: int = 24000
        self.MERT_SEGMENT_DURATION: float = 5.0
        self.MERT_OVERLAP_RATIO: float = 0.5

        # Device Configuration (CPU default, explicit GPU opt-in)
        self.MERT_DEVICE: str | None = 'cpu'  # Default: 'cpu', Options: 'cuda', 'mps'

        # Device-specific batch sizes (optimized per hardware)
        self.MERT_BATCH_SIZE_CUDA: int = 16  # RTX 3070Ti can handle 16
        self.MERT_BATCH_SIZE_MPS: int = 8
        self.MERT_BATCH_SIZE_CPU: int = 4

        # CUDA optimizations (RTX 3070Ti Ampere benefits from all of these)
        self.MERT_CUDA_PINNED_MEMORY: bool = True  # 10-20% speedup
        self.MERT_CUDA_NON_BLOCKING: bool = True   # Overlaps transfers
        self.MERT_CUDA_MIXED_PRECISION: bool = True  # ~2x speedup with Tensor Cores
        self.MERT_CUDA_AUTO_OOM_RECOVERY: bool = True  # Auto-reduce batch on OOM

        # Processing Configuration
        self.MERT_MAX_WORKERS: int = 4
        self.MERT_MEMORY_EFFICIENT: bool = False
        self.MERT_CHECKPOINT_INTERVAL: int = 10
        self.MERT_VERBOSE: bool = True

        # Environment variable overrides (useful for Docker/cluster deployment)
        self._apply_env_overrides()
    
    # =========================================================================
    # Model Configuration
    # =========================================================================

    @property
    def model_name(self) -> str:
        """MERT model identifier from Hugging Face."""
        return self.MERT_MODEL_NAME

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides for deployment flexibility."""
        if os.getenv('MESS_DEVICE'):
            # Examples: MESS_DEVICE=cuda, MESS_DEVICE=cpu, MESS_DEVICE=mps
            self.MERT_DEVICE = os.getenv('MESS_DEVICE')

        if os.getenv('MESS_WORKERS'):
            self.MERT_MAX_WORKERS = int(os.getenv('MESS_WORKERS'))

        if os.getenv('MESS_BATCH_SIZE'):
            # Override all batch sizes uniformly
            batch_size = int(os.getenv('MESS_BATCH_SIZE'))
            self.MERT_BATCH_SIZE_CUDA = batch_size
            self.MERT_BATCH_SIZE_MPS = batch_size
            self.MERT_BATCH_SIZE_CPU = batch_size

        # CUDA optimization toggles
        if os.getenv('MESS_CUDA_MIXED_PRECISION'):
            self.MERT_CUDA_MIXED_PRECISION = os.getenv('MESS_CUDA_MIXED_PRECISION') == '1'

    @property
    def device(self) -> str:
        """
        Device selection with CPU as safe default.

        Users must explicitly set device='cuda' or device='mps' to use GPU.
        This prevents accidental GPU usage and OOM errors.
        """
        if self.MERT_DEVICE:
            return self.MERT_DEVICE

        # Fallback to CPU if not explicitly set
        return 'cpu'

    @property
    def cache_dir(self) -> Path | None:
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
        """Device-adaptive batch size for optimal throughput."""
        device = self.device
        if device == 'cuda':
            return self.MERT_BATCH_SIZE_CUDA
        elif device == 'mps':
            return self.MERT_BATCH_SIZE_MPS
        else:
            return self.MERT_BATCH_SIZE_CPU

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

        # Validate device
        if self.device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Invalid device: {self.device}. Must be 'cpu', 'cuda', or 'mps'")

        # Ensure data root exists
        self.data_root.mkdir(parents=True, exist_ok=True)
        print(f"✓ Config validated. Data root: {self.data_root}")
    
    def get_device_info(self) -> dict[str, Any]:
        """Get device information for debugging."""
        import torch
        info = {
            'device': self.device,
            'pytorch_version': torch.__version__,
            'mps_available': torch.backends.mps.is_available(),
            'cuda_available': torch.cuda.is_available(),
            'batch_size': self.batch_size,
        }

        # Add CUDA-specific info
        if self.device == 'cuda' and torch.cuda.is_available():
            info.update({
                'cuda_device_name': torch.cuda.get_device_name(0),
                'cuda_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'cuda_pinned_memory': self.MERT_CUDA_PINNED_MEMORY,
                'cuda_non_blocking': self.MERT_CUDA_NON_BLOCKING,
                'cuda_mixed_precision': self.MERT_CUDA_MIXED_PRECISION,
                'cuda_oom_recovery': self.MERT_CUDA_AUTO_OOM_RECOVERY,
            })

        return info

    def get_path_info(self) -> dict[str, str]:
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
        print(f"  Device:           {self.device.upper()}")
        print(f"  Sample Rate:      {self.target_sample_rate} Hz")
        print(f"  Segment Duration: {self.segment_duration}s")
        print(f"  Overlap Ratio:    {self.overlap_ratio}")
        print(f"  Batch Size:       {self.batch_size}")
        print(f"  Max Workers:      {self.max_workers}")
        print(f"  Memory Efficient: {self.memory_efficient}")
        print(f"  Verbose:          {self.verbose}")

        # Show CUDA optimizations if using CUDA
        if self.device == 'cuda':
            print("-" * 60)
            print("  CUDA Optimizations:")
            print(f"    Pinned Memory:   {self.MERT_CUDA_PINNED_MEMORY}")
            print(f"    Non-blocking:    {self.MERT_CUDA_NON_BLOCKING}")
            print(f"    Mixed Precision: {self.MERT_CUDA_MIXED_PRECISION}")
            print(f"    OOM Recovery:    {self.MERT_CUDA_AUTO_OOM_RECOVERY}")

        print("-" * 60)
        print(f"  Project Root:     {self.project_root}")
        print(f"  Data Root:        {self.data_root}")
        print("=" * 60 + "\n")


# ============================================================================
# Global Configuration Instance
# ============================================================================

mess_config = MESSConfig()
