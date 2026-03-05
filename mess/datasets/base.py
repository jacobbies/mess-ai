"""
Base dataset interface for ML workflows.
Simplified for local development - focuses on audio files and feature paths.
"""
from abc import ABC, abstractmethod
from pathlib import Path

from .clip_index import ClipIndex, build_clip_records
from .metadata_table import DatasetMetadataTable


class BaseDataset(ABC):
    """
    Abstract base class for music datasets (ML-focused).

    Hybrid approach: Datasets are self-contained but use config.data_root as default.
    This allows both:
    - Easy default usage: dataset = SMDDataset() uses config.data_root
    - Flexible override: dataset = SMDDataset(data_root="/custom/path")
    """

    def __init__(self, data_root: Path | None = None):
        """
        Initialize dataset with optional data root.

        Args:
            data_root: Root data directory. If None, uses config.data_root.
        """
        if data_root is None:
            from ..config import mess_config
            data_root = mess_config.data_root

        self.data_root = Path(data_root)

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """Stable dataset identifier used in configs and artifact naming."""
        ...

    @property
    @abstractmethod
    def audio_dir(self) -> Path:
        """Directory containing audio files for this dataset."""
        ...

    @property
    @abstractmethod
    def embeddings_dir(self) -> Path:
        """Directory for storing MERT embeddings for this dataset."""
        ...

    @property
    def aggregated_dir(self) -> Path:
        """Directory for aggregated [13, 768] embeddings."""
        return self.embeddings_dir / "aggregated"

    @property
    def segments_dir(self) -> Path:
        """Directory for segment-level [num_segments, 13, 768] embeddings."""
        return self.embeddings_dir / "segments"

    @property
    def metadata_dir(self) -> Path:
        """Directory for dataset-level metadata artifacts."""
        return self.data_root / "metadata"

    @property
    def clip_index_path(self) -> Path:
        """Default clip index path for this dataset."""
        return self.metadata_dir / f"{self.dataset_id}_clip_index.csv"

    @property
    def metadata_table_path(self) -> Path:
        """Default canonical metadata table path for this dataset."""
        return self.metadata_dir / f"{self.dataset_id}_metadata.csv"

    def get_audio_files(self) -> list[Path]:
        """
        Get list of audio file paths for the dataset.

        Returns:
            List of Path objects to .wav files
        """
        if not self.audio_dir.exists():
            return []
        return sorted(
            path
            for path in self.audio_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".wav"
        )

    def get_feature_path(self, track_id: str, feature_type: str = "aggregated") -> Path:
        """
        Get path where features should be saved/loaded for a track.

        Args:
            track_id: Track identifier (usually filename stem)
            feature_type: Type of features (raw, segments, aggregated)

        Returns:
            Path to .npy feature file
        """
        if feature_type == "aggregated":
            return self.aggregated_dir / f"{track_id}.npy"
        else:
            return self.embeddings_dir / feature_type / f"{track_id}.npy"

    def load_metadata_table(
        self,
        path: str | Path | None = None,
        required: bool = False,
    ) -> DatasetMetadataTable | None:
        """
        Load canonical dataset metadata if available.

        Args:
            path: Optional metadata file path override.
            required: Raise when metadata file does not exist.
        """
        resolved_path = Path(path) if path is not None else self.metadata_table_path
        if not resolved_path.exists():
            if required:
                raise FileNotFoundError(f"Metadata table not found: {resolved_path}")
            return None
        return DatasetMetadataTable.from_path(resolved_path)

    def build_clip_index(
        self,
        segments_dir: str | Path | None = None,
        segment_duration: float = 5.0,
        overlap_ratio: float = 0.5,
        default_split: str = "unspecified",
        metadata_table: DatasetMetadataTable | None = None,
        metadata_path: str | Path | None = None,
        assign_splits: bool = False,
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> ClipIndex:
        """
        Build a clip index from per-track segment embeddings.

        If metadata is provided, recording/work identifiers are sourced from it.
        Otherwise recording_id defaults to track_id and work_id remains empty.
        """
        if metadata_table is not None and metadata_path is not None:
            raise ValueError("Provide either metadata_table or metadata_path, not both")

        resolved_segments = Path(segments_dir) if segments_dir is not None else self.segments_dir
        loaded_metadata = metadata_table
        if loaded_metadata is None:
            loaded_metadata = self.load_metadata_table(
                path=metadata_path,
                required=metadata_path is not None,
            )

        recording_map = loaded_metadata.to_recording_map() if loaded_metadata else None
        work_map = loaded_metadata.to_work_map() if loaded_metadata else None

        records = build_clip_records(
            dataset_id=self.dataset_id,
            segments_dir=resolved_segments,
            segment_duration=segment_duration,
            overlap_ratio=overlap_ratio,
            default_split=default_split,
            recording_map=recording_map,
            work_map=work_map,
        )
        index = ClipIndex(records)

        if assign_splits:
            index = index.assign_recording_splits(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )

        return index

    def save_clip_index(
        self,
        index: ClipIndex,
        path: str | Path | None = None,
    ) -> Path:
        """Persist clip index as CSV using path or dataset default location."""
        output_path = Path(path) if path is not None else self.clip_index_path
        index.to_csv(output_path)
        return output_path

    def load_clip_index(self, path: str | Path | None = None) -> ClipIndex:
        """Load clip index from path or dataset default location."""
        index_path = Path(path) if path is not None else self.clip_index_path
        return ClipIndex.from_path(index_path)

    def exists(self) -> bool:
        """Check if dataset directory exists."""
        audio_files = self.get_audio_files()
        return len(audio_files) > 0

    def __len__(self) -> int:
        """Number of audio files in dataset."""
        return len(self.get_audio_files())

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Dataset description."""
        ...
