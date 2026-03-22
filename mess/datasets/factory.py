"""
Dataset factory for creating dataset instances
"""
from pathlib import Path

from .base import BaseDataset


class SMDDataset(BaseDataset):
    """
    Saarland Music Dataset - 50 classical piano pieces at 44kHz.

    Self-contained dataset with its own directory structure:
    - Audio: data_root/audio/smd/wav-44/
    - Embeddings: data_root/embeddings/smd-emb/
    """

    dataset_id = "smd"
    audio_subdir = Path("audio") / "smd" / "wav-44"
    embeddings_subdir = Path("embeddings") / "smd-emb"
    name = "Saarland Music Dataset (SMD)"
    description = "Classical piano dataset with 50 recordings (44kHz WAV)"


class MAESTRODataset(BaseDataset):
    """
    MAESTRO Dataset - Large-scale classical piano performances.

    Self-contained dataset with its own directory structure:
    - Audio: data_root/audio/maestro/
    - Embeddings: data_root/embeddings/maestro-emb/
    """

    dataset_id = "maestro"
    audio_subdir = Path("audio") / "maestro"
    embeddings_subdir = Path("embeddings") / "maestro-emb"
    name = "MAESTRO Dataset"
    description = (
        "Classical piano performances dataset "
        "(MIDI Aligned Edited Synchronized TRack of Orchestral)"
    )


DEFAULT_DATASETS: dict[str, type[BaseDataset]] = {
    "smd": SMDDataset,
    "maestro": MAESTRODataset,
}


class DatasetFactory:
    """
    Factory for creating dataset instances.

    Uses a small registry of dataset classes keyed by dataset_id.
    """

    _datasets: dict[str, type[BaseDataset]] = DEFAULT_DATASETS.copy()

    @classmethod
    def _resolve_dataset_class(cls, dataset_type: str) -> type[BaseDataset]:
        try:
            return cls._datasets[dataset_type]
        except KeyError as exc:
            available = ", ".join(cls._datasets)
            raise ValueError(
                f"Unsupported dataset type '{dataset_type}'. Available: {available}"
            ) from exc

    @classmethod
    def get_dataset(cls, dataset_type: str, data_root: Path | None = None) -> BaseDataset:
        """
        Get a dataset instance.

        Args:
            dataset_type: Type of dataset ('smd', 'maestro', etc.)
            data_root: Root data directory (defaults to config.data_root)

        Returns:
            Dataset instance

        Raises:
            ValueError: If dataset type is not supported

        Examples:
            # Use default config.data_root
            dataset = DatasetFactory.get_dataset('smd')

            # Override with custom path
            dataset = DatasetFactory.get_dataset('maestro', Path('/custom/data'))
        """
        dataset_class = cls._resolve_dataset_class(dataset_type)
        return dataset_class(data_root)

    @classmethod
    def create_dataset(cls, dataset_type: str, data_root: Path | None = None) -> BaseDataset:
        """
        Create a dataset instance (alias for get_dataset).

        Args:
            dataset_type: Type of dataset ('smd', 'maestro', etc.)
            data_root: Root data directory (defaults to config.data_root)

        Returns:
            Dataset instance

        Raises:
            ValueError: If dataset type is not supported
        """
        return cls.get_dataset(dataset_type, data_root)

    @classmethod
    def get_available_datasets(cls) -> list[str]:
        """Get list of available dataset types."""
        return list(cls._datasets)

    @classmethod
    def register_dataset(cls, name: str, dataset_class: type[BaseDataset]) -> None:
        """Register a new dataset type."""
        if not name:
            raise ValueError("Dataset name must be non-empty.")
        if not issubclass(dataset_class, BaseDataset):
            raise TypeError("dataset_class must inherit from BaseDataset.")
        cls._datasets[name] = dataset_class
