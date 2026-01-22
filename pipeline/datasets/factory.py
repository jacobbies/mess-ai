"""
Dataset factory for creating dataset instances
"""
from pathlib import Path
from typing import Dict, Type, Optional
from .base import BaseDataset
from .smd import SMDDataset
from .maestro import MAESTRODataset


class DatasetFactory:
    """Factory for creating dataset instances."""

    _datasets: Dict[str, Type[BaseDataset]] = {
        'smd': SMDDataset,
        'maestro': MAESTRODataset,
    }

    # Default data directory (can be overridden)
    _default_data_dir = Path(__file__).parent.parent.parent / "data"

    @classmethod
    def get_dataset(cls, dataset_type: str, data_dir: Optional[Path] = None) -> BaseDataset:
        """
        Get a dataset instance (convenience method).

        Args:
            dataset_type: Type of dataset ('smd', 'maestro', etc.)
            data_dir: Root data directory (defaults to project data dir)

        Returns:
            Dataset instance

        Raises:
            ValueError: If dataset type is not supported
        """
        if data_dir is None:
            data_dir = cls._default_data_dir

        return cls.create_dataset(dataset_type, data_dir)

    @classmethod
    def create_dataset(cls, dataset_type: str, data_dir: Path) -> BaseDataset:
        """
        Create a dataset instance.

        Args:
            dataset_type: Type of dataset ('smd', 'maestro', etc.)
            data_dir: Root data directory

        Returns:
            Dataset instance

        Raises:
            ValueError: If dataset type is not supported
        """
        if dataset_type not in cls._datasets:
            available = ', '.join(cls._datasets.keys())
            raise ValueError(f"Unsupported dataset type '{dataset_type}'. Available: {available}")

        dataset_class = cls._datasets[dataset_type]
        return dataset_class(data_dir)

    @classmethod
    def get_available_datasets(cls) -> list[str]:
        """Get list of available dataset types."""
        return list(cls._datasets.keys())

    @classmethod
    def register_dataset(cls, name: str, dataset_class: Type[BaseDataset]) -> None:
        """Register a new dataset type."""
        cls._datasets[name] = dataset_class