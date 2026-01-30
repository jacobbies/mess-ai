"""
Dataset factory for creating dataset instances
"""
from pathlib import Path
from typing import Dict, Type, Optional
from .base import BaseDataset
from .smd import SMDDataset
from .maestro import MAESTRODataset


class DatasetFactory:
    """
    Factory for creating dataset instances.

    Uses hybrid approach: datasets default to config.data_root but can be overridden.
    """

    _datasets: Dict[str, Type[BaseDataset]] = {
        'smd': SMDDataset,
        'maestro': MAESTRODataset,
    }

    @classmethod
    def get_dataset(cls, dataset_type: str, data_root: Optional[Path] = None) -> BaseDataset:
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
        if dataset_type not in cls._datasets:
            available = ', '.join(cls._datasets.keys())
            raise ValueError(f"Unsupported dataset type '{dataset_type}'. Available: {available}")

        dataset_class = cls._datasets[dataset_type]
        return dataset_class(data_root)

    @classmethod
    def create_dataset(cls, dataset_type: str, data_root: Optional[Path] = None) -> BaseDataset:
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
        return list(cls._datasets.keys())

    @classmethod
    def register_dataset(cls, name: str, dataset_class: Type[BaseDataset]) -> None:
        """Register a new dataset type."""
        cls._datasets[name] = dataset_class