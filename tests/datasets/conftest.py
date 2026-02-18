"""Fixtures for dataset tests."""

import pytest

from mess.datasets.factory import DatasetFactory


@pytest.fixture(autouse=True)
def _restore_factory_registry():
    """Save and restore DatasetFactory._datasets around each test."""
    original = DatasetFactory._datasets.copy()
    yield
    DatasetFactory._datasets = original
