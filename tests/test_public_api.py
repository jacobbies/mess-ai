"""Tests for root-level public API exports in ``mess``."""

from __future__ import annotations

import pytest

import mess

pytestmark = pytest.mark.unit


def test_root_public_api_contract_names() -> None:
    expected = {
        "__version__",
        "MESSConfig",
        "mess_config",
        "DatasetFactory",
        "ClipIndex",
        "ClipRecord",
        "FeatureExtractor",
        "ExtractionPipeline",
        "LayerDiscoverySystem",
        "ASPECT_REGISTRY",
        "resolve_aspects",
        "load_features",
        "load_segment_features",
        "find_similar",
        "search_by_clip",
        "search_by_aspect",
        "search_by_aspects",
        "RetrievalSSLConfig",
        "TrainResult",
        "train_projection_head",
        "datasets",
        "extraction",
        "probing",
        "search",
        "training",
    }

    assert expected.issubset(set(mess.__all__))
    assert expected.issubset(set(dir(mess)))


def test_root_dataset_exports_are_accessible() -> None:
    assert mess.DatasetFactory.__name__ == "DatasetFactory"
    assert mess.ClipIndex.__name__ == "ClipIndex"
    assert mess.ClipRecord.__name__ == "ClipRecord"


def test_root_subpackages_are_accessible() -> None:
    assert mess.datasets.__name__ == "mess.datasets"
    assert mess.search.__name__ == "mess.search"


def test_root_search_callable_is_accessible() -> None:
    assert callable(mess.find_similar)


def test_root_version_is_non_empty() -> None:
    assert isinstance(mess.__version__, str)
    assert mess.__version__
