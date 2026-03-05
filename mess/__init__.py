"""Root public API for MESS.

This module intentionally exposes a small, stable surface for discovery and
autocomplete while keeping heavyweight/optional imports lazy.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import MESSConfig
    from .config import mess_config as mess_config
    from .datasets import ClipIndex, ClipRecord, DatasetFactory
    from .extraction import ExtractionPipeline, FeatureExtractor
    from .probing import ASPECT_REGISTRY, LayerDiscoverySystem, resolve_aspects
    from .search import (
        find_similar,
        load_features,
        load_segment_features,
        search_by_aspect,
        search_by_aspects,
        search_by_clip,
    )
    from .training import RetrievalSSLConfig, TrainResult, train_projection_head

__all__ = [
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
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "MESSConfig": (".config", "MESSConfig"),
    "mess_config": (".config", "mess_config"),
    "DatasetFactory": (".datasets", "DatasetFactory"),
    "ClipIndex": (".datasets", "ClipIndex"),
    "ClipRecord": (".datasets", "ClipRecord"),
    "FeatureExtractor": (".extraction", "FeatureExtractor"),
    "ExtractionPipeline": (".extraction", "ExtractionPipeline"),
    "LayerDiscoverySystem": (".probing", "LayerDiscoverySystem"),
    "ASPECT_REGISTRY": (".probing", "ASPECT_REGISTRY"),
    "resolve_aspects": (".probing", "resolve_aspects"),
    "load_features": (".search", "load_features"),
    "load_segment_features": (".search", "load_segment_features"),
    "find_similar": (".search", "find_similar"),
    "search_by_clip": (".search", "search_by_clip"),
    "search_by_aspect": (".search", "search_by_aspect"),
    "search_by_aspects": (".search", "search_by_aspects"),
    "RetrievalSSLConfig": (".training", "RetrievalSSLConfig"),
    "TrainResult": (".training", "TrainResult"),
    "train_projection_head": (".training", "train_projection_head"),
}

_LAZY_MODULES = {"datasets", "extraction", "probing", "search", "training"}


def _resolve_version() -> str:
    try:
        return version("mess-ai")
    except PackageNotFoundError:
        pass
    return "0.0.0+local"


def __getattr__(name: str) -> Any:
    if name == "__version__":
        resolved = _resolve_version()
        globals()[name] = resolved
        return resolved

    if name in _LAZY_MODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
