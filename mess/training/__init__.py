"""Retrieval-augmented training utilities for expressive embedding learning."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import RetrievalSSLConfig
    from .export import export_projection_clip_artifact, project_vectors_with_head
    from .trainer import ProjectionHead, TrainResult, train_projection_head

__all__ = [
    "RetrievalSSLConfig",
    "ProjectionHead",
    "TrainResult",
    "train_projection_head",
    "project_vectors_with_head",
    "export_projection_clip_artifact",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "RetrievalSSLConfig": (".config", "RetrievalSSLConfig"),
    "ProjectionHead": (".trainer", "ProjectionHead"),
    "TrainResult": (".trainer", "TrainResult"),
    "train_projection_head": (".trainer", "train_projection_head"),
    "project_vectors_with_head": (".export", "project_vectors_with_head"),
    "export_projection_clip_artifact": (".export", "export_projection_clip_artifact"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
