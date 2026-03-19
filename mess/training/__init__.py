"""Retrieval-augmented training utilities for expressive embedding learning."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import RetrievalSSLConfig
    from .context_config import ContextualizerConfig
    from .context_export import (
        contextualize_tracks,
        export_contextualizer_track_artifact,
        load_contextualizer_from_state,
        save_contextualized_embeddings,
    )
    from .context_trainer import ContextTrainResult, TrackSegments, train_contextualizer
    from .contextualizer import SegmentTransformer, late_interaction_score
    from .export import export_projection_clip_artifact, project_vectors_with_head
    from .trainer import ProjectionHead, TrainResult, train_projection_head

__all__ = [
    "RetrievalSSLConfig",
    "ProjectionHead",
    "TrainResult",
    "train_projection_head",
    "project_vectors_with_head",
    "export_projection_clip_artifact",
    "ContextualizerConfig",
    "SegmentTransformer",
    "late_interaction_score",
    "ContextTrainResult",
    "TrackSegments",
    "train_contextualizer",
    "load_contextualizer_from_state",
    "contextualize_tracks",
    "save_contextualized_embeddings",
    "export_contextualizer_track_artifact",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "RetrievalSSLConfig": (".config", "RetrievalSSLConfig"),
    "ProjectionHead": (".trainer", "ProjectionHead"),
    "TrainResult": (".trainer", "TrainResult"),
    "train_projection_head": (".trainer", "train_projection_head"),
    "project_vectors_with_head": (".export", "project_vectors_with_head"),
    "export_projection_clip_artifact": (".export", "export_projection_clip_artifact"),
    "ContextualizerConfig": (".context_config", "ContextualizerConfig"),
    "SegmentTransformer": (".contextualizer", "SegmentTransformer"),
    "late_interaction_score": (".contextualizer", "late_interaction_score"),
    "ContextTrainResult": (".context_trainer", "ContextTrainResult"),
    "TrackSegments": (".context_trainer", "TrackSegments"),
    "train_contextualizer": (".context_trainer", "train_contextualizer"),
    "load_contextualizer_from_state": (".context_export", "load_contextualizer_from_state"),
    "contextualize_tracks": (".context_export", "contextualize_tracks"),
    "save_contextualized_embeddings": (".context_export", "save_contextualized_embeddings"),
    "export_contextualizer_track_artifact": (
        ".context_export",
        "export_contextualizer_track_artifact",
    ),
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
