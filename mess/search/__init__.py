"""Public API for search and retrieval utilities."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .faiss_index import (
        ArtifactManifest,
        ArtifactValidationError,
        FAISSArtifact,
        build_clip_artifact,
        build_track_artifact,
        download_artifact_from_s3,
        find_latest_artifact_dir,
        load_artifact,
        load_latest_from_s3,
        remove_local_artifact_dir,
        save_artifact,
        upload_artifact_to_s3,
    )
    from .hybrid import hybrid_search
    from .search import (
        ClipLocation,
        ClipSearchResult,
        build_index,
        find_similar,
        load_features,
        load_segment_features,
        search_by_aspect,
        search_by_aspects,
        search_by_clip,
    )

__all__ = [
    "ClipLocation",
    "ClipSearchResult",
    "build_index",
    "find_similar",
    "load_features",
    "load_segment_features",
    "search_by_aspect",
    "search_by_aspects",
    "search_by_clip",
    "hybrid_search",
    "ArtifactManifest",
    "ArtifactValidationError",
    "FAISSArtifact",
    "build_track_artifact",
    "build_clip_artifact",
    "save_artifact",
    "load_artifact",
    "find_latest_artifact_dir",
    "upload_artifact_to_s3",
    "download_artifact_from_s3",
    "load_latest_from_s3",
    "remove_local_artifact_dir",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ClipLocation": (".search", "ClipLocation"),
    "ClipSearchResult": (".search", "ClipSearchResult"),
    "build_index": (".search", "build_index"),
    "find_similar": (".search", "find_similar"),
    "load_features": (".search", "load_features"),
    "load_segment_features": (".search", "load_segment_features"),
    "search_by_aspect": (".search", "search_by_aspect"),
    "search_by_aspects": (".search", "search_by_aspects"),
    "search_by_clip": (".search", "search_by_clip"),
    "hybrid_search": (".hybrid", "hybrid_search"),
    "ArtifactManifest": (".faiss_index", "ArtifactManifest"),
    "ArtifactValidationError": (".faiss_index", "ArtifactValidationError"),
    "FAISSArtifact": (".faiss_index", "FAISSArtifact"),
    "build_track_artifact": (".faiss_index", "build_track_artifact"),
    "build_clip_artifact": (".faiss_index", "build_clip_artifact"),
    "save_artifact": (".faiss_index", "save_artifact"),
    "load_artifact": (".faiss_index", "load_artifact"),
    "find_latest_artifact_dir": (".faiss_index", "find_latest_artifact_dir"),
    "upload_artifact_to_s3": (".faiss_index", "upload_artifact_to_s3"),
    "download_artifact_from_s3": (".faiss_index", "download_artifact_from_s3"),
    "load_latest_from_s3": (".faiss_index", "load_latest_from_s3"),
    "remove_local_artifact_dir": (".faiss_index", "remove_local_artifact_dir"),
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
