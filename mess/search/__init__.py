"""Public API for search and retrieval utilities."""

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
]
