"""Hybrid semantic + metadata retrieval utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np

from ..datasets.metadata_table import DatasetMetadataTable


def _normalize(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Expected non-empty [n, d] features, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return cast(np.ndarray, arr / norms)


def _tokenize_keyword(keyword: str | None) -> list[str]:
    if keyword is None:
        return []
    return [token for token in keyword.lower().split() if token]


def _keyword_score(
    track_id: str,
    metadata: DatasetMetadataTable,
    tokens: list[str],
) -> float:
    if not tokens:
        return 0.0
    text = metadata.text_for_track(track_id)
    if not text:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for token in tokens if token in text_lower)
    return matches / len(tokens)


def _passes_filters(
    track_id: str,
    metadata: DatasetMetadataTable,
    filters: Mapping[str, str],
) -> bool:
    row = metadata.get_row(track_id)
    if row is None:
        return False

    for key, expected in filters.items():
        value = row.get(key, "")
        if value.lower().strip() != expected.lower().strip():
            return False
    return True


def hybrid_search(
    *,
    query_track: str,
    features: np.ndarray,
    track_names: list[str],
    metadata: DatasetMetadataTable,
    keyword: str | None = None,
    filters: Mapping[str, str] | None = None,
    semantic_weight: float = 1.0,
    keyword_weight: float = 0.3,
    k: int = 10,
    exclude_self: bool = True,
) -> list[tuple[str, float]]:
    """Combine cosine semantic similarity with metadata keyword/filters."""
    if k <= 0:
        raise ValueError("k must be > 0")
    if semantic_weight < 0:
        raise ValueError("semantic_weight must be >= 0")
    if keyword_weight < 0:
        raise ValueError("keyword_weight must be >= 0")

    keyword_tokens = _tokenize_keyword(keyword)
    if semantic_weight + (keyword_weight if keyword_tokens else 0.0) <= 0:
        raise ValueError("At least one effective weight must be > 0")

    if len(track_names) != int(np.asarray(features).shape[0]):
        raise ValueError("features rows must match track_names length")
    if query_track not in track_names:
        raise ValueError(f"Query track '{query_track}' not found")

    normalized = _normalize(features)
    query_idx = track_names.index(query_track)
    similarities = normalized @ normalized[query_idx]

    active_filters = dict(filters or {})
    fused_scores: list[tuple[str, float]] = []
    for idx, track_id in enumerate(track_names):
        if exclude_self and idx == query_idx:
            continue
        if active_filters and not _passes_filters(track_id, metadata, active_filters):
            continue

        semantic_score = (float(similarities[idx]) + 1.0) / 2.0
        kw_score = _keyword_score(track_id, metadata, keyword_tokens)

        denom = semantic_weight + (keyword_weight if keyword_tokens else 0.0)
        fused = (semantic_weight * semantic_score + keyword_weight * kw_score) / denom
        fused_scores.append((track_id, fused))

    fused_scores.sort(key=lambda item: item[1], reverse=True)
    return fused_scores[:k]
