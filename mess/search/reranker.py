"""Two-stage search: FAISS first-pass + late-interaction reranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..training.contextualizer import late_interaction_score
from .search import _require_faiss


@dataclass(frozen=True)
class RerankResult:
    """Result from two-stage search with reranking scores."""

    track_id: str
    global_score: float
    rerank_score: float
    combined_score: float


def load_global_vectors(
    global_dir: str | Path,
) -> tuple[np.ndarray, list[str]]:
    """Load all global track vectors from a directory.

    Each file is expected as {track_id}.npy containing a [context_dim] vector.

    Returns:
        vectors: [N, context_dim] stacked global vectors.
        track_ids: List of track identifiers.
    """
    gdir = Path(global_dir)
    files = sorted(gdir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {gdir}")

    track_ids: list[str] = []
    vectors: list[np.ndarray] = []

    for f in files:
        track_ids.append(f.stem)
        vectors.append(np.load(f).astype(np.float32))

    return np.stack(vectors, axis=0), track_ids


def load_local_matrices(
    local_dir: str | Path,
    track_ids: list[str],
) -> list[np.ndarray]:
    """Load local segment matrices for specific tracks.

    Args:
        local_dir: Directory containing {track_id}.npy files.
        track_ids: Track IDs to load.

    Returns:
        List of [T_i, context_dim] arrays, one per track.
    """
    ldir = Path(local_dir)
    matrices: list[np.ndarray] = []

    for tid in track_ids:
        path = ldir / f"{tid}.npy"
        matrices.append(np.load(path).astype(np.float32))

    return matrices


def two_stage_search(
    query_track: str,
    global_dir: str | Path,
    local_dir: str | Path,
    k: int = 10,
    first_pass_k: int = 100,
    rerank_weight: float = 0.5,
) -> list[RerankResult]:
    """Two-stage track retrieval: FAISS global search + late-interaction reranking.

    Stage 1: Build FAISS IndexFlatIP from global vectors, retrieve top `first_pass_k`.
    Stage 2: Compute ColBERT-style MaxSim between query and candidate local vectors.
    Combined: (1 - w) * global_score + w * rerank_score.

    Args:
        query_track: Track ID of the query.
        global_dir: Directory with global track vectors (per-track .npy files).
        local_dir: Directory with local segment matrices (per-track .npy files).
        k: Number of final results to return.
        first_pass_k: Number of candidates from FAISS first pass.
        rerank_weight: Weight for reranking score (0 = global only, 1 = rerank only).

    Returns:
        Top-k results sorted by combined score, excluding the query track.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if first_pass_k < k:
        raise ValueError("first_pass_k must be >= k")
    if not 0 <= rerank_weight <= 1:
        raise ValueError("rerank_weight must be in [0, 1]")

    faiss = _require_faiss()

    # Stage 1: FAISS global search
    global_vectors, track_ids = load_global_vectors(global_dir)

    if query_track not in track_ids:
        raise KeyError(f"Query track {query_track!r} not found in global vectors")

    query_idx = track_ids.index(query_track)

    # Build flat index
    normalized = global_vectors.copy()
    faiss.normalize_L2(normalized)
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)

    # Search (retrieve extra to account for self-exclusion)
    query_vec = normalized[query_idx : query_idx + 1]
    actual_k = min(first_pass_k + 1, len(track_ids))
    distances, indices = index.search(query_vec, actual_k)

    # Filter out self
    candidate_indices: list[int] = []
    candidate_scores: list[float] = []
    for idx, score in zip(indices[0], distances[0], strict=False):
        if int(idx) == query_idx:
            continue
        candidate_indices.append(int(idx))
        candidate_scores.append(float(score))
        if len(candidate_indices) >= first_pass_k:
            break

    if not candidate_indices:
        return []

    # Stage 2: Late-interaction reranking
    candidate_track_ids = [track_ids[i] for i in candidate_indices]

    query_local = np.load(Path(local_dir) / f"{query_track}.npy").astype(np.float32)
    candidate_locals = load_local_matrices(local_dir, candidate_track_ids)

    # Compute reranking scores
    query_t = torch.from_numpy(query_local).unsqueeze(0)  # [1, T_q, D]
    query_len = torch.tensor([query_local.shape[0]])

    rerank_scores: list[float] = []
    for cand_local in candidate_locals:
        cand_t = torch.from_numpy(cand_local).unsqueeze(0)  # [1, T_c, D]
        cand_len = torch.tensor([cand_local.shape[0]])
        with torch.no_grad():
            score = late_interaction_score(query_t, cand_t, query_len, cand_len)
        rerank_scores.append(float(score.item()))

    # Combine scores
    results: list[RerankResult] = []
    for i, cand_idx in enumerate(candidate_indices):
        global_score = candidate_scores[i]
        rerank_score = rerank_scores[i]
        combined = (1 - rerank_weight) * global_score + rerank_weight * rerank_score
        results.append(
            RerankResult(
                track_id=track_ids[cand_idx],
                global_score=global_score,
                rerank_score=rerank_score,
                combined_score=combined,
            )
        )

    results.sort(key=lambda r: r.combined_score, reverse=True)
    return results[:k]
