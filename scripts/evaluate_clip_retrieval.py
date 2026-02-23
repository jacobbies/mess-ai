#!/usr/bin/env python3
"""Clip-level retrieval sanity evaluation (Recall@K)."""

import argparse
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from mess.search.search import ClipLocation, load_segment_features


def _parse_k_values(raw: str) -> List[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(k <= 0 for k in values):
        raise ValueError("--k-values must contain positive integers (e.g., 1,5,10)")
    return values


def _sample_indices(total: int, max_queries: int, rng: np.random.Generator) -> np.ndarray:
    if total <= 0:
        return np.array([], dtype=np.int64)
    if max_queries <= 0 or max_queries >= total:
        return np.arange(total, dtype=np.int64)
    return np.sort(rng.choice(total, size=max_queries, replace=False))


def _build_index(features: np.ndarray) -> faiss.IndexFlatIP:
    indexed = features.astype("float32", copy=True)
    faiss.normalize_L2(indexed)
    index = faiss.IndexFlatIP(indexed.shape[1])
    index.add(indexed)
    return index


def _search_indices(
    index: faiss.IndexFlatIP,
    query_vector: np.ndarray,
    k: int,
) -> List[int]:
    query = query_vector.astype("float32", copy=True).reshape(1, -1)
    faiss.normalize_L2(query)
    _, indices = index.search(query, k)
    return [int(i) for i in indices[0] if i >= 0]


def evaluate_synthetic_recall(
    features: np.ndarray,
    k_values: List[int],
    noise_std: float,
    max_queries: int,
    rng: np.random.Generator,
) -> Dict[int, float]:
    if noise_std < 0:
        raise ValueError("noise_std must be >= 0")

    index = _build_index(features)
    max_k = max(k_values)
    query_indices = _sample_indices(len(features), max_queries=max_queries, rng=rng)

    hits = {k: 0 for k in k_values}
    for idx in query_indices:
        noisy_query = features[idx] + rng.normal(0.0, noise_std, size=features.shape[1]).astype(
            np.float32
        )
        retrieved = _search_indices(index, noisy_query, max_k)

        for k in k_values:
            if int(idx) in retrieved[:k]:
                hits[k] += 1

    denom = max(len(query_indices), 1)
    return {k: hits[k] / denom for k in k_values}


def evaluate_same_piece_nearby_recall(
    features: np.ndarray,
    clip_locations: List[ClipLocation],
    k_values: List[int],
    nearby_window_seconds: float,
    max_queries: int,
    rng: np.random.Generator,
) -> tuple[Dict[int, float], int]:
    if nearby_window_seconds <= 0:
        raise ValueError("nearby_window_seconds must be > 0")

    index = _build_index(features)
    max_k = max(k_values) + 1  # include potential self-hit

    per_track_indices: Dict[str, List[int]] = {}
    for idx, loc in enumerate(clip_locations):
        per_track_indices.setdefault(loc.track_id, []).append(idx)

    candidate_queries: List[int] = []
    positive_sets: Dict[int, set[int]] = {}

    for idx, loc in enumerate(clip_locations):
        positives = {
            other_idx
            for other_idx in per_track_indices[loc.track_id]
            if other_idx != idx
            and abs(clip_locations[other_idx].start_time - loc.start_time) <= nearby_window_seconds
        }
        if positives:
            candidate_queries.append(idx)
            positive_sets[idx] = positives

    if not candidate_queries:
        return {k: 0.0 for k in k_values}, 0

    sampled_idx_positions = _sample_indices(
        len(candidate_queries), max_queries=max_queries, rng=rng
    )
    sampled_queries = [candidate_queries[pos] for pos in sampled_idx_positions]

    hits = {k: 0 for k in k_values}
    for query_idx in sampled_queries:
        retrieved = _search_indices(index, features[query_idx], max_k)
        retrieved_no_self = [idx for idx in retrieved if idx != query_idx]
        positives = positive_sets[query_idx]

        for k in k_values:
            if any(idx in positives for idx in retrieved_no_self[:k]):
                hits[k] += 1

    denom = len(sampled_queries)
    return ({k: hits[k] / denom for k in k_values}, denom)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate clip-level retrieval sanity")
    parser.add_argument("--dataset", default="smd", help="Dataset name (default: smd)")
    parser.add_argument(
        "--features-dir",
        help="Override path to segment embeddings directory (defaults to data/embeddings/<dataset>-emb/segments)",
    )
    parser.add_argument("--layer", type=int, help="Optional MERT layer (0-12)")
    parser.add_argument("--k-values", default="1,5,10", help="Comma-separated K list")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-queries", type=int, default=500, help="Max query clips per metric")
    parser.add_argument(
        "--synthetic-noise-std",
        type=float,
        default=0.01,
        help="Gaussian noise std for synthetic-positive queries",
    )
    parser.add_argument(
        "--nearby-window",
        type=float,
        default=10.0,
        help="Positive window in seconds for same-piece nearby recall",
    )
    args = parser.parse_args()

    k_values = _parse_k_values(args.k_values)
    features_dir = Path(args.features_dir) if args.features_dir else Path(
        f"data/embeddings/{args.dataset}-emb/segments"
    )

    if not features_dir.exists():
        print(f"Error: features directory not found: {features_dir}")
        return 1

    features, clip_locations = load_segment_features(
        features_dir=str(features_dir),
        layer=args.layer,
    )

    if len(features) == 0:
        print("Error: no segment vectors loaded")
        return 1

    rng = np.random.default_rng(args.seed)

    synthetic = evaluate_synthetic_recall(
        features=features,
        k_values=k_values,
        noise_std=args.synthetic_noise_std,
        max_queries=args.max_queries,
        rng=rng,
    )

    nearby, nearby_n = evaluate_same_piece_nearby_recall(
        features=features,
        clip_locations=clip_locations,
        k_values=k_values,
        nearby_window_seconds=args.nearby_window,
        max_queries=args.max_queries,
        rng=rng,
    )

    print("\nClip Retrieval Sanity Evaluation")
    print("=" * 70)
    print(f"Features dir: {features_dir}")
    print(f"Total indexed clips: {len(clip_locations)}")
    print(f"K values: {k_values}")

    print("\nSynthetic positives (query = clip + noise):")
    for k in k_values:
        print(f"Recall@{k}: {synthetic[k]:.4f}")

    print(
        "\nSame-piece nearby positives "
        f"(within +/-{args.nearby_window:.1f}s, queries with >=1 positive: {nearby_n}):"
    )
    for k in k_values:
        print(f"Recall@{k}: {nearby[k]:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
