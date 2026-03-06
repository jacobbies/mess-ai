"""Ranking metrics for retrieval evaluation."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from typing import TypeVar

ItemT = TypeVar("ItemT")


def _as_relevant_set(relevant_items: Collection[ItemT]) -> set[ItemT]:
    return set(relevant_items)


def recall_at_k(
    ranked_items: Sequence[ItemT],
    relevant_items: Collection[ItemT],
    k: int,
) -> float:
    """Compute Recall@K for one query."""
    if k <= 0:
        raise ValueError("k must be > 0")

    relevant = _as_relevant_set(relevant_items)
    if not relevant:
        return 0.0

    top_k = ranked_items[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(
    ranked_items: Sequence[ItemT],
    relevant_items: Collection[ItemT],
) -> float:
    """Compute reciprocal rank for one query."""
    relevant = _as_relevant_set(relevant_items)
    if not relevant:
        return 0.0

    for rank, item in enumerate(ranked_items, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_items: Sequence[ItemT],
    relevant_items: Collection[ItemT],
    k: int,
) -> float:
    """Compute nDCG@K for one query with binary relevance."""
    if k <= 0:
        raise ValueError("k must be > 0")

    relevant = _as_relevant_set(relevant_items)
    if not relevant:
        return 0.0

    dcg = 0.0
    for idx, item in enumerate(ranked_items[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(idx + 1.0)

    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(idx + 1.0) for idx in range(1, ideal_hits + 1))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def evaluate_rankings(
    rankings: Sequence[Sequence[ItemT]],
    relevant_sets: Sequence[Collection[ItemT]],
    k_values: Sequence[int],
) -> dict[str, object]:
    """Aggregate Recall@K, MRR, and nDCG@K across many queries."""
    if len(rankings) != len(relevant_sets):
        raise ValueError(
            "rankings and relevant_sets must have the same length: "
            f"{len(rankings)} vs {len(relevant_sets)}"
        )
    if not k_values:
        raise ValueError("k_values must be non-empty")

    sorted_k = sorted({int(k) for k in k_values})
    if any(k <= 0 for k in sorted_k):
        raise ValueError("k_values must be positive integers")

    valid_pairs = [
        (ranking, relevant)
        for ranking, relevant in zip(rankings, relevant_sets, strict=True)
        if len(set(relevant)) > 0
    ]

    if not valid_pairs:
        return {
            "query_count": 0,
            "mrr": 0.0,
            "recall": {f"@{k}": 0.0 for k in sorted_k},
            "ndcg": {f"@{k}": 0.0 for k in sorted_k},
        }

    total_mrr = 0.0
    total_recall = {k: 0.0 for k in sorted_k}
    total_ndcg = {k: 0.0 for k in sorted_k}

    for ranking, relevant in valid_pairs:
        total_mrr += reciprocal_rank(ranking, relevant)
        for k in sorted_k:
            total_recall[k] += recall_at_k(ranking, relevant, k)
            total_ndcg[k] += ndcg_at_k(ranking, relevant, k)

    n_queries = len(valid_pairs)
    return {
        "query_count": n_queries,
        "mrr": total_mrr / n_queries,
        "recall": {f"@{k}": total_recall[k] / n_queries for k in sorted_k},
        "ndcg": {f"@{k}": total_ndcg[k] / n_queries for k in sorted_k},
    }
