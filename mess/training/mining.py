"""Neighbor mining utilities for retrieval-augmented training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..datasets.clip_index import ClipRecord


@dataclass(frozen=True)
class MiningParams:
    """Rules for positive/negative mining from retrieved neighbors."""

    positives_per_query: int = 4
    negatives_per_query: int = 16
    min_time_separation_sec: float = 5.0
    require_cross_recording_positive: bool = False
    exclude_same_recording_negative: bool = True

    def validate(self) -> None:
        if self.positives_per_query <= 0:
            raise ValueError("positives_per_query must be > 0")
        if self.negatives_per_query <= 0:
            raise ValueError("negatives_per_query must be > 0")
        if self.min_time_separation_sec < 0:
            raise ValueError("min_time_separation_sec must be >= 0")


def _is_near_same_track(
    query: ClipRecord,
    candidate: ClipRecord,
    min_time_separation_sec: float,
) -> bool:
    if query.track_id != candidate.track_id:
        return False
    return abs(query.start_sec - candidate.start_sec) < min_time_separation_sec


def _valid_positive(query: ClipRecord, candidate: ClipRecord, params: MiningParams) -> bool:
    if query.clip_id == candidate.clip_id:
        return False
    if _is_near_same_track(query, candidate, params.min_time_separation_sec):
        return False
    if params.require_cross_recording_positive and query.recording_id == candidate.recording_id:
        return False
    return True


def _valid_negative(query: ClipRecord, candidate: ClipRecord, params: MiningParams) -> bool:
    if query.clip_id == candidate.clip_id:
        return False
    if _is_near_same_track(query, candidate, params.min_time_separation_sec):
        return False
    if params.exclude_same_recording_negative and query.recording_id == candidate.recording_id:
        return False
    return True


def mine_batch_indices(
    records: Sequence[ClipRecord],
    query_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    params: MiningParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mine positive and negative indices for each query in a batch.

    Args:
        records: Clip metadata aligned with index vectors.
        query_indices: Global vector indices for each query [batch].
        neighbor_indices: Retrieved neighbor indices [batch, search_k].
        params: Mining rules.
        rng: Random generator for fallback negative sampling.

    Returns:
        positive_indices: [batch, positives_per_query]
        positive_mask: [batch, positives_per_query]
        negative_indices: [batch, negatives_per_query]
    """
    params.validate()

    if query_indices.ndim != 1:
        raise ValueError("query_indices must be 1D")
    if neighbor_indices.ndim != 2 or neighbor_indices.shape[0] != query_indices.shape[0]:
        raise ValueError("neighbor_indices must be 2D with rows matching query_indices")

    n_items = len(records)
    all_indices = np.arange(n_items)
    batch = query_indices.shape[0]

    positives = np.zeros((batch, params.positives_per_query), dtype=np.int64)
    positive_mask = np.zeros((batch, params.positives_per_query), dtype=bool)
    negatives = np.zeros((batch, params.negatives_per_query), dtype=np.int64)

    for row_idx, query_idx in enumerate(query_indices):
        query_record = records[int(query_idx)]
        positive_candidates: list[int] = []
        negative_candidates: list[int] = []

        for candidate_idx in neighbor_indices[row_idx]:
            if candidate_idx < 0:
                continue
            candidate_record = records[int(candidate_idx)]

            if _valid_positive(query_record, candidate_record, params):
                positive_candidates.append(int(candidate_idx))
                continue

            if _valid_negative(query_record, candidate_record, params):
                negative_candidates.append(int(candidate_idx))

        seen_pos: set[int] = set()
        pos_slot = 0
        for idx in positive_candidates:
            if idx in seen_pos:
                continue
            seen_pos.add(idx)
            positives[row_idx, pos_slot] = idx
            positive_mask[row_idx, pos_slot] = True
            pos_slot += 1
            if pos_slot >= params.positives_per_query:
                break

        seen_neg: set[int] = set()
        neg_slot = 0
        for idx in negative_candidates:
            if idx in seen_neg or idx in seen_pos:
                continue
            seen_neg.add(idx)
            negatives[row_idx, neg_slot] = idx
            neg_slot += 1
            if neg_slot >= params.negatives_per_query:
                break

        if neg_slot < params.negatives_per_query:
            fallback_pool = [
                int(i)
                for i in all_indices
                if int(i) not in seen_pos
                and int(i) not in seen_neg
                and _valid_negative(query_record, records[int(i)], params)
            ]
            # If strict pool is empty, allow overlap with selected positives
            # rather than emitting invalid default indices.
            if not fallback_pool:
                fallback_pool = [
                    int(i)
                    for i in all_indices
                    if int(i) not in seen_neg
                    and _valid_negative(query_record, records[int(i)], params)
                ]
            if fallback_pool:
                needed = params.negatives_per_query - neg_slot
                sampled = rng.choice(
                    fallback_pool,
                    size=needed,
                    replace=len(fallback_pool) < needed,
                )
                negatives[row_idx, neg_slot:params.negatives_per_query] = sampled.astype(np.int64)

    return positives, positive_mask, negatives
