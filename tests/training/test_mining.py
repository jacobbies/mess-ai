"""Tests for neighbor mining guardrails."""

from __future__ import annotations

import numpy as np
import pytest

from mess.datasets.clip_index import ClipRecord
from mess.training.mining import MiningParams, mine_batch_indices

pytestmark = pytest.mark.unit


def _records() -> list[ClipRecord]:
    return [
        ClipRecord(
            clip_id="c0",
            dataset_id="smd",
            recording_id="rec_a",
            track_id="track_a",
            segment_idx=0,
            start_sec=0.0,
            end_sec=5.0,
            split="train",
            embedding_path="/tmp/a.npy",
        ),
        ClipRecord(
            clip_id="c1",
            dataset_id="smd",
            recording_id="rec_a",
            track_id="track_a",
            segment_idx=1,
            start_sec=2.5,
            end_sec=7.5,
            split="train",
            embedding_path="/tmp/a.npy",
        ),
        ClipRecord(
            clip_id="c2",
            dataset_id="smd",
            recording_id="rec_a",
            track_id="track_b",
            segment_idx=0,
            start_sec=0.0,
            end_sec=5.0,
            split="train",
            embedding_path="/tmp/b.npy",
        ),
        ClipRecord(
            clip_id="c3",
            dataset_id="smd",
            recording_id="rec_b",
            track_id="track_c",
            segment_idx=0,
            start_sec=0.0,
            end_sec=5.0,
            split="train",
            embedding_path="/tmp/c.npy",
        ),
        ClipRecord(
            clip_id="c4",
            dataset_id="smd",
            recording_id="rec_c",
            track_id="track_d",
            segment_idx=0,
            start_sec=0.0,
            end_sec=5.0,
            split="train",
            embedding_path="/tmp/d.npy",
        ),
    ]


def test_mining_excludes_self_and_nearby_same_track() -> None:
    records = _records()
    query_indices = np.array([0], dtype=np.int64)
    neighbor_indices = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)

    params = MiningParams(
        positives_per_query=2,
        negatives_per_query=2,
        min_time_separation_sec=5.0,
    )

    pos_idx, pos_mask, _ = mine_batch_indices(
        records=records,
        query_indices=query_indices,
        neighbor_indices=neighbor_indices,
        params=params,
        rng=np.random.default_rng(0),
    )

    chosen_positives = {
        int(idx)
        for idx, valid in zip(pos_idx[0], pos_mask[0], strict=True)
        if valid
    }
    assert 0 not in chosen_positives
    assert 1 not in chosen_positives


def test_mining_cross_recording_positive_and_negative_rules() -> None:
    records = _records()
    query_indices = np.array([0], dtype=np.int64)
    neighbor_indices = np.array([[0, 2, 3, 4, 1]], dtype=np.int64)

    params = MiningParams(
        positives_per_query=2,
        negatives_per_query=2,
        min_time_separation_sec=0.1,
        require_cross_recording_positive=True,
        exclude_same_recording_negative=True,
    )

    pos_idx, pos_mask, neg_idx = mine_batch_indices(
        records=records,
        query_indices=query_indices,
        neighbor_indices=neighbor_indices,
        params=params,
        rng=np.random.default_rng(1),
    )

    selected_positives = [
        records[int(idx)] for idx, valid in zip(pos_idx[0], pos_mask[0], strict=True) if valid
    ]
    assert selected_positives
    assert all(row.recording_id != records[0].recording_id for row in selected_positives)

    selected_negatives = [records[int(idx)] for idx in neg_idx[0]]
    assert all(row.recording_id != records[0].recording_id for row in selected_negatives)
