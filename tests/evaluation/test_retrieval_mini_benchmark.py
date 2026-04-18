"""Tiny retrieval benchmark regression checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mess.evaluation.metrics import evaluate_rankings
from mess.search.search import find_similar, load_features

pytestmark = [pytest.mark.integration, pytest.mark.regression]

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "retrieval_tiny.json"


def _load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _write_segment_embeddings(segments_dir: Path, fixture: dict[str, Any]) -> None:
    segments_dir.mkdir(parents=True, exist_ok=True)
    for track in fixture["tracks"]:
        vectors = np.asarray(track["segments"], dtype=np.float32)
        n_segments, dim = vectors.shape
        payload = np.zeros((n_segments, 13, 768), dtype=np.float32)
        payload[:, 0, :dim] = vectors
        np.save(segments_dir / f"{track['track_id']}.npy", payload)


def test_tiny_benchmark_retrieval_quality_guardrail(tmp_path: Path) -> None:
    fixture = _load_fixture()
    segments_dir = tmp_path / "segments"
    _write_segment_embeddings(segments_dir, fixture)

    features, track_names = load_features(str(segments_dir), layer=0)

    rankings: list[list[str]] = []
    relevant_sets: list[set[str]] = []
    positive_tracks: dict[str, list[str]] = fixture["positive_tracks"]

    for query_track, positives in positive_tracks.items():
        results = find_similar(query_track, features, track_names, k=len(track_names) - 1)
        ranked_tracks = [track_id for track_id, _ in results]
        rankings.append(ranked_tracks)
        relevant_sets.append(set(positives))

        assert ranked_tracks[0] in positives
        assert "noise_a" not in ranked_tracks[:2]

    report = evaluate_rankings(rankings, relevant_sets, k_values=[1, 2])

    assert report["query_count"] == len(positive_tracks)
    assert report["mrr"] >= 0.99
    assert report["recall"]["@1"] >= 0.99
    assert report["recall"]["@2"] >= 0.99
