"""Deterministic retrieval regression checks for tiny fixture data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mess.search.search import find_similar, load_features, load_segment_features

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


def test_track_level_golden_ranking_and_score_contract(tmp_path: Path) -> None:
    fixture = _load_fixture()
    segments_dir = tmp_path / "segments"
    _write_segment_embeddings(segments_dir, fixture)

    features, track_names = load_features(str(segments_dir), layer=0)
    results = find_similar("bach_a", features, track_names, k=4)

    result_ids = [track_id for track_id, _ in results]
    result_scores = [score for _, score in results]

    assert result_ids == ["bach_b", "chopin_b", "chopin_a", "noise_a"]
    assert result_scores == sorted(result_scores, reverse=True)
    assert all(np.isfinite(score) for score in result_scores)
    assert all(-1.0 <= score <= 1.0 + 1e-6 for score in result_scores)


def test_repeat_search_is_stable_for_topk_and_scores(tmp_path: Path) -> None:
    fixture = _load_fixture()
    segments_dir = tmp_path / "segments"
    _write_segment_embeddings(segments_dir, fixture)

    features_1, track_names_1 = load_features(str(segments_dir), layer=0)
    features_2, track_names_2 = load_features(str(segments_dir), layer=0)
    results_1 = find_similar("chopin_a", features_1, track_names_1, k=4)
    results_2 = find_similar("chopin_a", features_2, track_names_2, k=4)

    ids_1 = [track_id for track_id, _ in results_1]
    ids_2 = [track_id for track_id, _ in results_2]
    overlap_at_4 = len(set(ids_1) & set(ids_2)) / 4.0

    assert ids_1 == ids_2
    assert overlap_at_4 == 1.0
    assert [score for _, score in results_1] == pytest.approx(
        [score for _, score in results_2],
        abs=1e-8,
    )


def test_segment_feature_schema_from_fixture(tmp_path: Path) -> None:
    fixture = _load_fixture()
    segments_dir = tmp_path / "segments"
    _write_segment_embeddings(segments_dir, fixture)

    vectors, clip_locations = load_segment_features(str(segments_dir), layer=0)

    assert vectors.shape == (15, 768)
    assert len(clip_locations) == 15

    starts_by_track: dict[str, list[float]] = {}
    segment_idx_by_track: dict[str, list[int]] = {}
    for location in clip_locations:
        starts_by_track.setdefault(location.track_id, []).append(location.start_time)
        segment_idx_by_track.setdefault(location.track_id, []).append(location.segment_idx)

    for starts in starts_by_track.values():
        assert starts == sorted(starts)
        assert starts == pytest.approx([0.0, 2.5, 5.0])
    for indices in segment_idx_by_track.values():
        assert indices == [0, 1, 2]
