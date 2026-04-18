"""Workflow/system tests for the compact retrieval pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mess.datasets.clip_index import ClipIndex
from mess.search.faiss_index import (
    ArtifactValidationError,
    build_clip_artifact,
    load_artifact,
    save_artifact,
)
from mess.search.search import search_by_clip
from scripts.build_clip_index import main as build_clip_index_main

pytestmark = [pytest.mark.workflow, pytest.mark.integration]

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


def _write_recording_manifest(path: Path, fixture: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_id", "recording_id", "work_id"])
        writer.writeheader()
        for track in fixture["tracks"]:
            writer.writerow(
                {
                    "track_id": track["track_id"],
                    "recording_id": track["recording_id"],
                    "work_id": track["work_id"],
                }
            )


def _build_clip_index(
    *,
    fixture: dict[str, Any],
    segments_dir: Path,
    clip_index_path: Path,
    manifest_path: Path,
) -> int:
    return build_clip_index_main(
        dataset_id=str(fixture["dataset_id"]),
        segments_dir=segments_dir,
        output_path=clip_index_path,
        segment_duration=float(fixture["segment_duration"]),
        overlap_ratio=float(fixture["overlap_ratio"]),
        assign_splits=True,
        seed=13,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        manifest_path=manifest_path,
        manifest_track_col="track_id",
        manifest_recording_col="recording_id",
        manifest_work_col="work_id",
    )


def test_clip_retrieval_workflow_end_to_end_and_rerun_stability(tmp_path: Path) -> None:
    fixture = _load_fixture()
    dataset_id = str(fixture["dataset_id"])

    segments_dir = tmp_path / "embeddings" / f"{dataset_id}-emb" / "segments"
    clip_index_path = tmp_path / "metadata" / "clip_index.csv"
    manifest_path = tmp_path / "metadata" / "recording_manifest.csv"
    artifact_root = tmp_path / "indices"

    _write_segment_embeddings(segments_dir, fixture)
    _write_recording_manifest(manifest_path, fixture)

    first_exit = _build_clip_index(
        fixture=fixture,
        segments_dir=segments_dir,
        clip_index_path=clip_index_path,
        manifest_path=manifest_path,
    )
    assert first_exit == 0

    first_csv_bytes = clip_index_path.read_bytes()
    clip_index = ClipIndex.from_csv(clip_index_path)
    assert len(clip_index) == 15
    assert all(record.dataset_id == dataset_id for record in clip_index)
    assert all(record.work_id for record in clip_index)

    splits_by_recording: dict[str, set[str]] = {}
    for record in clip_index:
        splits_by_recording.setdefault(record.recording_id, set()).add(record.split)
    assert all(len(splits) == 1 for splits in splits_by_recording.values())

    second_exit = _build_clip_index(
        fixture=fixture,
        segments_dir=segments_dir,
        clip_index_path=clip_index_path,
        manifest_path=manifest_path,
    )
    assert second_exit == 0
    assert clip_index_path.read_bytes() == first_csv_bytes

    artifact = build_clip_artifact(
        dataset=dataset_id,
        clip_index=clip_index,
        layer=0,
    )
    first_artifact_dir = save_artifact(
        artifact,
        artifact_root=artifact_root,
        created_stamp="20260101T000000Z",
    )
    second_artifact_dir = save_artifact(
        artifact,
        artifact_root=artifact_root,
        created_stamp="20260101T000100Z",
    )

    for artifact_dir in (first_artifact_dir, second_artifact_dir):
        assert (artifact_dir / "index.faiss").exists()
        assert (artifact_dir / "manifest.json").exists()
        assert (artifact_dir / "checksums.json").exists()
        assert (artifact_dir / "clip_records.json.gz").exists()
        assert (artifact_dir / "vectors.npy").exists()

    loaded_first = load_artifact(first_artifact_dir)
    loaded_second = load_artifact(second_artifact_dir)
    assert loaded_first.manifest.ntotal == len(clip_index)
    assert loaded_first.manifest.dimension == 768
    assert loaded_first.clip_records is not None
    assert len(loaded_first.clip_records) == len(clip_index)
    assert loaded_first.vectors is not None
    assert loaded_first.vectors.shape == (len(clip_index), 768)

    first_results = search_by_clip(
        artifact=loaded_first,
        track_id="bach_a",
        start_sec=0.0,
        k=5,
        dedupe_window_seconds=0.0,
    )
    second_results = search_by_clip(
        artifact=loaded_second,
        track_id="bach_a",
        start_sec=0.0,
        k=5,
        dedupe_window_seconds=0.0,
    )

    assert len(first_results) == 5
    assert first_results[0].track_id == "bach_b"
    assert all(np.isfinite(result.similarity) for result in first_results)
    assert [result.similarity for result in first_results] == sorted(
        [result.similarity for result in first_results],
        reverse=True,
    )

    assert [result.clip_id for result in first_results] == [
        result.clip_id for result in second_results
    ]
    assert [result.similarity for result in first_results] == pytest.approx(
        [result.similarity for result in second_results],
        abs=1e-8,
    )


def test_workflow_detects_corrupt_artifact_payload(tmp_path: Path) -> None:
    fixture = _load_fixture()
    dataset_id = str(fixture["dataset_id"])
    segments_dir = tmp_path / "embeddings" / f"{dataset_id}-emb" / "segments"
    clip_index_path = tmp_path / "metadata" / "clip_index.csv"
    manifest_path = tmp_path / "metadata" / "recording_manifest.csv"

    _write_segment_embeddings(segments_dir, fixture)
    _write_recording_manifest(manifest_path, fixture)
    assert (
        _build_clip_index(
            fixture=fixture,
            segments_dir=segments_dir,
            clip_index_path=clip_index_path,
            manifest_path=manifest_path,
        )
        == 0
    )

    artifact = build_clip_artifact(
        dataset=dataset_id,
        clip_index=ClipIndex.from_csv(clip_index_path),
        layer=0,
    )
    artifact_dir = save_artifact(
        artifact,
        artifact_root=tmp_path / "indices",
        created_stamp="20260101T000000Z",
    )

    vectors_path = artifact_dir / "vectors.npy"
    original = vectors_path.read_bytes()
    vectors_path.write_bytes(original[:-32])

    with pytest.raises(ArtifactValidationError, match="File size mismatch"):
        load_artifact(artifact_dir)


def test_workflow_fails_on_bad_segments_then_recovers_after_fix(tmp_path: Path) -> None:
    fixture = _load_fixture()
    dataset_id = str(fixture["dataset_id"])
    segments_dir = tmp_path / "embeddings" / f"{dataset_id}-emb" / "segments"
    clip_index_path = tmp_path / "metadata" / "clip_index.csv"
    manifest_path = tmp_path / "metadata" / "recording_manifest.csv"

    _write_segment_embeddings(segments_dir, fixture)
    _write_recording_manifest(manifest_path, fixture)

    # Simulate partial-corruption: one track has an invalid 2D array instead of [segments, 13, 768].
    np.save(segments_dir / "bach_a.npy", np.zeros((13, 768), dtype=np.float32))

    with pytest.raises(ValueError, match="Expected segment embeddings shape"):
        _build_clip_index(
            fixture=fixture,
            segments_dir=segments_dir,
            clip_index_path=clip_index_path,
            manifest_path=manifest_path,
        )
    assert not clip_index_path.exists()

    _write_segment_embeddings(segments_dir, fixture)
    assert (
        _build_clip_index(
            fixture=fixture,
            segments_dir=segments_dir,
            clip_index_path=clip_index_path,
            manifest_path=manifest_path,
        )
        == 0
    )
    assert len(ClipIndex.from_csv(clip_index_path)) == 15
