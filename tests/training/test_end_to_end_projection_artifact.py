"""Integration test for training-to-artifact projection export."""

from __future__ import annotations

import numpy as np
import pytest

import mess.search.faiss_index as artifact_index
import mess.training.index as training_index
from mess.datasets.clip_index import ClipRecord
from mess.search.faiss_index import load_artifact
from mess.training import (
    RetrievalSSLConfig,
    export_projection_clip_artifact,
    project_vectors_with_head,
    train_projection_head,
)

pytestmark = pytest.mark.integration


class _FakeIndexFlatIP:
    def __init__(self, dim: int) -> None:
        self.d = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = np.asarray(vectors, dtype=np.float32).copy()

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sims = np.asarray(queries, dtype=np.float32) @ self._vectors.T
        topk_idx = np.argsort(-sims, axis=1)[:, :k]
        topk_scores = np.take_along_axis(sims, topk_idx, axis=1)
        return topk_scores.astype(np.float32), topk_idx.astype(np.int64)


class _FakeFaiss:
    IndexFlatIP = _FakeIndexFlatIP

    @staticmethod
    def normalize_L2(vectors: np.ndarray) -> None:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        vectors /= norms


def _synthetic_records(n: int) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    for idx in range(n):
        segment_idx = idx % 12
        start_sec = float(segment_idx * 2.5)
        records.append(
            ClipRecord(
                clip_id=f"smd:track_{idx % 6}:{segment_idx:05d}",
                dataset_id="smd",
                recording_id=f"rec_{idx % 4}",
                track_id=f"track_{idx % 6}",
                segment_idx=segment_idx,
                start_sec=start_sec,
                end_sec=start_sec + 5.0,
                split="train",
                embedding_path=f"/tmp/fake_track_{idx % 6}.npy",
            )
        )
    return records


def test_train_then_export_projection_clip_artifact(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_index, "_require_faiss", lambda: _FakeFaiss)

    rng = np.random.default_rng(11)
    base_vectors = rng.standard_normal((24, 48)).astype(np.float32)
    records = _synthetic_records(24)

    config = RetrievalSSLConfig(
        num_steps=6,
        batch_size=8,
        projection_dim=16,
        hidden_dim=32,
        search_k=12,
        positives_per_query=2,
        negatives_per_query=4,
        min_time_separation_sec=1.0,
        refresh_every=3,
        seed=7,
        device="cpu",
    )
    result = train_projection_head(base_vectors, records, config)

    artifact_dir = export_projection_clip_artifact(
        base_vectors=base_vectors,
        records=records,
        state_dict=result.online_state_dict,
        input_dim=result.input_dim,
        projection_dim=result.output_dim,
        hidden_dim=config.hidden_dim,
        artifact_root=tmp_path / "indices",
        artifact_name="projection_clip_index",
        feature_source_dir=str(tmp_path / "clip_index.csv"),
        layer=0,
        created_stamp="20260101T000000Z",
    )

    assert artifact_dir.name == "20260101T000000Z"
    assert (artifact_dir / "manifest.json").exists()
    assert (artifact_dir / "checksums.json").exists()

    loaded = load_artifact(artifact_dir)
    assert loaded.manifest.kind == "clip"
    assert loaded.manifest.dataset == "smd"
    assert loaded.manifest.dimension == result.output_dim
    assert loaded.manifest.ntotal == len(records)
    assert loaded.manifest.feature_source_dir == str(tmp_path / "clip_index.csv")
    assert loaded.clip_locations is not None
    assert len(loaded.clip_locations) == len(records)

    first_loc = loaded.clip_locations[0]
    assert first_loc.track_id == records[0].track_id
    assert first_loc.segment_idx == records[0].segment_idx
    assert first_loc.start_time == pytest.approx(records[0].start_sec)
    assert first_loc.end_time == pytest.approx(records[0].end_sec)

    projected_vectors = project_vectors_with_head(
        base_vectors=base_vectors,
        input_dim=result.input_dim,
        projection_dim=result.output_dim,
        hidden_dim=config.hidden_dim,
        state_dict=result.online_state_dict,
    )
    monkeypatch.setattr(artifact_index, "_require_faiss", lambda: _FakeFaiss)
    fake_index = _FakeIndexFlatIP(result.output_dim)
    fake_index.add(projected_vectors)
    object.__setattr__(loaded, "index", fake_index)

    scores, ids = loaded.search(projected_vectors[:1], k=5)
    assert scores.shape == (1, 5)
    assert ids.shape == (1, 5)
    assert int(ids[0, 0]) == 0
