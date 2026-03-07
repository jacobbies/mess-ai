"""Tests for FAISS index_factory artifact build paths."""

from __future__ import annotations

import numpy as np
import pytest

from mess.search.faiss_index import (
    _build_faiss_index,
    build_track_artifact,
    load_artifact,
    save_artifact,
)

pytestmark = pytest.mark.integration


def _make_vectors(n_rows: int = 320, dim: int = 32, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n_rows, dim)).astype(np.float32)


@pytest.mark.parametrize(
    "factory_string",
    ["HNSW32,Flat", "IVF16,PQ4x4", "OPQ4,IVF16,PQ4x4"],
)
def test_build_index_factory_train_add_search(factory_string):
    vectors = _make_vectors()
    index = _build_faiss_index(
        vectors,
        index_type="factory",
        factory_string=factory_string,
    )
    assert int(index.ntotal) == vectors.shape[0]

    query = vectors[:2].astype(np.float32, copy=True)
    distances, indices = index.search(query, 5)
    assert distances.shape == (2, 5)
    assert indices.shape == (2, 5)


def test_factory_manifest_persists_factory_string_and_training_size(tmp_path):
    vectors = _make_vectors(n_rows=128, dim=32, seed=42)
    feature_dir = tmp_path / "aggregated"
    feature_dir.mkdir()
    for idx, vector in enumerate(vectors):
        np.save(feature_dir / f"track_{idx:04d}.npy", vector)

    artifact = build_track_artifact(
        dataset="smd",
        features_dir=feature_dir,
        index_type="factory",
        factory_string="IVF16,PQ4x4",
    )
    assert artifact.manifest.index_type == "factory"
    assert artifact.manifest.factory_string == "IVF16,PQ4x4"
    assert artifact.manifest.train_size == 128

    artifact_dir = save_artifact(
        artifact,
        artifact_root=tmp_path / "indices",
        created_stamp="20260101T000000Z",
    )
    loaded = load_artifact(artifact_dir)
    assert loaded.manifest.index_type == "factory"
    assert loaded.manifest.factory_string == "IVF16,PQ4x4"
    assert loaded.manifest.train_size == 128


def test_factory_index_requires_factory_string():
    vectors = _make_vectors(n_rows=32, dim=16, seed=7)
    with pytest.raises(ValueError, match="factory_string is required"):
        _build_faiss_index(vectors, index_type="factory")
