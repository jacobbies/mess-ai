"""Unit tests for two-stage search reranker."""

from __future__ import annotations

import numpy as np
import pytest

from mess.search.reranker import (
    RerankResult,
    load_global_vectors,
    load_local_matrices,
    two_stage_search,
)

pytestmark = pytest.mark.unit


def _save_synthetic_data(
    tmp_path,
    n_tracks: int = 8,
    context_dim: int = 16,
    min_segments: int = 3,
    max_segments: int = 6,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Create synthetic global/local embeddings on disk."""
    rng = np.random.default_rng(seed)

    global_dir = tmp_path / "global"
    local_dir = tmp_path / "local"
    global_dir.mkdir()
    local_dir.mkdir()

    track_ids: list[str] = []
    all_globals: list[np.ndarray] = []

    for i in range(n_tracks):
        tid = f"track_{i:03d}"
        track_ids.append(tid)

        # Global vector
        g = rng.standard_normal(context_dim).astype(np.float32)
        g /= np.linalg.norm(g) + 1e-12
        np.save(global_dir / f"{tid}.npy", g)
        all_globals.append(g)

        # Local matrix
        n_seg = rng.integers(min_segments, max_segments + 1)
        local = rng.standard_normal((n_seg, context_dim)).astype(np.float32)
        norms = np.linalg.norm(local, axis=1, keepdims=True)
        local /= np.clip(norms, 1e-12, None)
        np.save(local_dir / f"{tid}.npy", local)

    return track_ids, np.stack(all_globals)


class TestLoadGlobalVectors:
    def test_loads_correct_shapes(self, tmp_path) -> None:
        track_ids, _ = _save_synthetic_data(tmp_path, n_tracks=5, context_dim=8)
        vectors, loaded_ids = load_global_vectors(tmp_path / "global")

        assert vectors.shape == (5, 8)
        assert loaded_ids == sorted(track_ids)

    def test_raises_on_empty_dir(self, tmp_path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            load_global_vectors(empty)


class TestLoadLocalMatrices:
    def test_loads_requested_tracks(self, tmp_path) -> None:
        track_ids, _ = _save_synthetic_data(tmp_path, n_tracks=4, context_dim=8)
        matrices = load_local_matrices(tmp_path / "local", track_ids[:2])

        assert len(matrices) == 2
        for m in matrices:
            assert m.ndim == 2
            assert m.shape[1] == 8


class TestTwoStageSearch:
    def test_returns_correct_count(self, tmp_path, monkeypatch) -> None:
        """Should return k results, excluding the query track."""
        _mock_faiss(monkeypatch)
        track_ids, _ = _save_synthetic_data(tmp_path, n_tracks=10, context_dim=16)

        results = two_stage_search(
            query_track=track_ids[0],
            global_dir=tmp_path / "global",
            local_dir=tmp_path / "local",
            k=5,
            first_pass_k=9,
        )

        assert len(results) == 5
        assert all(isinstance(r, RerankResult) for r in results)

    def test_excludes_query_track(self, tmp_path, monkeypatch) -> None:
        _mock_faiss(monkeypatch)
        track_ids, _ = _save_synthetic_data(tmp_path, n_tracks=6, context_dim=16)

        results = two_stage_search(
            query_track=track_ids[0],
            global_dir=tmp_path / "global",
            local_dir=tmp_path / "local",
            k=5,
            first_pass_k=5,
        )

        result_track_ids = {r.track_id for r in results}
        assert track_ids[0] not in result_track_ids

    def test_sorted_by_combined_score(self, tmp_path, monkeypatch) -> None:
        _mock_faiss(monkeypatch)
        track_ids, _ = _save_synthetic_data(tmp_path, n_tracks=8, context_dim=16)

        results = two_stage_search(
            query_track=track_ids[0],
            global_dir=tmp_path / "global",
            local_dir=tmp_path / "local",
            k=5,
            first_pass_k=7,
        )

        scores = [r.combined_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_weight_zero_uses_global_only(self, tmp_path, monkeypatch) -> None:
        _mock_faiss(monkeypatch)
        track_ids, _ = _save_synthetic_data(tmp_path, n_tracks=6, context_dim=16)

        results = two_stage_search(
            query_track=track_ids[0],
            global_dir=tmp_path / "global",
            local_dir=tmp_path / "local",
            k=3,
            first_pass_k=5,
            rerank_weight=0.0,
        )

        for r in results:
            assert abs(r.combined_score - r.global_score) < 1e-6

    def test_missing_query_raises(self, tmp_path, monkeypatch) -> None:
        _mock_faiss(monkeypatch)
        _save_synthetic_data(tmp_path, n_tracks=4, context_dim=16)

        with pytest.raises(KeyError, match="not_a_track"):
            two_stage_search(
                query_track="not_a_track",
                global_dir=tmp_path / "global",
                local_dir=tmp_path / "local",
                k=2,
                first_pass_k=3,
            )

    def test_invalid_k(self, tmp_path, monkeypatch) -> None:
        _mock_faiss(monkeypatch)
        _save_synthetic_data(tmp_path, n_tracks=4, context_dim=16)

        with pytest.raises(ValueError, match="k must be > 0"):
            two_stage_search(
                query_track="track_000",
                global_dir=tmp_path / "global",
                local_dir=tmp_path / "local",
                k=0,
                first_pass_k=3,
            )


# ---------------------------------------------------------------------------
# Fake FAISS for testing
# ---------------------------------------------------------------------------


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


def _mock_faiss(monkeypatch) -> None:
    """Monkeypatch faiss in the reranker module."""
    import mess.search.reranker as reranker_mod
    import mess.search.search as search_mod

    monkeypatch.setattr(search_mod, "_require_faiss", lambda: _FakeFaiss)
    # The reranker imports _require_faiss from search, so we also need to
    # ensure the reranker's reference picks up the mock.
    monkeypatch.setattr(reranker_mod, "_require_faiss", lambda: _FakeFaiss)
