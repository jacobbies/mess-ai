"""Smoke test for scripts/evaluate_retrieval.py JSON report output."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mess.datasets.clip_index import ClipIndex, ClipRecord
from mess.training.trainer import ProjectionHead

pytestmark = pytest.mark.integration

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_retrieval.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("evaluate_retrieval_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_segment_embeddings(path: Path, layer0_vectors: np.ndarray) -> None:
    segments = np.zeros((layer0_vectors.shape[0], 13, layer0_vectors.shape[1]), dtype=np.float32)
    segments[:, 0, :] = layer0_vectors
    np.save(path, segments)


def _save_identity_checkpoint(path: Path, dim: int) -> None:
    model = ProjectionHead(input_dim=dim, output_dim=dim, hidden_dim=None)
    linear = model.network
    assert isinstance(linear, torch.nn.Linear)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(dim))
        linear.bias.zero_()

    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    payload = {
        "input_dim": dim,
        "output_dim": dim,
        "config": {"hidden_dim": None},
        "online_state_dict": state,
        "target_state_dict": state,
    }
    torch.save(payload, path)


def test_evaluate_retrieval_script_emits_report(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()

    track_a = np.array([[1.0, 0.0, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0]], dtype=np.float32)
    track_b = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.95, 0.05, 0.0]], dtype=np.float32)
    _write_segment_embeddings(emb_dir / "track_a.npy", track_a)
    _write_segment_embeddings(emb_dir / "track_b.npy", track_b)

    records = [
        ClipRecord(
            clip_id="smd:track_a:00000",
            dataset_id="smd",
            recording_id="rec_a",
            track_id="track_a",
            segment_idx=0,
            start_sec=0.0,
            end_sec=5.0,
            split="train",
            embedding_path=str(emb_dir / "track_a.npy"),
        ),
        ClipRecord(
            clip_id="smd:track_a:00001",
            dataset_id="smd",
            recording_id="rec_a",
            track_id="track_a",
            segment_idx=1,
            start_sec=2.5,
            end_sec=7.5,
            split="train",
            embedding_path=str(emb_dir / "track_a.npy"),
        ),
        ClipRecord(
            clip_id="smd:track_b:00000",
            dataset_id="smd",
            recording_id="rec_b",
            track_id="track_b",
            segment_idx=0,
            start_sec=0.0,
            end_sec=5.0,
            split="train",
            embedding_path=str(emb_dir / "track_b.npy"),
        ),
        ClipRecord(
            clip_id="smd:track_b:00001",
            dataset_id="smd",
            recording_id="rec_b",
            track_id="track_b",
            segment_idx=1,
            start_sec=2.5,
            end_sec=7.5,
            split="train",
            embedding_path=str(emb_dir / "track_b.npy"),
        ),
    ]
    clip_index_path = tmp_path / "clip_index.csv"
    ClipIndex(records).to_csv(clip_index_path)

    checkpoint_path = tmp_path / "projection.pt"
    _save_identity_checkpoint(checkpoint_path, dim=4)

    report_path = tmp_path / "report.json"
    module = _load_script_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_retrieval.py",
            "--clip-index",
            str(clip_index_path),
            "--report-json",
            str(report_path),
            "--layer",
            "0",
            "--k-values",
            "1,2",
            "--max-queries",
            "0",
            "--seed",
            "123",
            "--protocol",
            "both",
            "--projection-checkpoint",
            str(checkpoint_path),
        ],
    )

    assert module.main() == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["dataset_summary"]["num_clips"] == 4
    assert set(report["systems"]) == {"baseline", "projection"}
    baseline = report["systems"]["baseline"]["protocols"]
    projection = report["systems"]["projection"]["protocols"]
    assert set(baseline) == {"clip_to_clip", "clip_to_track"}
    assert set(projection) == {"clip_to_clip", "clip_to_track"}

    assert baseline["clip_to_clip"]["query_count"] == 4
    assert baseline["clip_to_track"]["query_count"] == 4
    assert report["systems"]["projection"]["metadata"]["encoder"] == "online"
