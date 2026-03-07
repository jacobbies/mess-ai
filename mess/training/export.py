"""Projection-head export helpers for deployable FAISS artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from ..datasets.clip_index import ClipRecord
from ..search.faiss_index import (
    IndexType,
    build_clip_artifact_from_vectors,
    save_artifact,
)
from ..search.search import ClipLocation
from .trainer import ProjectionHead


def _cpu_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in state_dict.items()}


def project_vectors_with_head(
    *,
    base_vectors: np.ndarray,
    input_dim: int,
    projection_dim: int,
    hidden_dim: int | None,
    state_dict: Mapping[str, torch.Tensor],
    batch_size: int = 1024,
) -> np.ndarray:
    """Project base clip vectors using a trained projection-head state dict."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if input_dim <= 0:
        raise ValueError("input_dim must be > 0")
    if projection_dim <= 0:
        raise ValueError("projection_dim must be > 0")

    vectors = np.asarray(base_vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError(f"base_vectors must be non-empty 2D array, got {vectors.shape}")
    if vectors.shape[1] != input_dim:
        raise ValueError(f"Expected base_vectors width {input_dim}, got {vectors.shape[1]}")

    model = ProjectionHead(
        input_dim=input_dim,
        output_dim=projection_dim,
        hidden_dim=hidden_dim,
    ).cpu()
    model.load_state_dict(_cpu_state_dict(state_dict))
    model.eval()

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, vectors.shape[0], batch_size):
            end = min(start + batch_size, vectors.shape[0])
            batch = torch.from_numpy(vectors[start:end])
            projected = model(batch).cpu().numpy().astype(np.float32)
            outputs.append(projected)

    return np.vstack(outputs).astype(np.float32)


def records_to_clip_locations(records: Sequence[ClipRecord]) -> list[ClipLocation]:
    """Convert clip-index records to FAISS clip-location metadata rows."""
    return [
        ClipLocation(
            track_id=record.track_id,
            segment_idx=record.segment_idx,
            start_time=record.start_sec,
            end_time=record.end_sec,
        )
        for record in records
    ]


def _resolve_dataset(records: Sequence[ClipRecord], dataset: str | None) -> str:
    if dataset is not None:
        return dataset
    dataset_ids = sorted({record.dataset_id for record in records})
    if not dataset_ids:
        raise ValueError("Cannot infer dataset from empty records")
    if len(dataset_ids) > 1:
        raise ValueError(
            "Multiple dataset IDs found in records. Provide dataset explicitly for export."
        )
    return dataset_ids[0]


def export_projection_clip_artifact(
    *,
    base_vectors: np.ndarray,
    records: Sequence[ClipRecord],
    state_dict: Mapping[str, torch.Tensor],
    input_dim: int,
    projection_dim: int,
    hidden_dim: int | None,
    artifact_root: str | Path,
    artifact_name: str = "projection_clip_index",
    feature_source_dir: str = "<projection-head>",
    dataset: str | None = None,
    layer: int | None = None,
    index_type: IndexType = "flatip",
    model_name: str = "m-a-p/MERT-v1-95M+projection-head",
    nlist: int = 1024,
    created_stamp: str | None = None,
    batch_size: int = 1024,
) -> Path:
    """Project vectors with the trained head and persist a clip FAISS artifact."""
    if not records:
        raise ValueError("records must be non-empty for artifact export")

    projected_vectors = project_vectors_with_head(
        base_vectors=base_vectors,
        input_dim=input_dim,
        projection_dim=projection_dim,
        hidden_dim=hidden_dim,
        state_dict=state_dict,
        batch_size=batch_size,
    )

    clip_locations = records_to_clip_locations(records)
    resolved_dataset = _resolve_dataset(records, dataset)

    artifact = build_clip_artifact_from_vectors(
        dataset=resolved_dataset,
        vectors=projected_vectors,
        clip_locations=clip_locations,
        artifact_name=artifact_name,
        layer=layer,
        index_type=index_type,
        model_name=model_name,
        nlist=nlist,
        feature_source_dir=feature_source_dir,
    )
    return save_artifact(
        artifact,
        artifact_root=artifact_root,
        created_stamp=created_stamp,
    )
