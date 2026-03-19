"""Export helpers for contextualized track embeddings and FAISS artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import torch

from ..search.faiss_index import (
    IndexType,
    build_clip_artifact_from_vectors,
    save_artifact,
)
from ..search.search import ClipLocation
from .context_trainer import TrackSegments, collate_track_batch
from .contextualizer import SegmentTransformer


def load_contextualizer_from_state(
    state_dict: Mapping[str, torch.Tensor],
    *,
    input_dim: int,
    context_dim: int,
    num_layers: int = 2,
    num_heads: int = 4,
    ff_dim: int = 512,
    max_segments: int = 512,
    dropout: float = 0.1,
    pool_mode: str = "mean",
) -> SegmentTransformer:
    """Reconstruct a SegmentTransformer from a checkpoint state dict.

    Args:
        state_dict: Trained model weights (from ContextTrainResult).
        **model_kwargs: Architecture parameters (input_dim, context_dim, etc.).

    Returns:
        Loaded SegmentTransformer in eval mode on CPU.
    """
    cpu_state = {name: tensor.detach().cpu() for name, tensor in state_dict.items()}
    model: SegmentTransformer = SegmentTransformer(
        input_dim=input_dim,
        context_dim=context_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_segments=max_segments,
        dropout=dropout,
        pool_mode=pool_mode,
    )
    model = cast(SegmentTransformer, model.cpu())
    model.load_state_dict(cpu_state)
    model.eval()
    return model


def contextualize_tracks(
    model: SegmentTransformer,
    tracks: list[TrackSegments],
    batch_size: int = 8,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run inference over all tracks producing global and local vectors.

    Args:
        model: Trained SegmentTransformer in eval mode.
        tracks: Pre-loaded track segments.
        batch_size: Inference batch size.

    Returns:
        global_vectors: [N, context_dim] track-level pooled vectors.
        local_matrices: List of [T_i, context_dim] per-track local vectors.
    """
    device = next(model.parameters()).device
    all_globals: list[torch.Tensor] = []
    all_locals: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(tracks), batch_size):
            end = min(start + batch_size, len(tracks))
            batch_indices = np.arange(start, end)
            segments, lengths = collate_track_batch(tracks, batch_indices, device)
            global_out, local_out = model(segments, lengths)

            all_globals.append(global_out.cpu())

            # Extract unpadded local vectors for each track in batch
            for i, _ in enumerate(batch_indices):
                t = int(lengths[i].item())
                local_np = local_out[i, :t, :].cpu().numpy().astype(np.float32)
                all_locals.append(local_np)

    global_vectors = torch.cat(all_globals, dim=0).numpy().astype(np.float32)
    return global_vectors, all_locals


def save_contextualized_embeddings(
    global_vectors: np.ndarray,
    local_matrices: list[np.ndarray],
    track_ids: list[str],
    output_dir: str | Path,
) -> Path:
    """Save contextualized embeddings to disk.

    Layout:
        output_dir/global/{track_id}.npy   # [context_dim] per track
        output_dir/local/{track_id}.npy    # [T, context_dim] per track

    Args:
        global_vectors: [N, context_dim] track-level vectors.
        local_matrices: List of [T_i, context_dim] per-track local vectors.
        track_ids: Track identifiers aligned with vectors.
        output_dir: Root output directory.

    Returns:
        Output directory path.
    """
    if len(track_ids) != len(local_matrices):
        raise ValueError(
            f"track_ids length ({len(track_ids)}) must match "
            f"local_matrices length ({len(local_matrices)})"
        )
    if global_vectors.shape[0] != len(track_ids):
        raise ValueError(
            f"global_vectors rows ({global_vectors.shape[0]}) must match "
            f"track_ids length ({len(track_ids)})"
        )

    out = Path(output_dir)
    global_dir = out / "global"
    local_dir = out / "local"
    global_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    for i, track_id in enumerate(track_ids):
        np.save(global_dir / f"{track_id}.npy", global_vectors[i])
        np.save(local_dir / f"{track_id}.npy", local_matrices[i])

    return out


def export_contextualizer_track_artifact(
    *,
    global_vectors: np.ndarray,
    track_ids: list[str],
    artifact_root: str | Path,
    artifact_name: str = "contextualizer_track_index",
    dataset: str = "unknown",
    layer: int | None = None,
    index_type: IndexType = "flatip",
    model_name: str = "m-a-p/MERT-v1-95M+contextualizer",
    nlist: int = 1024,
    feature_source_dir: str = "<contextualizer>",
    created_stamp: str | None = None,
) -> Path:
    """Build and save a FAISS track-level artifact from global vectors.

    Uses the existing clip artifact infrastructure with one clip location per
    track (segment_idx=0, covering the full track).

    Args:
        global_vectors: [N, context_dim] track-level vectors.
        track_ids: Track identifiers aligned with vectors.
        artifact_root: Output root directory.
        artifact_name: Logical artifact name.
        dataset: Dataset identifier for manifest.
        layer: Layer annotation for manifest.
        index_type: FAISS index type.
        model_name: Model name for manifest.
        nlist: IVF centroid count.
        feature_source_dir: Source annotation.
        created_stamp: Optional deterministic timestamp.

    Returns:
        Artifact directory path.
    """
    if global_vectors.shape[0] != len(track_ids):
        raise ValueError(
            f"global_vectors rows ({global_vectors.shape[0]}) must match "
            f"track_ids length ({len(track_ids)})"
        )

    # Create one ClipLocation per track for the artifact metadata
    clip_locations = [
        ClipLocation(
            track_id=tid,
            segment_idx=0,
            start_time=0.0,
            end_time=0.0,
        )
        for tid in track_ids
    ]

    artifact = build_clip_artifact_from_vectors(
        dataset=dataset,
        vectors=global_vectors,
        clip_locations=clip_locations,
        artifact_name=artifact_name,
        layer=layer,
        index_type=index_type,
        model_name=model_name,
        nlist=nlist,
        feature_source_dir=feature_source_dir,
    )

    artifact_dir = save_artifact(
        artifact,
        artifact_root=artifact_root,
        created_stamp=created_stamp,
    )
    return cast(Path, artifact_dir)
