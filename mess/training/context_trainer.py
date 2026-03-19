"""Training loop for segment transformer contextualizer."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch

from ..datasets.clip_index import ClipRecord
from .context_config import ContextualizerConfig
from .contextualizer import SegmentTransformer, late_interaction_score
from .index import FaissRetrievalIndex, should_refresh_index
from .losses import multi_positive_info_nce
from .trainer import _cpu_state_dict, _ema_update, _resolve_device

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class TrackSegments:
    """All segment embeddings for a single track."""

    track_id: str
    recording_id: str
    dataset_id: str
    segments: np.ndarray  # [T, input_dim]
    clip_records: list[ClipRecord]


def group_records_by_track(records: Sequence[ClipRecord]) -> dict[str, list[ClipRecord]]:
    """Group clip records by track_id, sorted by segment_idx within each track."""
    by_track: dict[str, list[ClipRecord]] = defaultdict(list)
    for record in records:
        by_track[record.track_id].append(record)
    return {
        track_id: sorted(clips, key=lambda r: r.segment_idx)
        for track_id, clips in by_track.items()
    }


def load_track_segments(
    records_by_track: dict[str, list[ClipRecord]],
    layer: int | None,
) -> list[TrackSegments]:
    """Load segment embeddings grouped by track.

    Each track's segments are loaded from the .npy files referenced by clip records.
    Uses mmap for memory efficiency and caches files to avoid reloading.

    Args:
        records_by_track: Track ID -> sorted list of ClipRecords.
        layer: MERT layer to select (0-12), or None for flattened.

    Returns:
        List of TrackSegments, one per track.
    """
    file_cache: dict[str, np.ndarray] = {}
    tracks: list[TrackSegments] = []

    for track_id, clip_records in records_by_track.items():
        if not clip_records:
            continue

        segment_vectors: list[np.ndarray] = []
        for record in clip_records:
            path = record.embedding_path
            if path not in file_cache:
                file_cache[path] = np.load(path, mmap_mode="r")
            raw = file_cache[path]

            # Handle different embedding shapes (mirrors NpySegmentEmbeddingStore)
            if raw.ndim == 2:
                # [13, dim] single segment
                vec = raw
            elif raw.ndim == 3:
                # [segments, 13, dim]
                vec = raw[record.segment_idx]
            elif raw.ndim == 4:
                # [segments, 13, time, dim] - average over time
                vec = raw[record.segment_idx].mean(axis=1)
            else:
                raise ValueError(f"Unexpected embedding shape {raw.shape} for {path}")

            # Select layer or flatten
            if layer is not None:
                vec = vec[layer]  # [dim]
            else:
                vec = vec.reshape(-1)  # [13*dim]

            segment_vectors.append(np.asarray(vec, dtype=np.float32))

        first = clip_records[0]
        tracks.append(
            TrackSegments(
                track_id=track_id,
                recording_id=first.recording_id,
                dataset_id=first.dataset_id,
                segments=np.stack(segment_vectors, axis=0),  # [T, input_dim]
                clip_records=clip_records,
            )
        )

    return tracks


def collate_track_batch(
    tracks: list[TrackSegments],
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad variable-length track segments into a batch.

    Args:
        tracks: All tracks.
        indices: Indices into `tracks` to form this batch.
        device: Target device.

    Returns:
        segments: [B, max_T, input_dim] zero-padded.
        lengths: [B] number of valid segments per track.
    """
    selected = [tracks[int(i)] for i in indices]
    max_t = max(t.segments.shape[0] for t in selected)
    input_dim = selected[0].segments.shape[1]

    batch_segments = np.zeros((len(selected), max_t, input_dim), dtype=np.float32)
    batch_lengths: np.ndarray[Any, np.dtype[np.int64]] = np.zeros(
        len(selected), dtype=np.int64
    )

    for idx, track in enumerate(selected):
        t = track.segments.shape[0]
        batch_segments[idx, :t, :] = track.segments
        batch_lengths[idx] = t

    return (
        torch.from_numpy(batch_segments).to(device),
        torch.from_numpy(batch_lengths).to(device),
    )


# ---------------------------------------------------------------------------
# Track-level mining
# ---------------------------------------------------------------------------


def _mine_track_pairs(
    tracks: list[TrackSegments],
    query_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    config: ContextualizerConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mine positive and negative track indices for each query track.

    Applies leakage guards at track level:
    - Self-exclusion (query track != positive/negative track)
    - Recording-level separation for negatives
    - Cross-recording requirement for positives (optional)

    Args:
        tracks: All tracks.
        query_indices: [B] indices into tracks for this batch.
        neighbor_indices: [B, search_k] neighbor indices from FAISS.
        config: Training config with mining parameters.
        rng: Random generator for fallback sampling.

    Returns:
        positive_indices: [B, positives_per_query]
        positive_mask: [B, positives_per_query] bool
        negative_indices: [B, negatives_per_query]
    """
    n_tracks = len(tracks)
    all_indices = np.arange(n_tracks)
    batch = query_indices.shape[0]

    positives = np.zeros((batch, config.positives_per_query), dtype=np.int64)
    positive_mask = np.zeros((batch, config.positives_per_query), dtype=bool)
    negatives = np.zeros((batch, config.negatives_per_query), dtype=np.int64)

    for row_idx, q_idx in enumerate(query_indices):
        query = tracks[int(q_idx)]
        pos_candidates: list[int] = []
        neg_candidates: list[int] = []

        for cand_idx in neighbor_indices[row_idx]:
            if cand_idx < 0:
                continue
            cand_idx_int = int(cand_idx)
            candidate = tracks[cand_idx_int]

            # Self-exclusion
            if candidate.track_id == query.track_id:
                continue

            # Check positive validity
            is_valid_pos = True
            if config.require_cross_recording_positive:
                if candidate.recording_id == query.recording_id:
                    is_valid_pos = False

            # Check negative validity
            is_valid_neg = True
            if config.exclude_same_recording_negative:
                if candidate.recording_id == query.recording_id:
                    is_valid_neg = False

            if is_valid_pos and len(pos_candidates) < config.positives_per_query:
                pos_candidates.append(cand_idx_int)
            elif is_valid_neg:
                neg_candidates.append(cand_idx_int)

        # Fill positive slots
        seen_pos: set[int] = set()
        pos_slot = 0
        for idx in pos_candidates:
            if idx in seen_pos:
                continue
            seen_pos.add(idx)
            positives[row_idx, pos_slot] = idx
            positive_mask[row_idx, pos_slot] = True
            pos_slot += 1
            if pos_slot >= config.positives_per_query:
                break

        # Fill negative slots
        seen_neg: set[int] = set()
        neg_slot = 0
        for idx in neg_candidates:
            if idx in seen_neg or idx in seen_pos:
                continue
            seen_neg.add(idx)
            negatives[row_idx, neg_slot] = idx
            neg_slot += 1
            if neg_slot >= config.negatives_per_query:
                break

        # Fallback sampling for negatives
        if neg_slot < config.negatives_per_query:
            fallback_pool = [
                int(i)
                for i in all_indices
                if int(i) not in seen_pos
                and int(i) not in seen_neg
                and tracks[int(i)].track_id != query.track_id
            ]
            if config.exclude_same_recording_negative:
                fallback_pool = [
                    i for i in fallback_pool
                    if tracks[i].recording_id != query.recording_id
                ]
            if not fallback_pool:
                # Relax: allow overlap with positives
                fallback_pool = [
                    int(i)
                    for i in all_indices
                    if int(i) not in seen_neg
                    and tracks[int(i)].track_id != query.track_id
                ]
            if fallback_pool:
                needed = config.negatives_per_query - neg_slot
                sampled = rng.choice(
                    fallback_pool,
                    size=needed,
                    replace=len(fallback_pool) < needed,
                )
                negatives[row_idx, neg_slot:config.negatives_per_query] = sampled.astype(
                    np.int64
                )

    return positives, positive_mask, negatives


# ---------------------------------------------------------------------------
# Local loss helpers
# ---------------------------------------------------------------------------


def _pairwise_li_scores(
    query_locals: torch.Tensor,
    query_lengths: torch.Tensor,
    candidate_locals: torch.Tensor,
    candidate_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute late-interaction scores between each query and its candidates.

    Args:
        query_locals: [B, T_q, D] local segment vectors for queries.
        query_lengths: [B] valid segment counts.
        candidate_locals: [B, n_cand, T_c, D] local vectors for candidates.
        candidate_lengths: [B, n_cand] valid segment counts.

    Returns:
        scores: [B, n_cand] late-interaction similarity scores.
    """
    batch, n_cand = candidate_locals.shape[:2]

    # Reshape for batched late_interaction_score:
    # Process each query against all its candidates
    scores = torch.zeros(batch, n_cand, device=query_locals.device)

    for b in range(batch):
        q = query_locals[b : b + 1]  # [1, T_q, D]
        q_len = query_lengths[b : b + 1]  # [1]
        docs = candidate_locals[b]  # [n_cand, T_c, D]
        d_lens = candidate_lengths[b]  # [n_cand]
        scores[b] = late_interaction_score(q, docs, q_len, d_lens).squeeze(0)

    return scores


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------


@dataclass
class ContextTrainResult:
    """Outputs from contextualizer training."""

    input_dim: int
    context_dim: int
    used_device: str
    steps_completed: int
    skipped_steps: int
    final_index_version: int
    metrics: list[dict[str, float | int]]
    online_state_dict: dict[str, torch.Tensor]
    target_state_dict: dict[str, torch.Tensor]
    model_kwargs: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _embed_all_tracks_global(
    model: SegmentTransformer,
    tracks: list[TrackSegments],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run all tracks through model, return global vectors as numpy."""
    model.eval()
    all_globals: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(tracks), batch_size):
            end = min(start + batch_size, len(tracks))
            batch_indices = np.arange(start, end)
            segments, lengths = collate_track_batch(tracks, batch_indices, device)
            global_out, _ = model(segments, lengths)
            all_globals.append(global_out.cpu())

    stacked = torch.cat(all_globals, dim=0)
    return cast(
        np.ndarray[Any, np.dtype[np.float32]],
        np.asarray(stacked.cpu().numpy(), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_contextualizer(
    tracks: list[TrackSegments],
    config: ContextualizerConfig,
) -> ContextTrainResult:
    """Train segment transformer contextualizer with dual multi-task loss.

    Follows the same structure as train_projection_head():
    - Dual online/target models with EMA
    - FAISS index refresh for hard mining
    - Multi-positive InfoNCE for global loss
    - ColBERT-style late-interaction for local loss

    Args:
        tracks: Pre-loaded track segments (from load_track_segments).
        config: Training hyperparameters.

    Returns:
        ContextTrainResult with trained state dicts and metrics.
    """
    config.validate()

    if not tracks:
        raise ValueError("tracks must be non-empty")

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    device = _resolve_device(config.device)

    model_kwargs = dict(
        input_dim=config.input_dim,
        context_dim=config.context_dim,
        num_layers=config.num_transformer_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        max_segments=config.max_segments,
        dropout=config.dropout,
        pool_mode=config.pool_mode,
    )

    online_model = SegmentTransformer(
        input_dim=config.input_dim,
        context_dim=config.context_dim,
        num_layers=config.num_transformer_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        max_segments=config.max_segments,
        dropout=config.dropout,
        pool_mode=config.pool_mode,
    ).to(device)
    target_model = copy.deepcopy(online_model).to(device)
    target_model.eval()

    optimizer = torch.optim.AdamW(
        online_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Filter to train splits
    train_track_ids = set()
    for track in tracks:
        for clip in track.clip_records:
            if clip.split in set(config.train_splits):
                train_track_ids.add(track.track_id)
                break

    train_indices = np.array(
        [i for i, t in enumerate(tracks) if t.track_id in train_track_ids],
        dtype=np.int64,
    )
    if train_indices.size == 0:
        raise ValueError(f"No tracks found for train_splits={config.train_splits}")

    # Initial global embeddings for FAISS index
    target_globals = _embed_all_tracks_global(
        target_model, tracks, device, batch_size=max(config.batch_size, 8)
    )
    retrieval_index = FaissRetrievalIndex.build(target_globals)

    metrics: list[dict[str, float | int]] = []
    skipped_steps = 0

    for step in range(1, config.num_steps + 1):
        # Index refresh
        if step > 1 and should_refresh_index(step, config.refresh_every):
            target_globals = _embed_all_tracks_global(
                target_model, tracks, device, batch_size=max(config.batch_size, 8)
            )
            retrieval_index.rebuild(target_globals)

        # LR warmup
        if config.warmup_steps > 0 and step <= config.warmup_steps:
            warmup_factor = step / config.warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.learning_rate * warmup_factor

        # Sample batch
        replace = train_indices.size < config.batch_size
        batch_indices = rng.choice(
            train_indices,
            size=config.batch_size,
            replace=replace,
        ).astype(np.int64)

        # FAISS search for neighbors
        _, neighbor_indices = retrieval_index.search(
            target_globals[batch_indices],
            k=min(config.search_k, len(tracks)),
        )

        # Mine track-level positives and negatives
        pos_indices, pos_mask_np, neg_indices = _mine_track_pairs(
            tracks, batch_indices, neighbor_indices, config, rng
        )

        if not pos_mask_np.any():
            skipped_steps += 1
            continue

        # Forward pass: online model for queries
        online_model.train()
        q_segments, q_lengths = collate_track_batch(tracks, batch_indices, device)
        q_global, q_local = online_model(q_segments, q_lengths)

        # Forward pass: target model for positives and negatives
        with torch.no_grad():
            target_model.eval()

            # Positives: [B, n_pos]
            n_pos = pos_indices.shape[1]
            pos_flat = pos_indices.reshape(-1)
            p_segments, p_lengths = collate_track_batch(tracks, pos_flat, device)
            p_global_flat, p_local_flat = target_model(p_segments, p_lengths)
            p_global = p_global_flat.reshape(config.batch_size, n_pos, -1)

            # Negatives: [B, n_neg]
            n_neg = neg_indices.shape[1]
            neg_flat = neg_indices.reshape(-1)
            n_segments, n_lengths = collate_track_batch(tracks, neg_flat, device)
            n_global_flat, n_local_flat = target_model(n_segments, n_lengths)
            n_global = n_global_flat.reshape(config.batch_size, n_neg, -1)

        # Compute losses
        pos_mask_t = torch.from_numpy(pos_mask_np).to(device)
        loss_total = torch.tensor(0.0, device=device)
        loss_global_val = 0.0
        loss_local_val = 0.0

        # Global loss: multi_positive_info_nce on pooled vectors
        if config.global_loss_weight > 0:
            loss_global = multi_positive_info_nce(
                query_embeddings=q_global,
                positive_embeddings=p_global,
                negative_embeddings=n_global,
                positive_mask=pos_mask_t,
                temperature=config.temperature,
            )
            loss_total = loss_total + config.global_loss_weight * loss_global
            loss_global_val = float(loss_global.item())

        # Local loss: InfoNCE with late-interaction scoring
        if config.local_loss_weight > 0:
            # Reshape local vectors for late-interaction
            # Positive locals: [B*n_pos, T_p, D] -> [B, n_pos, T_p, D]
            t_p = p_local_flat.shape[1]
            p_local = p_local_flat.reshape(config.batch_size, n_pos, t_p, -1)
            p_lengths_2d = p_lengths.reshape(config.batch_size, n_pos)

            # Negative locals: [B*n_neg, T_n, D] -> [B, n_neg, T_n, D]
            t_n = n_local_flat.shape[1]
            n_local = n_local_flat.reshape(config.batch_size, n_neg, t_n, -1)
            n_lengths_2d = n_lengths.reshape(config.batch_size, n_neg)

            # Compute late-interaction scores as logits
            # Positive scores: [B, n_pos]
            pos_li_scores = _pairwise_li_scores(
                q_local, q_lengths, p_local, p_lengths_2d
            )
            # Negative scores: [B, n_neg]
            neg_li_scores = _pairwise_li_scores(
                q_local, q_lengths, n_local, n_lengths_2d
            )

            # InfoNCE over late-interaction scores
            pos_logits = pos_li_scores / config.temperature
            neg_logits = neg_li_scores / config.temperature

            pos_exp = torch.exp(pos_logits) * pos_mask_t.float()
            pos_sum = pos_exp.sum(dim=1)
            neg_sum = torch.exp(neg_logits).sum(dim=1)

            numerator = torch.clamp(pos_sum, min=1e-12)
            denominator = torch.clamp(pos_sum + neg_sum, min=1e-12)

            valid_rows = pos_mask_t.any(dim=1)
            loss_local_per_row = -torch.log(numerator / denominator)
            loss_local = loss_local_per_row[valid_rows].mean()

            loss_total = loss_total + config.local_loss_weight * loss_local
            loss_local_val = float(loss_local.item())

        # Backward + optimize
        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(online_model.parameters(), max_norm=1.0)
        optimizer.step()

        # EMA update
        _ema_update(
            online_model=online_model,
            target_model=target_model,
            decay=config.ema_decay,
        )

        metrics.append(
            {
                "step": step,
                "loss_total": float(loss_total.item()),
                "loss_global": loss_global_val,
                "loss_local": loss_local_val,
                "valid_queries": int(pos_mask_np.any(axis=1).sum()),
                "index_version": retrieval_index.version,
            }
        )

    if not metrics:
        raise RuntimeError(
            "Training produced no valid steps; mining may be too strict for this dataset"
        )

    return ContextTrainResult(
        input_dim=config.input_dim,
        context_dim=config.context_dim,
        used_device=str(device),
        steps_completed=len(metrics),
        skipped_steps=skipped_steps,
        final_index_version=retrieval_index.version,
        metrics=metrics,
        online_state_dict=_cpu_state_dict(online_model),
        target_state_dict=_cpu_state_dict(target_model),
        model_kwargs=model_kwargs,
    )
