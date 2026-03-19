"""Segment transformer contextualizer for track-level representation learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SegmentTransformer(nn.Module):
    """Transformer over frozen MERT segment sequences within a track.

    Produces dual outputs:
    - Global: pooled track-level vector for FAISS stage-1 retrieval
    - Local: per-segment contextualized vectors for ColBERT-style reranking

    Architecture:
        Input projection (768 -> 256) + learnable positional embeddings
        -> pre-norm transformer encoder (2 layers, 4 heads)
        -> residual from projected input (locality preservation)
        -> output LayerNorm
        -> L2-normalized outputs
    """

    def __init__(
        self,
        input_dim: int = 768,
        context_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 512,
        max_segments: int = 512,
        dropout: float = 0.1,
        pool_mode: str = "mean",
    ) -> None:
        super().__init__()
        if context_dim % num_heads != 0:
            raise ValueError("context_dim must be divisible by num_heads")
        if pool_mode not in {"mean", "cls"}:
            raise ValueError("pool_mode must be 'mean' or 'cls'")

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.pool_mode = pool_mode

        # Input projection: map MERT dim to context dim
        self.input_proj = nn.Linear(input_dim, context_dim)

        # Learnable positional embeddings (segment order matters)
        self.pos_embed = nn.Embedding(max_segments, context_dim)

        # Pre-norm transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Residual projection: separate path from input to preserve locality
        self.residual_proj = nn.Linear(input_dim, context_dim)

        # Output normalization after residual add
        self.output_norm = nn.LayerNorm(context_dim)

    def forward(
        self,
        segments: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing global and local representations.

        Args:
            segments: [B, T, input_dim] frozen MERT segment embeddings.
            lengths: [B] number of valid segments per track (for padding mask).

        Returns:
            global_out: [B, context_dim] L2-normalized pooled track vector.
            local_out: [B, T, context_dim] L2-normalized per-segment vectors.
        """
        batch_size, max_t, _ = segments.shape

        # Build padding mask: True where positions are padding (PyTorch convention)
        positions = torch.arange(max_t, device=segments.device).unsqueeze(0)
        padding_mask = positions >= lengths.unsqueeze(1)  # [B, T]

        # Input projection + positional embeddings
        projected = self.input_proj(segments)  # [B, T, context_dim]
        pos_indices = torch.arange(max_t, device=segments.device)
        projected = projected + self.pos_embed(pos_indices)  # broadcast over batch

        # Transformer encoder with padding mask
        transformed = self.transformer(
            projected,
            src_key_padding_mask=padding_mask,
        )  # [B, T, context_dim]

        # Residual from input (locality preservation)
        residual = self.residual_proj(segments)  # [B, T, context_dim]
        combined = transformed + residual

        # Output normalization
        local_out = self.output_norm(combined)  # [B, T, context_dim]

        # Global pooling (masked to exclude padding)
        valid_mask = ~padding_mask  # [B, T] True where valid
        if self.pool_mode == "mean":
            # Masked mean pooling
            mask_expanded = valid_mask.unsqueeze(-1).float()  # [B, T, 1]
            pooled_sum = (local_out * mask_expanded).sum(dim=1)  # [B, context_dim]
            pooled_count = mask_expanded.sum(dim=1).clamp(min=1.0)  # [B, 1]
            global_out = pooled_sum / pooled_count
        else:
            # CLS mode: use first segment as global representation
            global_out = local_out[:, 0, :]  # [B, context_dim]

        # L2-normalize both outputs
        global_out = F.normalize(global_out, dim=-1)
        local_out = F.normalize(local_out, dim=-1)

        return global_out, local_out


def late_interaction_score(
    query_locals: torch.Tensor,
    doc_locals: torch.Tensor,
    query_lengths: torch.Tensor,
    doc_lengths: torch.Tensor,
) -> torch.Tensor:
    """ColBERT-style MaxSim scoring between segment sequences.

    For each query segment i, finds the max cosine similarity across all
    doc segments j, then averages over query segments:
        s(Q, D) = mean_i max_j cos(h_i^Q, h_j^D)

    Args:
        query_locals: [n_q, T_q, D] L2-normalized query segment vectors.
        doc_locals: [n_d, T_d, D] L2-normalized doc segment vectors.
        query_lengths: [n_q] valid segment counts for queries.
        doc_lengths: [n_d] valid segment counts for docs.

    Returns:
        scores: [n_q, n_d] late-interaction similarity scores.
    """
    n_q, t_q, dim = query_locals.shape
    n_d, t_d, _ = doc_locals.shape

    # Pairwise segment cosine similarities: [n_q, n_d, T_q, T_d]
    # query_locals[q, i, d] * doc_locals[n, j, d] -> sim[q, n, i, j]
    sim = torch.einsum("qid,njd->qnij", query_locals, doc_locals)

    # Mask out padding positions in doc dimension
    doc_positions = torch.arange(t_d, device=doc_locals.device)
    doc_valid = doc_positions.unsqueeze(0) < doc_lengths.unsqueeze(1)  # [n_d, T_d]
    doc_mask = doc_valid.unsqueeze(0).unsqueeze(2)  # [1, n_d, 1, T_d]
    sim = sim.masked_fill(~doc_mask, float("-inf"))

    # MaxSim: max over doc segments for each query segment
    max_sim = sim.max(dim=-1).values  # [n_q, n_d, T_q]

    # Mask out padding positions in query dimension and average
    query_positions = torch.arange(t_q, device=query_locals.device)
    query_valid = query_positions.unsqueeze(0) < query_lengths.unsqueeze(1)  # [n_q, T_q]
    query_mask = query_valid.unsqueeze(1)  # [n_q, 1, T_q]

    # Replace padded query positions with 0 before averaging
    max_sim = max_sim.masked_fill(~query_mask, 0.0)
    scores = max_sim.sum(dim=-1) / query_lengths.unsqueeze(1).float().clamp(min=1.0)

    return scores  # [n_q, n_d]
