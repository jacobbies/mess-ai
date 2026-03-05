"""Loss functions for retrieval-augmented metric learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_positive_info_nce(
    query_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    positive_mask: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute a multi-positive InfoNCE objective.

    Args:
        query_embeddings: [batch, dim]
        positive_embeddings: [batch, n_pos, dim]
        negative_embeddings: [batch, n_neg, dim]
        positive_mask: Optional bool mask [batch, n_pos] for valid positives.
        temperature: Softmax temperature.

    Returns:
        Scalar loss tensor.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if query_embeddings.ndim != 2:
        raise ValueError(f"query_embeddings must be 2D, got {query_embeddings.shape}")
    if positive_embeddings.ndim != 3:
        raise ValueError(f"positive_embeddings must be 3D, got {positive_embeddings.shape}")
    if negative_embeddings.ndim != 3:
        raise ValueError(f"negative_embeddings must be 3D, got {negative_embeddings.shape}")

    batch, dim = query_embeddings.shape
    if positive_embeddings.shape[0] != batch or positive_embeddings.shape[2] != dim:
        raise ValueError("positive_embeddings shape mismatch with query_embeddings")
    if negative_embeddings.shape[0] != batch or negative_embeddings.shape[2] != dim:
        raise ValueError("negative_embeddings shape mismatch with query_embeddings")

    q = F.normalize(query_embeddings, dim=1)
    p = F.normalize(positive_embeddings, dim=2)
    n = F.normalize(negative_embeddings, dim=2)

    pos_logits = torch.einsum("bd,bpd->bp", q, p) / temperature
    neg_logits = torch.einsum("bd,bnd->bn", q, n) / temperature

    if positive_mask is None:
        pos_mask = torch.ones_like(pos_logits, dtype=torch.bool)
    else:
        if positive_mask.shape != pos_logits.shape:
            raise ValueError("positive_mask shape mismatch with positive logits")
        pos_mask = positive_mask.to(dtype=torch.bool)

    valid_rows = pos_mask.any(dim=1)
    if not torch.any(valid_rows):
        raise ValueError("No valid positive pairs in batch")

    masked_pos_exp = torch.exp(pos_logits) * pos_mask.to(pos_logits.dtype)
    pos_sum = masked_pos_exp.sum(dim=1)
    neg_sum = torch.exp(neg_logits).sum(dim=1)

    numerator = torch.clamp(pos_sum, min=1e-12)
    denominator = torch.clamp(pos_sum + neg_sum, min=1e-12)

    loss_per_row = -torch.log(numerator / denominator)
    return loss_per_row[valid_rows].mean()


def batch_similarity_stats(
    query_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    positive_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute simple cosine stats for a training batch."""
    q = F.normalize(query_embeddings, dim=1)
    p = F.normalize(positive_embeddings, dim=2)
    n = F.normalize(negative_embeddings, dim=2)

    pos_sim = torch.einsum("bd,bpd->bp", q, p)
    neg_sim = torch.einsum("bd,bnd->bn", q, n)

    if positive_mask is None:
        pos_values = pos_sim.reshape(-1)
    else:
        pos_values = pos_sim[positive_mask]

    return {
        "pos_sim_mean": float(pos_values.mean().item()) if pos_values.numel() else float("nan"),
        "neg_sim_mean": float(neg_sim.mean().item()) if neg_sim.numel() else float("nan"),
    }
