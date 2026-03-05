"""Tests for retrieval training losses."""

from __future__ import annotations

import pytest
import torch

from mess.training.losses import multi_positive_info_nce

pytestmark = pytest.mark.unit


def test_multi_positive_info_nce_prefers_close_positives() -> None:
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    positives_good = torch.tensor([[[0.95, 0.05], [0.9, 0.1]]], dtype=torch.float32)
    positives_bad = torch.tensor([[[0.0, 1.0], [0.1, 0.9]]], dtype=torch.float32)
    negatives = torch.tensor([[[0.0, 1.0], [-0.1, 1.0]]], dtype=torch.float32)

    good_loss = multi_positive_info_nce(query, positives_good, negatives, temperature=0.07)
    bad_loss = multi_positive_info_nce(query, positives_bad, negatives, temperature=0.07)

    assert good_loss.item() < bad_loss.item()


def test_multi_positive_info_nce_honors_positive_mask() -> None:
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    positives = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    negatives = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)

    mask = torch.tensor([[True, False]])
    loss = multi_positive_info_nce(
        query_embeddings=query,
        positive_embeddings=positives,
        negative_embeddings=negatives,
        positive_mask=mask,
        temperature=0.07,
    )

    assert torch.isfinite(loss)
    assert loss.item() < 0.2
