"""Unit tests for SegmentTransformer and late_interaction_score."""

from __future__ import annotations

import pytest
import torch

from mess.training.contextualizer import SegmentTransformer, late_interaction_score

pytestmark = pytest.mark.unit


class TestSegmentTransformer:
    def _make_model(self, **kwargs: object) -> SegmentTransformer:
        defaults = dict(input_dim=64, context_dim=32, num_layers=1, num_heads=4, ff_dim=64)
        defaults.update(kwargs)
        return SegmentTransformer(**defaults)

    def test_forward_shapes(self) -> None:
        model = self._make_model()
        segments = torch.randn(2, 5, 64)
        lengths = torch.tensor([5, 3])

        global_out, local_out = model(segments, lengths)

        assert global_out.shape == (2, 32)
        assert local_out.shape == (2, 5, 32)

    def test_outputs_are_l2_normalized(self) -> None:
        model = self._make_model()
        segments = torch.randn(3, 4, 64)
        lengths = torch.tensor([4, 2, 3])

        global_out, local_out = model(segments, lengths)

        # Global vectors should have unit norm
        global_norms = torch.linalg.norm(global_out, dim=-1)
        torch.testing.assert_close(global_norms, torch.ones(3), atol=1e-5, rtol=0)

        # Local vectors should have unit norm
        local_norms = torch.linalg.norm(local_out, dim=-1)
        expected_norms = torch.ones(3, 4)
        torch.testing.assert_close(local_norms, expected_norms, atol=1e-5, rtol=0)

    def test_padding_mask_handles_variable_lengths(self) -> None:
        """Padding positions should not affect global pooling."""
        model = self._make_model()
        model.eval()

        segments = torch.randn(1, 6, 64)
        lengths_full = torch.tensor([6])
        lengths_partial = torch.tensor([3])

        with torch.no_grad():
            global_full, _ = model(segments, lengths_full)
            global_partial, _ = model(segments, lengths_partial)

        # Different lengths should produce different global vectors
        assert not torch.allclose(global_full, global_partial, atol=1e-4)

    def test_residual_preserves_signal(self) -> None:
        """With zero-initialized transformer, output should approximate projected input."""
        model = self._make_model(num_layers=1)

        # Zero out transformer weights to make it pass through
        for param in model.transformer.parameters():
            param.data.zero_()

        segments = torch.randn(2, 3, 64)
        lengths = torch.tensor([3, 2])

        with torch.no_grad():
            _, local_out = model(segments, lengths)

        # The residual path should still produce non-zero output
        assert local_out.abs().sum() > 0

    def test_single_segment_track(self) -> None:
        model = self._make_model()
        segments = torch.randn(1, 1, 64)
        lengths = torch.tensor([1])

        global_out, local_out = model(segments, lengths)

        assert global_out.shape == (1, 32)
        assert local_out.shape == (1, 1, 32)

    def test_cls_pool_mode(self) -> None:
        model = self._make_model(pool_mode="cls")
        segments = torch.randn(2, 4, 64)
        lengths = torch.tensor([4, 3])

        global_out, local_out = model(segments, lengths)

        assert global_out.shape == (2, 32)
        assert local_out.shape == (2, 4, 32)

    def test_invalid_pool_mode(self) -> None:
        with pytest.raises(ValueError, match="pool_mode"):
            self._make_model(pool_mode="max")

    def test_invalid_head_divisibility(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            self._make_model(context_dim=30, num_heads=4)


class TestLateInteractionScore:
    def test_output_shape(self) -> None:
        query = torch.randn(3, 5, 32)
        query = torch.nn.functional.normalize(query, dim=-1)
        doc = torch.randn(4, 6, 32)
        doc = torch.nn.functional.normalize(doc, dim=-1)
        q_len = torch.tensor([5, 3, 4])
        d_len = torch.tensor([6, 4, 5, 2])

        scores = late_interaction_score(query, doc, q_len, d_len)

        assert scores.shape == (3, 4)

    def test_self_similarity_is_high(self) -> None:
        """A track scored against itself should produce high similarity."""
        segments = torch.randn(1, 4, 32)
        segments = torch.nn.functional.normalize(segments, dim=-1)
        lengths = torch.tensor([4])

        scores = late_interaction_score(segments, segments, lengths, lengths)

        # Self-similarity should be close to 1.0
        assert scores.item() > 0.95

    def test_respects_padding(self) -> None:
        """Padding segments should not affect scores."""
        query = torch.randn(1, 4, 16)
        query = torch.nn.functional.normalize(query, dim=-1)
        doc = torch.randn(1, 6, 16)
        doc = torch.nn.functional.normalize(doc, dim=-1)

        # Score with full doc vs truncated doc lengths
        scores_full = late_interaction_score(
            query, doc, torch.tensor([4]), torch.tensor([6])
        )
        scores_short = late_interaction_score(
            query, doc, torch.tensor([4]), torch.tensor([3])
        )

        # Different doc lengths should generally produce different scores
        assert not torch.allclose(scores_full, scores_short, atol=1e-4)

    def test_single_segment_each(self) -> None:
        q = torch.tensor([[[1.0, 0.0, 0.0]]])
        d = torch.tensor([[[0.0, 1.0, 0.0]]])
        scores = late_interaction_score(q, d, torch.tensor([1]), torch.tensor([1]))

        # Orthogonal vectors should score 0
        assert abs(scores.item()) < 1e-5
