"""Unit tests for retrieval ranking metrics."""

from __future__ import annotations

import pytest

from mess.evaluation.metrics import evaluate_rankings, ndcg_at_k, recall_at_k, reciprocal_rank

pytestmark = pytest.mark.unit


def test_per_query_metric_primitives() -> None:
    ranking = [1, 2, 3, 4]
    relevant = {2, 4}

    assert recall_at_k(ranking, relevant, k=1) == 0.0
    assert recall_at_k(ranking, relevant, k=2) == 0.5
    assert reciprocal_rank(ranking, relevant) == 0.5
    assert ndcg_at_k(ranking, relevant, k=2) == pytest.approx(0.386852807, rel=1e-6)


def test_evaluate_rankings_aggregate_metrics() -> None:
    rankings = [
        [1, 2, 3, 4],
        [3, 1, 2, 4],
    ]
    relevants = [
        {2, 4},
        {3},
    ]
    report = evaluate_rankings(rankings, relevants, k_values=[1, 2, 5])

    assert report["query_count"] == 2
    assert report["mrr"] == pytest.approx(0.75)
    recall = report["recall"]
    assert recall == {
        "@1": pytest.approx(0.5),
        "@2": pytest.approx(0.75),
        "@5": pytest.approx(1.0),
    }
    ndcg = report["ndcg"]
    assert ndcg["@1"] == pytest.approx(0.5)
    assert ndcg["@2"] == pytest.approx(0.693426403, rel=1e-6)


def test_evaluate_rankings_handles_empty_relevant_sets() -> None:
    report = evaluate_rankings(
        rankings=[[1, 2, 3]],
        relevant_sets=[set()],
        k_values=[1, 5],
    )
    assert report["query_count"] == 0
    assert report["mrr"] == 0.0
    assert report["recall"] == {"@1": 0.0, "@5": 0.0}
    assert report["ndcg"] == {"@1": 0.0, "@5": 0.0}
