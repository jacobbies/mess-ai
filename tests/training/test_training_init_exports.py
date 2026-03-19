"""Tests for training package exports and lazy imports."""

from __future__ import annotations

import pytest

import mess.training as training

pytestmark = pytest.mark.unit


def test_lazy_import_training_exports() -> None:
    assert training.RetrievalSSLConfig.__name__ == "RetrievalSSLConfig"
    assert training.TrainResult.__name__ == "TrainResult"
    assert callable(training.train_projection_head)
    assert callable(training.project_vectors_with_head)
    assert callable(training.export_projection_clip_artifact)


def test_public_api_contract_names() -> None:
    expected = {
        "RetrievalSSLConfig",
        "ProjectionHead",
        "TrainResult",
        "train_projection_head",
        "project_vectors_with_head",
        "export_projection_clip_artifact",
        "ContextualizerConfig",
        "SegmentTransformer",
        "late_interaction_score",
        "ContextTrainResult",
        "TrackSegments",
        "train_contextualizer",
        "load_contextualizer_from_state",
        "contextualize_tracks",
        "save_contextualized_embeddings",
        "export_contextualizer_track_artifact",
    }
    assert expected == set(training.__all__)
    assert expected.issubset(set(dir(training)))
