"""Retrieval-augmented training utilities for expressive embedding learning."""

from .config import RetrievalSSLConfig
from .trainer import ProjectionHead, TrainResult, train_projection_head

__all__ = [
    "RetrievalSSLConfig",
    "ProjectionHead",
    "TrainResult",
    "train_projection_head",
]
