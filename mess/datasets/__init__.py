"""Dataset contracts and torch adapters."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BaseDataset",
    "DatasetFactory",
    "SMDDataset",
    "MAESTRODataset",
    "ClipRecord",
    "ClipIndex",
    "EmbeddingStore",
    "TargetStore",
    "FeatureStore",
    "AudioStore",
    "NpySegmentEmbeddingStore",
    "NPZSegmentTargetStore",
    "GeneralTorchDataset",
    "ClipDataset",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BaseDataset": (".base", "BaseDataset"),
    "DatasetFactory": (".factory", "DatasetFactory"),
    "SMDDataset": (".smd", "SMDDataset"),
    "MAESTRODataset": (".maestro", "MAESTRODataset"),
    "ClipRecord": (".clip_index", "ClipRecord"),
    "ClipIndex": (".clip_index", "ClipIndex"),
    "EmbeddingStore": (".stores", "EmbeddingStore"),
    "TargetStore": (".stores", "TargetStore"),
    "FeatureStore": (".stores", "FeatureStore"),
    "AudioStore": (".stores", "AudioStore"),
    "NpySegmentEmbeddingStore": (".stores", "NpySegmentEmbeddingStore"),
    "NPZSegmentTargetStore": (".stores", "NPZSegmentTargetStore"),
    "GeneralTorchDataset": (".torch_dataset", "GeneralTorchDataset"),
    "ClipDataset": (".torch_clip_dataset", "ClipDataset"),
}


def __getattr__(name: str) -> Any:
    """Lazily import dataset modules to keep serving installs lightweight."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'mess.datasets' has no attribute {name!r}")

    module_path, attr_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
