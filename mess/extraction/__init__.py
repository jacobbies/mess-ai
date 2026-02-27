"""
MERT Feature Extraction Package

Public API:
    FeatureExtractor     - Core MERT model + inference + track extraction
    ExtractionPipeline   - Dataset-level batch processing
    load_features        - Load pre-extracted features from disk
    save_features        - Save features to disk as .npy files
    features_exist       - Check if features already exist
    load_audio           - Load and preprocess audio (no model needed)
    segment_audio        - Segment audio into overlapping windows
    validate_audio_file  - Validate audio file before extraction
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "FeatureExtractor",
    "ExtractionPipeline",
    "load_features",
    "save_features",
    "features_exist",
    "load_audio",
    "segment_audio",
    "validate_audio_file",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FeatureExtractor": (".extractor", "FeatureExtractor"),
    "ExtractionPipeline": (".pipeline", "ExtractionPipeline"),
    "load_features": (".storage", "load_features"),
    "save_features": (".storage", "save_features"),
    "features_exist": (".storage", "features_exist"),
    "load_audio": (".audio", "load_audio"),
    "segment_audio": (".audio", "segment_audio"),
    "validate_audio_file": (".audio", "validate_audio_file"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
