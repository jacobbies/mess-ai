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

from .extractor import FeatureExtractor
from .pipeline import ExtractionPipeline
from .audio import load_audio, segment_audio, validate_audio_file
from .storage import load_features, save_features, features_exist

__all__ = [
    'FeatureExtractor',
    'ExtractionPipeline',
    'load_features',
    'save_features',
    'features_exist',
    'load_audio',
    'segment_audio',
    'validate_audio_file',
]
