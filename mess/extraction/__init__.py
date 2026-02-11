"""
MERT Feature Extraction Package

Public API:
    FeatureExtractor     - Core MERT model + inference + track extraction
    ExtractionPipeline   - Dataset-level batch processing
    FeatureStorage       - Feature save/load/exists
    load_audio           - Load and preprocess audio (no model needed)
    segment_audio        - Segment audio into overlapping windows
    validate_audio_file  - Validate audio file before extraction
"""

from .extractor import FeatureExtractor
from .pipeline import ExtractionPipeline
from .audio import load_audio, segment_audio, validate_audio_file
from .storage import FeatureStorage

__all__ = [
    'FeatureExtractor',
    'ExtractionPipeline',
    'FeatureStorage',
    'load_audio',
    'segment_audio',
    'validate_audio_file',
]
