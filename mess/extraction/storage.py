"""
Feature persistence for MERT embeddings.

Module-level functions for saving, loading, and checking .npy features.
No model dependency — can be used independently for feature inspection.

Usage:
    from mess.extraction.storage import load_features, save_features, features_exist

    if features_exist(audio_path, output_dir, track_id="Beethoven_Op027"):
        features = load_features(audio_path, output_dir, track_id="Beethoven_Op027")
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union, Iterable


def _resolve_base_dir(
    output_dir: Union[str, Path],
    dataset: Optional[str] = None
) -> Path:
    """Resolve base output directory with optional dataset subdirectory."""
    output_dir = Path(output_dir)
    return output_dir / dataset if dataset else output_dir


def _resolve_filename(
    audio_path: Union[str, Path],
    track_id: Optional[str] = None
) -> str:
    """Derive filename from track_id or audio path stem."""
    return track_id if track_id else Path(audio_path).stem


def features_exist(
    audio_path: Union[str, Path],
    output_dir: Union[str, Path],
    track_id: Optional[str] = None,
    dataset: Optional[str] = None
) -> bool:
    """
    Check if features already exist on disk.

    Args:
        audio_path: Path to audio file (used for filename fallback)
        output_dir: Root feature directory
        track_id: Custom track ID (optional, defaults to audio file stem)
        dataset: Dataset name for subdirectory (optional)

    Returns:
        True if aggregated features exist, False otherwise
    """
    if not output_dir:
        return False

    base_dir = _resolve_base_dir(output_dir, dataset)
    filename = _resolve_filename(audio_path, track_id)
    aggregated_path = base_dir / "aggregated" / f"{filename}.npy"
    return aggregated_path.exists()


def load_features(
    audio_path: Union[str, Path],
    output_dir: Union[str, Path],
    track_id: Optional[str] = None,
    dataset: Optional[str] = None
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load pre-extracted features from disk (~1000x faster than re-extracting).

    Args:
        audio_path: Path to audio file (used for filename fallback)
        output_dir: Root feature directory
        track_id: Custom track ID (optional, defaults to audio file stem)
        dataset: Dataset name for subdirectory (optional)

    Returns:
        Dict with 'raw', 'segments', 'aggregated' keys, or None if missing
    """
    try:
        base_dir = _resolve_base_dir(output_dir, dataset)
        filename = _resolve_filename(audio_path, track_id)

        features = {}
        for feature_type in ['raw', 'segments', 'aggregated']:
            feature_path = base_dir / feature_type / f"{filename}.npy"
            if not feature_path.exists():
                return None  # Missing features, need to re-extract
            features[feature_type] = np.load(feature_path)

        return features

    except Exception as e:
        logging.warning(f"Error loading existing features for {audio_path}: {e}")
        return None


def load_selected_features(
    audio_path: Union[str, Path],
    output_dir: Union[str, Path],
    feature_types: Iterable[str],
    track_id: Optional[str] = None,
    dataset: Optional[str] = None
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load a selected subset of feature arrays from disk.

    Returns None if any requested feature is missing.
    """
    try:
        base_dir = _resolve_base_dir(output_dir, dataset)
        filename = _resolve_filename(audio_path, track_id)

        features: Dict[str, np.ndarray] = {}
        for feature_type in feature_types:
            feature_path = base_dir / feature_type / f"{filename}.npy"
            if not feature_path.exists():
                return None
            features[feature_type] = np.load(feature_path)

        return features

    except Exception as e:
        logging.warning(f"Error loading selected features for {audio_path}: {e}")
        return None


def save_features(
    features: Dict[str, np.ndarray],
    audio_path: Union[str, Path],
    output_dir: Union[str, Path],
    track_id: Optional[str] = None,
    dataset: Optional[str] = None
) -> None:
    """
    Save extracted features to disk as .npy files.

    Args:
        features: Dict with 'raw', 'segments', 'aggregated' arrays
        audio_path: Path to audio file (used for filename fallback)
        output_dir: Root feature directory
        track_id: Custom track ID (optional, defaults to audio file stem)
        dataset: Dataset name for subdirectory (optional)
    """
    base_dir = _resolve_base_dir(output_dir, dataset)
    filename = _resolve_filename(audio_path, track_id)

    for feature_type, data in features.items():
        type_dir = base_dir / feature_type
        type_dir.mkdir(parents=True, exist_ok=True)

        save_path = type_dir / f"{filename}.npy"
        np.save(save_path, data)

    logging.info(f"Features saved for {filename}")
