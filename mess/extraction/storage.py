"""
Feature persistence for MERT embeddings.

Module-level functions for saving, loading, and checking .npy features.
No model dependency — can be used independently for feature inspection.

Usage:
    from mess.extraction.storage import load_features, save_features, features_exist

    if features_exist(audio_path, output_dir, track_id="Beethoven_Op027"):
        features = load_features(audio_path, output_dir, track_id="Beethoven_Op027")
"""

import fcntl
import logging
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def _resolve_base_dir(
    output_dir: str | Path,
    dataset: str | None = None
) -> Path:
    """Resolve base output directory with optional dataset subdirectory."""
    output_dir = Path(output_dir)
    return output_dir / dataset if dataset else output_dir


def _resolve_filename(
    audio_path: str | Path,
    track_id: str | None = None
) -> str:
    """Derive filename from track_id or audio path stem."""
    return track_id if track_id else Path(audio_path).stem


def features_exist(
    audio_path: str | Path,
    output_dir: str | Path,
    track_id: str | None = None,
    dataset: str | None = None
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


def features_exist_for_types(
    audio_path: str | Path,
    output_dir: str | Path,
    feature_types: Iterable[str],
    track_id: str | None = None,
    dataset: str | None = None
) -> bool:
    """
    Check whether all requested feature arrays exist on disk.

    Args:
        audio_path: Path to audio file (used for filename fallback)
        output_dir: Root feature directory
        feature_types: Required feature types to validate
        track_id: Custom track ID (optional, defaults to audio file stem)
        dataset: Dataset name for subdirectory (optional)

    Returns:
        True when every requested feature file exists, else False
    """
    if not output_dir:
        return False

    base_dir = _resolve_base_dir(output_dir, dataset)
    filename = _resolve_filename(audio_path, track_id)

    for feature_type in feature_types:
        feature_path = base_dir / feature_type / f"{filename}.npy"
        if not feature_path.exists():
            return False

    return True


def load_features(
    audio_path: str | Path,
    output_dir: str | Path,
    track_id: str | None = None,
    dataset: str | None = None
) -> dict[str, np.ndarray] | None:
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
    audio_path: str | Path,
    output_dir: str | Path,
    feature_types: Iterable[str],
    track_id: str | None = None,
    dataset: str | None = None
) -> dict[str, np.ndarray] | None:
    """
    Load a selected subset of feature arrays from disk.

    Returns None if any requested feature is missing.
    """
    try:
        base_dir = _resolve_base_dir(output_dir, dataset)
        filename = _resolve_filename(audio_path, track_id)

        features: dict[str, np.ndarray] = {}
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
    features: dict[str, np.ndarray],
    audio_path: str | Path,
    output_dir: str | Path,
    track_id: str | None = None,
    dataset: str | None = None
) -> None:
    """
    Save extracted features atomically with file locking.

    Uses temp files + atomic rename to prevent partial writes and file locks
    to prevent concurrent writes to the same track in parallel extraction.

    Args:
        features: Dict with 'raw', 'segments', 'aggregated' arrays
        audio_path: Path to audio file (used for filename fallback)
        output_dir: Root feature directory
        track_id: Custom track ID (optional, defaults to audio file stem)
        dataset: Dataset name for subdirectory (optional)
    """
    base_dir = _resolve_base_dir(output_dir, dataset)
    filename = _resolve_filename(audio_path, track_id)

    # Create lock file directory
    lock_dir = base_dir / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file_path = lock_dir / f"{filename}.lock"

    # Acquire exclusive lock
    try:
        with open(lock_file_path, 'w') as lock_file:
            try:
                # Non-blocking lock - fail fast if another process is writing
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Another process is writing this file, skip gracefully
                logging.debug(f"Features for {filename} being written by another process, skipping")
                return

            try:
                # Write each feature type atomically
                for feature_type, data in features.items():
                    type_dir = base_dir / feature_type
                    type_dir.mkdir(parents=True, exist_ok=True)

                    final_path = type_dir / f"{filename}.npy"

                    # Write to temp file first
                    with tempfile.NamedTemporaryFile(
                        mode='wb',
                        dir=type_dir,
                        delete=False,
                        suffix='.tmp'
                    ) as tmp:
                        np.save(tmp, data)
                        tmp_path = tmp.name

                    # Atomic rename (overwrites existing file if present)
                    shutil.move(tmp_path, final_path)

                logging.info(f"Features saved for {filename}")

            finally:
                # Release lock
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    finally:
        # Clean up lock file
        try:
            lock_file_path.unlink()
        except FileNotFoundError:
            pass
