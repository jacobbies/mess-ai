"""
Feature persistence for MERT embeddings.

Handles saving, loading, and checking existence of extracted .npy features.
No model dependency — can be used independently for feature inspection.

Usage:
    from mess.extraction.cache import FeatureCache

    cache = FeatureCache()
    if cache.exists(audio_path, output_dir, track_id="Beethoven_Op027"):
        features = cache.load(audio_path, output_dir, track_id="Beethoven_Op027")
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union


class FeatureCache:
    """
    Feature save/load/exists operations for .npy MERT embeddings.

    Consolidates the path-resolution logic (track_id fallback, dataset
    subdirectory) into a single resolve_path() method, eliminating
    duplication across exists/load/save.
    """

    def resolve_path(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        track_id: Optional[str] = None,
        dataset: Optional[str] = None
    ) -> Path:
        """
        Resolve the base output directory and filename for a track.

        Args:
            audio_path: Path to audio file (used for filename fallback)
            output_dir: Root feature directory
            track_id: Custom track ID (optional, defaults to audio file stem)
            dataset: Dataset name for subdirectory (optional)

        Returns:
            Tuple-like Path: output_dir / [dataset] with filename derivable
        """
        output_dir = Path(output_dir)
        if dataset:
            output_dir = output_dir / dataset
        return output_dir

    def _filename(self, audio_path: Union[str, Path], track_id: Optional[str] = None) -> str:
        """Derive filename from track_id or audio path stem."""
        return track_id if track_id else Path(audio_path).stem

    def exists(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        track_id: Optional[str] = None,
        dataset: Optional[str] = None
    ) -> bool:
        """
        Check if features already exist on disk.

        Returns:
            True if aggregated features exist, False otherwise
        """
        if not output_dir:
            return False

        base_dir = self.resolve_path(audio_path, output_dir, track_id, dataset)
        filename = self._filename(audio_path, track_id)
        aggregated_path = base_dir / "aggregated" / f"{filename}.npy"
        return aggregated_path.exists()

    def load(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        track_id: Optional[str] = None,
        dataset: Optional[str] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Load pre-extracted features from disk (~1000x faster than re-extracting).

        Returns:
            Dict with 'raw', 'segments', 'aggregated' keys, or None if missing
        """
        try:
            base_dir = self.resolve_path(audio_path, output_dir, track_id, dataset)
            filename = self._filename(audio_path, track_id)

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

    def save(
        self,
        features: Dict[str, np.ndarray],
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        track_id: Optional[str] = None,
        dataset: Optional[str] = None
    ) -> None:
        """Save extracted features to disk as .npy files."""
        base_dir = self.resolve_path(audio_path, output_dir, track_id, dataset)
        filename = self._filename(audio_path, track_id)

        for feature_type, data in features.items():
            type_dir = base_dir / feature_type
            type_dir.mkdir(parents=True, exist_ok=True)

            save_path = type_dir / f"{filename}.npy"
            np.save(save_path, data)

        logging.info(f"Features saved for {filename}")
