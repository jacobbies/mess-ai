import pickle
import json
import logging
from pathlib import Path
from typing import List, Optional
from .faiss_index import FAISSIndex


class IndexCache:
    """
    Handles persistence of FAISS indices and metadata to disk.
    
    Provides fast startup by avoiding index rebuilding when features haven't changed.
    """
    
    def __init__(self, cache_dir: str = "data/cache/faiss"):
        """
        Initialize index cache.
        
        Args:
            cache_dir: Directory to store cached indices and metadata
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_index_path(self, feature_type: str) -> Path:
        """Get path for cached FAISS index file."""
        return self.cache_dir / f"{feature_type}_index.faiss"
    
    def _get_metadata_path(self, feature_type: str) -> Path:
        """Get path for cached metadata file."""
        return self.cache_dir / f"{feature_type}_metadata.json"
    
    def index_exists(self, feature_type: str) -> bool:
        """
        Check if a cached index exists for the given feature type.
        
        Args:
            feature_type: Type of features ('aggregated', 'segments', 'raw')
            
        Returns:
            True if both index and metadata files exist
        """
        index_path = self._get_index_path(feature_type)
        metadata_path = self._get_metadata_path(feature_type)
        
        return index_path.exists() and metadata_path.exists()
    
    def save_index(self, faiss_index: FAISSIndex, feature_type: str) -> None:
        """
        Save FAISS index and metadata to cache.
        
        Args:
            faiss_index: The FAISS index to cache
            feature_type: Type of features ('aggregated', 'segments', 'raw')
        """
        try:
            index_path = self._get_index_path(feature_type)
            metadata_path = self._get_metadata_path(feature_type)
            
            # Save FAISS index
            faiss_index.save_index(str(index_path))
            
            # Save metadata (track names and other info)
            metadata = {
                "track_names": faiss_index.get_track_names(),
                "feature_type": feature_type,
                "dimension": faiss_index.dimension,
                "num_tracks": len(faiss_index.get_track_names())
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logging.info(f"Cached FAISS index for {feature_type} features to {self.cache_dir}")
            
        except Exception as e:
            logging.error(f"Failed to cache FAISS index: {e}")
            # Clean up partially written files
            for path in [index_path, metadata_path]:
                if path.exists():
                    path.unlink()
            raise
    
    def load_index(self, feature_type: str, expected_track_names: Optional[List[str]] = None) -> FAISSIndex:
        """
        Load FAISS index from cache.
        
        Args:
            feature_type: Type of features ('aggregated', 'segments', 'raw')
            expected_track_names: Optional list of expected track names for validation
            
        Returns:
            Loaded FAISS index
            
        Raises:
            FileNotFoundError: If cached files don't exist
            ValueError: If cached data doesn't match expected track names
        """
        index_path = self._get_index_path(feature_type)
        metadata_path = self._get_metadata_path(feature_type)
        
        if not self.index_exists(feature_type):
            raise FileNotFoundError(f"Cached index not found for {feature_type} features")
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            cached_track_names = metadata["track_names"]
            dimension = metadata["dimension"]
            
            # Validate against expected track names if provided
            if expected_track_names is not None:
                if set(cached_track_names) != set(expected_track_names):
                    raise ValueError(
                        f"Cached track names don't match current features. "
                        f"Expected {len(expected_track_names)} tracks, "
                        f"cached has {len(cached_track_names)} tracks."
                    )
            
            # Create and load FAISS index
            faiss_index = FAISSIndex(dimension=dimension)
            faiss_index.load_index(str(index_path), cached_track_names)
            
            logging.info(f"Loaded cached FAISS index for {feature_type} features")
            return faiss_index
            
        except Exception as e:
            logging.error(f"Failed to load cached FAISS index: {e}")
            raise
    
    def clear_cache(self, feature_type: Optional[str] = None) -> None:
        """
        Clear cached indices.
        
        Args:
            feature_type: Specific feature type to clear, or None to clear all
        """
        if feature_type:
            # Clear specific feature type
            paths_to_remove = [
                self._get_index_path(feature_type),
                self._get_metadata_path(feature_type)
            ]
        else:
            # Clear all cached files
            paths_to_remove = list(self.cache_dir.glob("*"))
        
        removed_count = 0
        for path in paths_to_remove:
            if path.exists():
                path.unlink()
                removed_count += 1
        
        logging.info(f"Cleared {removed_count} cached files")
    
    def get_cache_info(self) -> dict:
        """
        Get information about cached indices.
        
        Returns:
            Dictionary with cache statistics
        """
        info = {
            "cache_dir": str(self.cache_dir),
            "cached_features": []
        }
        
        # Check for each feature type
        for feature_type in ["aggregated", "segments", "raw"]:
            if self.index_exists(feature_type):
                metadata_path = self._get_metadata_path(feature_type)
                index_path = self._get_index_path(feature_type)
                
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    info["cached_features"].append({
                        "feature_type": feature_type,
                        "num_tracks": metadata["num_tracks"],
                        "dimension": metadata["dimension"],
                        "index_size_mb": round(index_path.stat().st_size / (1024 * 1024), 2)
                    })
                except Exception as e:
                    logging.warning(f"Could not read metadata for {feature_type}: {e}")
        
        return info