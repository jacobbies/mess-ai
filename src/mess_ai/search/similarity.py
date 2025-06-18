import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from .faiss_index import FAISSIndex
from .cache import IndexCache


class SimilaritySearchEngine:
    """
    High-level interface for music similarity search using FAISS.
    
    Handles feature loading, index building, and similarity queries.
    """
    
    def __init__(self, features_dir: str = "data/processed/features", 
                 feature_type: str = "aggregated", 
                 cache_dir: Optional[str] = None):
        """
        Initialize similarity search engine.
        
        Args:
            features_dir: Directory containing extracted features
            feature_type: Type of features to use ('aggregated', 'segments', 'raw')
            cache_dir: Directory for caching FAISS indices (optional)
        """
        self.features_dir = Path(features_dir)
        self.feature_type = feature_type
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize components
        self.faiss_index = None
        self.cache = IndexCache(cache_dir) if cache_dir else None
        self.features = {}
        self.track_names = []
        
        # Load features and build index
        self._initialize()
    
    def _initialize(self):
        """Load features and build/load FAISS index."""
        try:
            # Load features
            self._load_features()
            
            # Try to load cached index first
            if self.cache and self.cache.index_exists(self.feature_type):
                logging.info("Loading cached FAISS index...")
                self.faiss_index = self.cache.load_index(self.feature_type, self.track_names)
            else:
                # Build new index
                logging.info("Building new FAISS index...")
                self._build_index()
                
                # Cache the index if cache is enabled
                if self.cache:
                    self.cache.save_index(self.faiss_index, self.feature_type)
                    
        except Exception as e:
            logging.error(f"Failed to initialize similarity search engine: {e}")
            raise
    
    def _load_features(self):
        """Load all precomputed features into memory."""
        feature_path = self.features_dir / self.feature_type
        if not feature_path.exists():
            raise FileNotFoundError(f"Features directory not found: {feature_path}")
        
        feature_files = list(feature_path.glob("*.npy"))
        if not feature_files:
            raise FileNotFoundError(f"No feature files found in {feature_path}")
        
        logging.info(f"Loading {len(feature_files)} feature files...")
        
        for feature_file in sorted(feature_files):  # Sort for consistent ordering
            track_name = feature_file.stem  # Remove .npy extension
            features = np.load(feature_file)
            
            # Handle different feature types
            if self.feature_type == "aggregated":
                # Shape: (13, 768) -> flatten to (9984,)
                features = features.flatten()
            elif self.feature_type == "segments":
                # Shape: (num_segments, 13, 768) -> mean over segments
                features = np.mean(features, axis=0).flatten()
            elif self.feature_type == "raw":
                # Shape: (num_segments, 13, time_steps, 768) -> mean over time and segments
                features = np.mean(features, axis=(0, 2)).flatten()
            
            self.features[track_name] = features
            self.track_names.append(track_name)
        
        logging.info(f"Loaded features for {len(self.track_names)} tracks")
        logging.info(f"Feature dimensionality: {features.shape[0]}")
    
    def _build_index(self):
        """Build FAISS index from loaded features."""
        if not self.features:
            raise RuntimeError("No features loaded. Cannot build index.")
        
        # Convert features dict to matrix
        feature_matrix = np.array([self.features[name] for name in self.track_names])
        
        # Initialize and build FAISS index
        dimension = feature_matrix.shape[1]
        self.faiss_index = FAISSIndex(dimension=dimension)
        self.faiss_index.build_index(feature_matrix, self.track_names)
        
        logging.info("FAISS index built successfully")
    
    def search(self, track_name: str, top_k: int = 5, exclude_self: bool = True) -> List[Tuple[str, float]]:
        """
        Find tracks similar to the given track.
        
        Args:
            track_name: Name of the reference track (without .npy extension)
            top_k: Number of similar tracks to return
            exclude_self: Whether to exclude the reference track from results
            
        Returns:
            List of (track_name, similarity_score) tuples, sorted by similarity descending
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not initialized")
        
        try:
            # Use FAISS index to find similar tracks
            similar_tracks, similarity_scores = self.faiss_index.search_by_track_name(
                track_name, top_k, exclude_self
            )
            
            # Return as list of tuples
            return list(zip(similar_tracks, similarity_scores))
            
        except Exception as e:
            logging.error(f"Error searching for similar tracks to '{track_name}': {e}")
            raise
    
    def get_track_names(self) -> List[str]:
        """Get list of all available track names."""
        return self.track_names.copy()
    
    def get_random_recommendations(self, count: int = 5) -> List[str]:
        """Get random track recommendations (fallback method)."""
        import random
        return random.sample(self.track_names, min(count, len(self.track_names)))
    
    def rebuild_index(self):
        """Rebuild the FAISS index from scratch."""
        logging.info("Rebuilding FAISS index...")
        self._build_index()
        
        # Update cache if enabled
        if self.cache:
            self.cache.save_index(self.faiss_index, self.feature_type)
        
        logging.info("FAISS index rebuilt successfully")
    
    def get_feature_vector(self, track_name: str) -> np.ndarray:
        """
        Get the feature vector for a specific track.
        
        Args:
            track_name: Name of the track
            
        Returns:
            Feature vector as numpy array
        """
        if track_name not in self.features:
            raise ValueError(f"Track '{track_name}' not found in features")
        
        return self.features[track_name].copy()
    
    def get_stats(self) -> dict:
        """Get statistics about the search engine."""
        return {
            "total_tracks": len(self.track_names),
            "feature_type": self.feature_type,
            "feature_dimension": len(next(iter(self.features.values()))) if self.features else 0,
            "index_type": "FAISS IndexFlatIP" if self.faiss_index else "Not initialized",
            "cache_enabled": self.cache is not None
        }