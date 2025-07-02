# src/mess_ai/models/recommender.py
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import sys
import os

# Import from local backend structure
from ..search.similarity import SimilaritySearchEngine


class MusicRecommender:
    def __init__(self, features_dir="data/processed/features", feature_type="aggregated"):
        """
        Initialize music recommender with FAISS-powered similarity search.
        
        Args:
            features_dir: Directory containing extracted features
            feature_type: Type of features to use ('aggregated', 'segments', 'raw')
        """
        self.features_dir = Path(features_dir)
        self.feature_type = feature_type
        
        # Initialize FAISS-based search engine
        cache_dir = str(self.features_dir.parent / "cache" / "faiss")
        self.search_engine = SimilaritySearchEngine(
            features_dir=str(self.features_dir),
            feature_type=feature_type,
            cache_dir=cache_dir
        )
        
        logging.info(f"Music recommender initialized with FAISS search engine")
    
    def find_similar_tracks(self, track_name: str, top_k: int = 5, exclude_self: bool = True) -> List[Tuple[str, float]]:
        """
        Find tracks similar to the given track using FAISS.
        
        Args:
            track_name: Name of the reference track (without .npy extension)
            top_k: Number of similar tracks to return
            exclude_self: Whether to exclude the reference track from results
            
        Returns:
            List of (track_name, similarity_score) tuples, sorted by similarity descending
        """
        try:
            return self.search_engine.search(track_name, top_k, exclude_self)
        except Exception as e:
            logging.error(f"Error finding similar tracks for '{track_name}': {e}")
            raise
    
    def get_track_names(self) -> List[str]:
        """Get list of all available track names."""
        return self.search_engine.get_track_names()
    
    def get_random_recommendations(self, count: int = 5) -> List[str]:
        """Get random track recommendations (fallback method)."""
        return self.search_engine.get_random_recommendations(count)
    
    def get_stats(self) -> dict:
        """Get statistics about the recommender system."""
        return self.search_engine.get_stats()
        