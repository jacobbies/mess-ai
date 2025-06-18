# src/mess_ai/models/recommender.py
import random
import torch
import numpy as np
from pathlib import Path
import os
import logging
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional


class MusicRecommender:
    def __init__(self, features_dir="data/processed/features", feature_type="aggregated"):
        """
        Initialize music recommender with precomputed MERT features.
        
        Args:
            features_dir: Directory containing extracted features
            feature_type: Type of features to use ('aggregated', 'segments', 'raw')
        """
        self.features_dir = Path(features_dir)
        self.feature_type = feature_type
        self.features_cache = {}
        self.track_names = []
        
        # Load features on initialization
        self._load_features()
    
    def _load_features(self):
        """Load all precomputed features into memory."""
        try:
            feature_path = self.features_dir / self.feature_type
            if not feature_path.exists():
                raise FileNotFoundError(f"Features directory not found: {feature_path}")
            
            feature_files = list(feature_path.glob("*.npy"))
            if not feature_files:
                raise FileNotFoundError(f"No feature files found in {feature_path}")
            
            logging.info(f"Loading {len(feature_files)} feature files...")
            
            for feature_file in feature_files:
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
                
                self.features_cache[track_name] = features
                self.track_names.append(track_name)
            
            logging.info(f"Loaded features for {len(self.track_names)} tracks")
            logging.info(f"Feature dimensionality: {features.shape[0]}")
            
        except Exception as e:
            logging.error(f"Failed to load features: {e}")
            raise
    
    def find_similar_tracks(self, track_name: str, top_k: int = 5, exclude_self: bool = True) -> List[Tuple[str, float]]:
        """
        Find tracks similar to the given track.
        
        Args:
            track_name: Name of the reference track (without .npy extension)
            top_k: Number of similar tracks to return
            exclude_self: Whether to exclude the reference track from results
            
        Returns:
            List of (track_name, similarity_score) tuples, sorted by similarity descending
        """
        try:
            if track_name not in self.features_cache:
                available_tracks = list(self.features_cache.keys())[:5]  # Show first 5 for debugging
                raise ValueError(f"Track '{track_name}' not found. Available tracks include: {available_tracks}")
            
            # Get reference track features
            ref_features = self.features_cache[track_name].reshape(1, -1)
            
            # Calculate similarities with all tracks
            similarities = []
            for other_track, other_features in self.features_cache.items():
                if exclude_self and other_track == track_name:
                    continue
                
                other_features = other_features.reshape(1, -1)
                similarity = cosine_similarity(ref_features, other_features)[0, 0]
                similarities.append((other_track, float(similarity)))
            
            # Sort by similarity (descending) and return top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logging.error(f"Error finding similar tracks for '{track_name}': {e}")
            raise
    
    def get_track_names(self) -> List[str]:
        """Get list of all available track names."""
        return self.track_names.copy()
    
    def get_random_recommendations(self, count: int = 5) -> List[str]:
        """Get random track recommendations (fallback method)."""
        return random.sample(self.track_names, min(count, len(self.track_names)))
        