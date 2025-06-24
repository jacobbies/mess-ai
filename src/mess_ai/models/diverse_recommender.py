"""
Enhanced Music Recommender with Diversity Strategies

This module extends the basic recommender with diversity-aware recommendation
strategies that can be easily integrated into the existing system.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mess_ai.models.recommender import MusicRecommender


class DiverseMusicRecommender(MusicRecommender):
    """
    Extended music recommender with diversity strategies.
    
    Inherits from MusicRecommender to maintain compatibility while
    adding new diversity-aware recommendation methods.
    """
    
    def __init__(self, features_dir="data/processed/features", feature_type="aggregated"):
        """Initialize with parent class."""
        super().__init__(features_dir, feature_type)
        
        # Cache for feature vectors to avoid repeated loading
        self._feature_cache = {}
        self._load_all_features()
        
        # Pre-compute clusters for cluster-based diversity
        self._clusters = None
        self._cluster_representatives = None
        
    def _load_all_features(self):
        """Load all feature vectors into memory for faster access."""
        feature_path = self.features_dir / self.feature_type
        
        for track_name in self.get_track_names():
            feature_file = feature_path / f"{track_name}.npy"
            if feature_file.exists():
                features = np.load(feature_file)
                # Flatten aggregated features
                if self.feature_type == "aggregated":
                    features = features.flatten()
                self._feature_cache[track_name] = features
        
        logging.info(f"Loaded {len(self._feature_cache)} feature vectors into cache")
    
    def find_similar_tracks_mmr(self, track_name: str, top_k: int = 5, 
                               lambda_param: float = 0.5, 
                               candidate_pool_size: int = 50) -> List[Tuple[str, float]]:
        """
        Find similar tracks using Maximal Marginal Relevance (MMR).
        
        MMR balances relevance and diversity by iteratively selecting items
        that are similar to the query but dissimilar to already selected items.
        
        Args:
            track_name: Name of the reference track
            top_k: Number of recommendations to return
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
            candidate_pool_size: Size of initial candidate pool
            
        Returns:
            List of (track_name, relevance_score) tuples
        """
        # Get initial candidates using standard similarity
        candidates = self.search_engine.search(track_name, candidate_pool_size, exclude_self=True)
        
        if not candidates:
            return []
        
        # Get query embedding
        query_embedding = self._get_normalized_embedding(track_name)
        
        # Prepare candidate data
        candidate_names = [c[0] for c in candidates]
        candidate_embeddings = np.array([
            self._get_normalized_embedding(name) for name in candidate_names
        ])
        
        # Compute relevance scores (cosine similarity to query)
        relevance_scores = np.dot(candidate_embeddings, query_embedding)
        
        # MMR selection
        selected_indices = []
        selected_embeddings = []
        
        for _ in range(min(top_k, len(candidate_names))):
            mmr_scores = []
            
            for idx, candidate_emb in enumerate(candidate_embeddings):
                if idx in selected_indices:
                    continue
                
                # Relevance term
                relevance = relevance_scores[idx]
                
                # Diversity term (max similarity to already selected items)
                if selected_embeddings:
                    similarities_to_selected = [
                        np.dot(candidate_emb, sel_emb) 
                        for sel_emb in selected_embeddings
                    ]
                    max_similarity = max(similarities_to_selected)
                else:
                    max_similarity = 0.0
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append((idx, mmr_score))
            
            # Select item with highest MMR score
            if mmr_scores:
                best_idx, _ = max(mmr_scores, key=lambda x: x[1])
                selected_indices.append(best_idx)
                selected_embeddings.append(candidate_embeddings[best_idx])
        
        # Return selected items with their original relevance scores
        results = [
            (candidate_names[idx], float(relevance_scores[idx]))
            for idx in selected_indices
        ]
        
        return results
    
    def find_similar_tracks_diverse(self, track_name: str, top_k: int = 5,
                                   diversity_weight: float = 0.3) -> List[Tuple[str, float]]:
        """
        Find similar tracks with built-in diversity.
        
        This is a simplified version of MMR that's easier to tune.
        
        Args:
            track_name: Name of the reference track
            top_k: Number of recommendations to return  
            diversity_weight: Weight for diversity (0=pure similarity, 1=pure diversity)
            
        Returns:
            List of (track_name, score) tuples
        """
        return self.find_similar_tracks_mmr(
            track_name, 
            top_k=top_k, 
            lambda_param=1.0 - diversity_weight
        )
    
    def find_complementary_tracks(self, track_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find tracks that are complementary (moderately similar, not too close).
        
        Good for creating playlists with variety while maintaining coherence.
        
        Args:
            track_name: Name of the reference track
            top_k: Number of recommendations to return
            
        Returns:
            List of (track_name, score) tuples
        """
        # Get a large pool of candidates
        candidates = self.search_engine.search(track_name, top_k=100, exclude_self=True)
        
        # Filter for moderate similarity (0.5 - 0.75 range)
        complementary = [
            (name, score) for name, score in candidates
            if 0.5 <= score <= 0.75
        ]
        
        # If not enough, expand the range
        if len(complementary) < top_k:
            complementary = [
                (name, score) for name, score in candidates
                if 0.4 <= score <= 0.8
            ]
        
        # Sort by how close to the "sweet spot" of 0.65
        complementary.sort(key=lambda x: abs(x[1] - 0.65))
        
        return complementary[:top_k]
    
    def find_exploration_tracks(self, track_name: str, top_k: int = 5,
                               exploration_radius: float = 0.6) -> List[Tuple[str, float]]:
        """
        Find tracks for exploration (somewhat related but different).
        
        Args:
            track_name: Name of the reference track
            top_k: Number of recommendations to return
            exploration_radius: Similarity threshold for exploration
            
        Returns:
            List of (track_name, score) tuples
        """
        # Get query embedding
        query_embedding = self._get_normalized_embedding(track_name)
        
        # Find all tracks within exploration radius
        exploration_candidates = []
        
        for candidate_name, candidate_embedding in self._feature_cache.items():
            if candidate_name == track_name:
                continue
            
            # Compute similarity
            candidate_norm = candidate_embedding / np.linalg.norm(candidate_embedding)
            similarity = np.dot(query_embedding, candidate_norm)
            
            # Check if in exploration range
            if exploration_radius - 0.1 <= similarity <= exploration_radius + 0.1:
                exploration_candidates.append((candidate_name, similarity))
        
        # Sort by similarity and return top k
        exploration_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return exploration_candidates[:top_k]
    
    def get_recommendations(self, track_name: str, strategy: str = "balanced",
                           top_k: int = 5, **kwargs) -> List[Tuple[str, float]]:
        """
        Get recommendations using specified strategy.
        
        Strategies:
        - 'similar': Traditional similarity-based (default)
        - 'balanced': MMR with lambda=0.7 (good default)
        - 'diverse': MMR with lambda=0.5 (more diversity)  
        - 'complementary': Moderately similar tracks
        - 'exploration': Tracks for discovery
        
        Args:
            track_name: Name of the reference track
            strategy: Recommendation strategy to use
            top_k: Number of recommendations to return
            **kwargs: Additional arguments for specific strategies
            
        Returns:
            List of (track_name, score) tuples
        """
        if strategy == "similar":
            return self.find_similar_tracks(track_name, top_k)
        elif strategy == "balanced":
            return self.find_similar_tracks_mmr(track_name, top_k, lambda_param=0.7)
        elif strategy == "diverse":
            return self.find_similar_tracks_mmr(track_name, top_k, lambda_param=0.5)
        elif strategy == "complementary":
            return self.find_complementary_tracks(track_name, top_k)
        elif strategy == "exploration":
            return self.find_exploration_tracks(track_name, top_k, **kwargs)
        else:
            logging.warning(f"Unknown strategy '{strategy}', falling back to 'similar'")
            return self.find_similar_tracks(track_name, top_k)
    
    def _get_normalized_embedding(self, track_name: str) -> np.ndarray:
        """Get L2-normalized embedding for a track."""
        if track_name not in self._feature_cache:
            raise ValueError(f"Track '{track_name}' not found in feature cache")
        
        embedding = self._feature_cache[track_name]
        norm = np.linalg.norm(embedding)
        
        if norm > 0:
            return embedding / norm
        else:
            return embedding
    
    def compute_diversity_score(self, recommendations: List[str]) -> float:
        """
        Compute diversity score for a list of recommendations.
        
        Args:
            recommendations: List of track names
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(recommendations) < 2:
            return 0.0
        
        # Get embeddings
        embeddings = [self._get_normalized_embedding(track) for track in recommendations]
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        
        return diversity
    
    def get_strategy_comparison(self, track_name: str, top_k: int = 5) -> Dict[str, Dict]:
        """
        Compare different recommendation strategies for a given track.
        
        Args:
            track_name: Name of the reference track
            top_k: Number of recommendations per strategy
            
        Returns:
            Dictionary with strategy names as keys and recommendation info as values
        """
        strategies = ["similar", "balanced", "diverse", "complementary", "exploration"]
        comparison = {}
        
        for strategy in strategies:
            try:
                recommendations = self.get_recommendations(track_name, strategy, top_k)
                rec_names = [r[0] for r in recommendations]
                
                comparison[strategy] = {
                    "recommendations": recommendations,
                    "diversity_score": float(self.compute_diversity_score(rec_names)),
                    "avg_similarity": float(np.mean([r[1] for r in recommendations])) if recommendations else 0.0
                }
            except Exception as e:
                logging.error(f"Error with {strategy} strategy: {e}")
                comparison[strategy] = {
                    "recommendations": [],
                    "diversity_score": 0.0,
                    "avg_similarity": 0.0
                }
        
        return comparison


# Convenience function for easy integration
def create_diverse_recommender(features_dir="data/processed/features", 
                              feature_type="aggregated") -> DiverseMusicRecommender:
    """
    Create a diverse music recommender instance.
    
    Args:
        features_dir: Directory containing extracted features
        feature_type: Type of features to use
        
    Returns:
        DiverseMusicRecommender instance
    """
    return DiverseMusicRecommender(features_dir, feature_type)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize recommender
    recommender = create_diverse_recommender()
    
    # Get some track names
    tracks = recommender.get_track_names()
    if tracks:
        test_track = tracks[0]
        print(f"\nTesting with track: {test_track}\n")
        
        # Compare strategies
        comparison = recommender.get_strategy_comparison(test_track, top_k=5)
        
        for strategy, results in comparison.items():
            print(f"\n=== {strategy.upper()} Strategy ===")
            print(f"Diversity Score: {results['diversity_score']:.3f}")
            print(f"Avg Similarity: {results['avg_similarity']:.3f}")
            print("Recommendations:")
            for track, score in results['recommendations']:
                print(f"  - {track}: {score:.3f}")