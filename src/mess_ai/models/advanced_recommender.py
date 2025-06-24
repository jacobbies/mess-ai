"""
Advanced music recommender with diverse recommendation strategies.

This module extends the basic recommender with multiple recommendation types
to provide more varied and interesting music suggestions.
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Literal
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mess_ai.search.similarity import SimilaritySearchEngine
from mess_ai.search.diverse_similarity import DiverseSimilaritySearch
from mess_ai.analysis.embedding_analyzer import EmbeddingAnalyzer


RecommendationType = Literal[
    'similar',      # Traditional most similar tracks
    'diverse',      # Similar but with diversity promotion
    'contrasting',  # Intentionally different but related
    'exploratory',  # Samples from different regions
    'complementary', # Focus on different feature aspects
    'surprise',     # Mix of different strategies
]


class AdvancedMusicRecommender:
    """Enhanced recommender with multiple recommendation strategies."""
    
    def __init__(self, features_dir: str = "data/processed/features", 
                 feature_type: str = "aggregated"):
        """Initialize advanced recommender with all components."""
        self.features_dir = Path(features_dir)
        self.feature_type = feature_type
        
        # Initialize base search engine
        cache_dir = str(self.features_dir.parent / "cache" / "faiss")
        self.search_engine = SimilaritySearchEngine(
            features_dir=str(self.features_dir),
            feature_type=feature_type,
            cache_dir=cache_dir
        )
        
        # Load embeddings for advanced search
        self._load_embeddings()
        
        # Initialize advanced search components
        self.diverse_search = DiverseSimilaritySearch(self.embeddings)
        self.analyzer = EmbeddingAnalyzer(str(self.features_dir))
        
        logging.info("Advanced music recommender initialized")
        
    def _load_embeddings(self):
        """Load embeddings from search engine."""
        self.embeddings = {}
        for track_name in self.search_engine.track_names:
            self.embeddings[track_name] = self.search_engine.get_feature_vector(track_name)
            
    def get_recommendations(self,
                          track_name: str,
                          recommendation_type: RecommendationType = 'similar',
                          n_results: int = 5,
                          **kwargs) -> List[Tuple[str, float, Dict]]:
        """
        Get recommendations using specified strategy.
        
        Args:
            track_name: Reference track
            recommendation_type: Type of recommendations to generate
            n_results: Number of recommendations
            **kwargs: Additional parameters for specific recommendation types
            
        Returns:
            List of (track_name, score, metadata) tuples
        """
        if track_name not in self.embeddings:
            raise ValueError(f"Track '{track_name}' not found")
            
        if recommendation_type == 'similar':
            return self._get_similar_recommendations(track_name, n_results, **kwargs)
        elif recommendation_type == 'diverse':
            return self._get_diverse_recommendations(track_name, n_results, **kwargs)
        elif recommendation_type == 'contrasting':
            return self._get_contrasting_recommendations(track_name, n_results, **kwargs)
        elif recommendation_type == 'exploratory':
            return self._get_exploratory_recommendations(track_name, n_results, **kwargs)
        elif recommendation_type == 'complementary':
            return self._get_complementary_recommendations(track_name, n_results, **kwargs)
        elif recommendation_type == 'surprise':
            return self._get_surprise_recommendations(track_name, n_results, **kwargs)
        else:
            raise ValueError(f"Unknown recommendation type: {recommendation_type}")
            
    def _get_similar_recommendations(self, track_name: str, n_results: int, 
                                   metric: str = 'cosine', **kwargs) -> List[Tuple[str, float, Dict]]:
        """Traditional similarity-based recommendations."""
        results = self.diverse_search.search_similar(track_name, metric, n_results)
        
        # Add metadata
        enhanced_results = []
        for track, score in results:
            metadata = {
                'type': 'similar',
                'metric': metric,
                'description': f"High {metric} similarity",
            }
            enhanced_results.append((track, score, metadata))
            
        return enhanced_results
        
    def _get_diverse_recommendations(self, track_name: str, n_results: int,
                                   diversity_weight: float = 0.3, **kwargs) -> List[Tuple[str, float, Dict]]:
        """Diverse recommendations using MMR."""
        results = self.diverse_search.search_diverse(
            track_name, n_results, diversity_weight
        )
        
        enhanced_results = []
        for i, (track, score) in enumerate(results):
            metadata = {
                'type': 'diverse',
                'diversity_weight': diversity_weight,
                'description': 'Similar but diverse selection',
                'rank': i + 1,
            }
            enhanced_results.append((track, score, metadata))
            
        return enhanced_results
        
    def _get_contrasting_recommendations(self, track_name: str, n_results: int,
                                       contrast_range: Tuple[float, float] = (0.4, 0.7),
                                       **kwargs) -> List[Tuple[str, float, Dict]]:
        """Contrasting but related recommendations."""
        results = self.diverse_search.search_contrasting(
            track_name, n_results, contrast_range
        )
        
        enhanced_results = []
        for track, distance in results:
            metadata = {
                'type': 'contrasting',
                'distance': distance,
                'description': f"Contrasting style (distance: {distance:.2f})",
                'contrast_level': 'moderate' if distance < 0.55 else 'strong',
            }
            # Convert distance to similarity for consistency
            similarity = 1 - distance
            enhanced_results.append((track, similarity, metadata))
            
        return enhanced_results
        
    def _get_exploratory_recommendations(self, track_name: str, n_results: int,
                                       exploration_clusters: int = 3, **kwargs) -> List[Tuple[str, float, Dict]]:
        """Exploratory recommendations from different regions."""
        results = self.diverse_search.search_exploratory(
            track_name, n_results, exploration_clusters
        )
        
        # Get cluster information
        clusters = self.analyzer.cluster_embeddings(n_clusters=exploration_clusters)
        ref_idx = self.analyzer.track_names.index(track_name)
        ref_cluster = clusters['labels'][ref_idx]
        
        enhanced_results = []
        for track, score in results:
            track_idx = self.analyzer.track_names.index(track)
            track_cluster = clusters['labels'][track_idx]
            
            metadata = {
                'type': 'exploratory',
                'cluster': int(track_cluster),
                'from_same_cluster': track_cluster == ref_cluster,
                'description': f"From cluster {track_cluster}" + 
                             (" (same)" if track_cluster == ref_cluster else " (different)"),
            }
            enhanced_results.append((track, score, metadata))
            
        return enhanced_results
        
    def _get_complementary_recommendations(self, track_name: str, n_results: int,
                                         **kwargs) -> List[Tuple[str, float, Dict]]:
        """Complementary recommendations focusing on different features."""
        results = self.diverse_search.search_complementary(track_name, n_results)
        
        enhanced_results = []
        for track, score in results:
            metadata = {
                'type': 'complementary',
                'description': 'Complementary musical features',
                'feature_focus': 'inverse variance weighted',
            }
            enhanced_results.append((track, score, metadata))
            
        return enhanced_results
        
    def _get_surprise_recommendations(self, track_name: str, n_results: int,
                                    **kwargs) -> List[Tuple[str, float, Dict]]:
        """Surprise recommendations mixing different strategies."""
        # Mix different recommendation types
        strategies = [
            ('similar', 2),
            ('diverse', 1),
            ('contrasting', 1),
            ('exploratory', 1),
        ]
        
        all_recommendations = []
        
        for strategy, count in strategies:
            if count >= n_results - len(all_recommendations):
                count = n_results - len(all_recommendations)
                
            if count > 0:
                recs = self.get_recommendations(track_name, strategy, count)
                all_recommendations.extend(recs)
                
            if len(all_recommendations) >= n_results:
                break
                
        # Shuffle for surprise element
        import random
        random.shuffle(all_recommendations)
        
        return all_recommendations[:n_results]
        
    def analyze_recommendations(self, track_name: str, 
                              recommendation_type: RecommendationType = 'similar',
                              n_results: int = 5) -> Dict:
        """
        Analyze the quality and diversity of recommendations.
        
        Returns detailed analysis including diversity metrics, feature distributions, etc.
        """
        # Get recommendations
        recommendations = self.get_recommendations(track_name, recommendation_type, n_results)
        rec_tracks = [r[0] for r in recommendations]
        
        # Get embeddings
        ref_embedding = self.embeddings[track_name]
        rec_embeddings = np.array([self.embeddings[t] for t in rec_tracks])
        
        # Calculate diversity metrics
        from sklearn.metrics.pairwise import pairwise_distances
        
        # Pairwise distances among recommendations
        internal_distances = pairwise_distances(rec_embeddings, metric='cosine')
        avg_internal_distance = np.mean(internal_distances[np.triu_indices_from(internal_distances, k=1)])
        
        # Distances to reference
        ref_distances = pairwise_distances([ref_embedding], rec_embeddings, metric='cosine')[0]
        
        # Feature variance
        feature_variance = np.var(rec_embeddings, axis=0)
        avg_feature_variance = np.mean(feature_variance)
        
        # Coverage of embedding space
        pca_2d = self.analyzer.reduce_dimensions(method='pca', n_components=2)
        ref_idx = self.analyzer.track_names.index(track_name)
        ref_pca = pca_2d[ref_idx]
        
        rec_indices = [self.analyzer.track_names.index(t) for t in rec_tracks]
        rec_pca = pca_2d[rec_indices]
        
        # Calculate spread in PCA space
        pca_spread = np.std(rec_pca, axis=0)
        
        analysis = {
            'recommendation_type': recommendation_type,
            'n_recommendations': len(recommendations),
            'diversity_metrics': {
                'avg_internal_distance': float(avg_internal_distance),
                'min_internal_distance': float(np.min(internal_distances[internal_distances > 0])),
                'max_internal_distance': float(np.max(internal_distances)),
                'avg_distance_to_reference': float(np.mean(ref_distances)),
                'distance_range': (float(np.min(ref_distances)), float(np.max(ref_distances))),
            },
            'feature_metrics': {
                'avg_feature_variance': float(avg_feature_variance),
                'top_varying_features': np.argsort(feature_variance)[-10:].tolist(),
            },
            'space_coverage': {
                'pca_spread': pca_spread.tolist(),
                'relative_spread': (pca_spread / np.std(pca_2d, axis=0)).tolist(),
            },
            'recommendations': [
                {
                    'track': track,
                    'score': float(score),
                    'metadata': metadata,
                    'distance_to_ref': float(ref_distances[i]),
                }
                for i, (track, score, metadata) in enumerate(recommendations)
            ],
        }
        
        return analysis
        
    def get_recommendation_explanation(self, 
                                     track_name: str,
                                     recommended_track: str,
                                     recommendation_type: RecommendationType) -> str:
        """Generate human-readable explanation for why a track was recommended."""
        
        explanations = {
            'similar': f"{recommended_track} shares very similar musical characteristics with {track_name}, "
                      f"including comparable tonal patterns and rhythmic structures.",
            
            'diverse': f"{recommended_track} is musically related to {track_name} but offers enough variety "
                      f"to keep your listening experience fresh and interesting.",
            
            'contrasting': f"{recommended_track} provides an interesting contrast to {track_name} while "
                          f"maintaining some musical connection, perfect for expanding your taste.",
            
            'exploratory': f"{recommended_track} comes from a different musical region than {track_name}, "
                          f"helping you discover new styles within the classical repertoire.",
            
            'complementary': f"{recommended_track} complements {track_name} by emphasizing different "
                           f"musical features, creating a well-rounded listening experience.",
            
            'surprise': f"{recommended_track} was selected using a mix of strategies to surprise you "
                       f"with something unexpected yet enjoyable based on {track_name}.",
        }
        
        return explanations.get(recommendation_type, 
                              f"{recommended_track} was recommended based on {recommendation_type} strategy.")