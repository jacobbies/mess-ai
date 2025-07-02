"""
Unified Music Recommender with Multiple Strategy Support

A single interface that allows developers to easily switch between different 
recommendation algorithms and strategies.
"""
import logging
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from pathlib import Path

from .recommender import MusicRecommender as BaseRecommender
from ..search.similarity import SimilaritySearchEngine


class RecommendationStrategy(Enum):
    """Available recommendation strategies."""
    SIMILARITY = "similarity"          # Basic cosine similarity
    DIVERSE = "diverse"               # Diverse recommendations (MMR, cluster-based)
    POPULAR = "popular"               # Popularity-based
    RANDOM = "random"                 # Random recommendations
    HYBRID = "hybrid"                 # Combination of strategies


class RecommendationMode(Enum):
    """Recommendation modes for diverse strategies."""
    MMR = "mmr"                      # Maximal Marginal Relevance
    CLUSTER = "cluster"              # Cluster-based diversity
    NOVELTY = "novelty"              # Novel/less popular items
    SERENDIPITY = "serendipity"      # Unexpected but relevant
    CONTRAST = "contrast"            # Deliberately different


class UnifiedMusicRecommender:
    """
    Unified interface for music recommendations with multiple strategies.
    
    Allows developers to easily switch between different recommendation
    algorithms without changing their code.
    
    Example:
        recommender = UnifiedMusicRecommender()
        
        # Basic similarity
        recs = recommender.recommend("track_id", strategy="similarity")
        
        # Diverse recommendations
        recs = recommender.recommend("track_id", strategy="diverse", mode="mmr")
        
        # Hybrid approach
        recs = recommender.recommend("track_id", strategy="hybrid")
    """
    
    def __init__(self, features_dir: str = "data/processed/features", **kwargs):
        self.features_dir = Path(features_dir)
        self.logger = logging.getLogger(__name__)
        
        # Initialize base components
        self._base_recommender = None
        self._diverse_recommender = None
        self._similarity_engine = None
        
        # Strategy configurations
        self._strategy_configs = {
            RecommendationStrategy.SIMILARITY: {},
            RecommendationStrategy.DIVERSE: {"mode": RecommendationMode.MMR, "lambda_param": 0.5},
            RecommendationStrategy.POPULAR: {"boost_factor": 1.5},
            RecommendationStrategy.RANDOM: {"seed": None},
            RecommendationStrategy.HYBRID: {"weights": {"similarity": 0.6, "diverse": 0.4}}
        }
        
        # Update configs with any provided kwargs
        for strategy, config in kwargs.items():
            if hasattr(RecommendationStrategy, strategy.upper()):
                self._strategy_configs[RecommendationStrategy(strategy)] = config
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize recommendation components."""
        try:
            # Base similarity recommender
            self._base_recommender = BaseRecommender(
                features_dir=str(self.features_dir)
            )
            self._similarity_engine = self._base_recommender.search_engine
            self.logger.info("Base recommender initialized")
            
            # Try to initialize diverse recommender
            try:
                from .diverse_recommender import DiverseMusicRecommender
                self._diverse_recommender = DiverseMusicRecommender(
                    features_dir=str(self.features_dir)
                )
                self.logger.info("Diverse recommender initialized")
            except ImportError:
                self.logger.warning("Diverse recommender not available")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize recommenders: {e}")
            raise
    
    def recommend(
        self, 
        track_id: str, 
        n_recommendations: int = 5,
        strategy: str = "similarity",
        mode: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations using specified strategy.
        
        Args:
            track_id: Track to get recommendations for
            n_recommendations: Number of recommendations to return
            strategy: Recommendation strategy ("similarity", "diverse", "popular", "random", "hybrid")
            mode: Mode for diverse strategy ("mmr", "cluster", "novelty", "serendipity", "contrast")
            **kwargs: Additional parameters for specific strategies
            
        Returns:
            List of (track_id, score) tuples
        """
        try:
            strategy_enum = RecommendationStrategy(strategy.lower())
        except ValueError:
            self.logger.warning(f"Unknown strategy '{strategy}', falling back to similarity")
            strategy_enum = RecommendationStrategy.SIMILARITY
        
        # Get strategy config and merge with kwargs
        config = self._strategy_configs[strategy_enum].copy()
        config.update(kwargs)
        
        # Route to appropriate method
        if strategy_enum == RecommendationStrategy.SIMILARITY:
            return self._similarity_recommendations(track_id, n_recommendations, **config)
        elif strategy_enum == RecommendationStrategy.DIVERSE:
            return self._diverse_recommendations(track_id, n_recommendations, mode, **config)
        elif strategy_enum == RecommendationStrategy.POPULAR:
            return self._popular_recommendations(track_id, n_recommendations, **config)
        elif strategy_enum == RecommendationStrategy.RANDOM:
            return self._random_recommendations(track_id, n_recommendations, **config)
        elif strategy_enum == RecommendationStrategy.HYBRID:
            return self._hybrid_recommendations(track_id, n_recommendations, **config)
        else:
            # Fallback to similarity
            return self._similarity_recommendations(track_id, n_recommendations)
    
    def _similarity_recommendations(
        self, 
        track_id: str, 
        n_recommendations: int, 
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Basic similarity-based recommendations."""
        if not self._base_recommender:
            raise RuntimeError("Base recommender not initialized")
        
        return self._base_recommender.recommend(track_id, n_recommendations)
    
    def _diverse_recommendations(
        self, 
        track_id: str, 
        n_recommendations: int, 
        mode: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Diverse recommendations using various algorithms."""
        if not self._diverse_recommender:
            # Fallback to similarity if diverse recommender not available
            self.logger.warning("Diverse recommender not available, falling back to similarity")
            return self._similarity_recommendations(track_id, n_recommendations)
        
        # Default to MMR if no mode specified
        if mode is None:
            mode = "mmr"
        
        try:
            mode_enum = RecommendationMode(mode.lower())
        except ValueError:
            self.logger.warning(f"Unknown mode '{mode}', falling back to MMR")
            mode_enum = RecommendationMode.MMR
        
        return self._diverse_recommender.get_diverse_recommendations(
            track_id, 
            strategy=mode_enum.value, 
            n_recommendations=n_recommendations,
            **kwargs
        )
    
    def _popular_recommendations(
        self, 
        track_id: str, 
        n_recommendations: int,
        boost_factor: float = 1.5,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Popularity-boosted recommendations."""
        # Get base similarity recommendations
        base_recs = self._similarity_recommendations(track_id, n_recommendations * 2)
        
        # Boost popular items (simple heuristic: higher track IDs = more popular)
        boosted_recs = []
        for track, score in base_recs:
            # Simple popularity boost based on track name/ID patterns
            popularity_boost = 1.0
            if any(keyword in track.lower() for keyword in ['bach', 'mozart', 'beethoven']):
                popularity_boost = boost_factor
            
            boosted_score = score * popularity_boost
            boosted_recs.append((track, boosted_score))
        
        # Sort by boosted score and return top N
        boosted_recs.sort(key=lambda x: x[1], reverse=True)
        return boosted_recs[:n_recommendations]
    
    def _random_recommendations(
        self, 
        track_id: str, 
        n_recommendations: int,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Random recommendations from the dataset."""
        import random
        
        if seed is not None:
            random.seed(seed)
        
        # Get all available tracks
        if self._similarity_engine:
            all_tracks = list(self._similarity_engine.track_names)
            # Remove the query track
            if track_id in all_tracks:
                all_tracks.remove(track_id)
            
            # Sample random tracks
            n_sample = min(n_recommendations, len(all_tracks))
            random_tracks = random.sample(all_tracks, n_sample)
            
            # Assign random scores between 0.3 and 0.8
            return [(track, random.uniform(0.3, 0.8)) for track in random_tracks]
        
        return []
    
    def _hybrid_recommendations(
        self, 
        track_id: str, 
        n_recommendations: int,
        weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Hybrid recommendations combining multiple strategies."""
        if weights is None:
            weights = {"similarity": 0.6, "diverse": 0.4}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Get recommendations from each strategy
        strategy_results = {}
        
        if "similarity" in weights:
            strategy_results["similarity"] = self._similarity_recommendations(
                track_id, n_recommendations * 2
            )
        
        if "diverse" in weights and self._diverse_recommender:
            strategy_results["diverse"] = self._diverse_recommendations(
                track_id, n_recommendations * 2
            )
        
        if "popular" in weights:
            strategy_results["popular"] = self._popular_recommendations(
                track_id, n_recommendations * 2
            )
        
        # Combine results with weighted scores
        combined_scores = {}
        
        for strategy, recs in strategy_results.items():
            weight = weights.get(strategy, 0)
            for track, score in recs:
                if track not in combined_scores:
                    combined_scores[track] = 0
                combined_scores[track] += score * weight
        
        # Sort by combined score and return top N
        final_recs = [(track, score) for track, score in combined_scores.items()]
        final_recs.sort(key=lambda x: x[1], reverse=True)
        
        return final_recs[:n_recommendations]
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available recommendation strategies."""
        strategies = [strategy.value for strategy in RecommendationStrategy]
        
        # Remove diverse if not available
        if not self._diverse_recommender and "diverse" in strategies:
            strategies.remove("diverse")
        
        return strategies
    
    def get_available_modes(self) -> List[str]:
        """Get list of available modes for diverse strategy."""
        if not self._diverse_recommender:
            return []
        return [mode.value for mode in RecommendationMode]
    
    def configure_strategy(self, strategy: str, **config):
        """Configure parameters for a specific strategy."""
        try:
            strategy_enum = RecommendationStrategy(strategy.lower())
            self._strategy_configs[strategy_enum].update(config)
            self.logger.info(f"Updated config for {strategy}: {config}")
        except ValueError:
            self.logger.error(f"Unknown strategy: {strategy}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about available strategies and their configurations."""
        return {
            "available_strategies": self.get_available_strategies(),
            "available_modes": self.get_available_modes(),
            "current_configs": {
                strategy.value: config 
                for strategy, config in self._strategy_configs.items()
            },
            "components_available": {
                "base_recommender": self._base_recommender is not None,
                "diverse_recommender": self._diverse_recommender is not None,
                "similarity_engine": self._similarity_engine is not None
            }
        }


# Convenience aliases for backward compatibility
MusicRecommenderUnified = UnifiedMusicRecommender
FlexibleRecommender = UnifiedMusicRecommender