"""
ML models and recommendation engines
"""

from .metadata import TrackMetadata
from .recommender import MusicRecommender
from .unified_recommender import UnifiedMusicRecommender
from .async_unified_recommender import AsyncUnifiedMusicRecommender, RecommendationResult, RecommendationRequest

try:
    from .diverse_recommender import DiverseMusicRecommender
    __all__ = [
        "TrackMetadata", 
        "MusicRecommender", 
        "DiverseMusicRecommender", 
        "UnifiedMusicRecommender",
        "AsyncUnifiedMusicRecommender",
        "RecommendationResult",
        "RecommendationRequest"
    ]
except ImportError:
    __all__ = [
        "TrackMetadata", 
        "MusicRecommender", 
        "UnifiedMusicRecommender",
        "AsyncUnifiedMusicRecommender",
        "RecommendationResult",
        "RecommendationRequest"
    ]