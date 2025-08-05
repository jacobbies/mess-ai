"""
Music recommendation API routes.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

from core.dependencies import RecommendationServiceDep
from core.services.async_recommendation_service import AsyncRecommendationService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommend", tags=["recommendations"])


@router.get("/{track_name}")
async def get_recommendations(
    track_name: str, 
    top_k: int = Query(5, ge=1, le=50, description="Number of recommendations to return"),
    strategy: Optional[str] = Query(
        "similar",
        description="Recommendation strategy: similar (default), balanced, diverse, complementary, exploration"
    ),
    recommendation_service: AsyncRecommendationService = RecommendationServiceDep
):
    """
    Get music recommendations based on MERT feature similarity with metadata.
    
    Strategies:
    - **similar**: Traditional similarity-based recommendations (highest cosine similarity)
    - **balanced**: Balanced relevance and diversity using MMR (λ=0.7)
    - **diverse**: More diverse recommendations using MMR (λ=0.5)
    - **complementary**: Moderately similar tracks (0.5-0.75 similarity range)
    - **exploration**: Discovery-oriented recommendations (around 0.6 similarity)
    """
    try:
        if not recommendation_service.is_available():
            raise HTTPException(status_code=503, detail="Recommender not available - features not loaded")
        
        # Map old strategy names to new ones
        strategy_mapping = {
            "similar": "similarity",
            "balanced": "hybrid",
            "diverse": "diverse",
            "complementary": "diverse",
            "exploration": "random"
        }
        
        mapped_strategy = strategy_mapping.get(strategy, strategy)
        
        # Use diverse mode for specific strategies
        mode = None
        if strategy == "diverse":
            mode = "mmr"
        elif strategy == "complementary":
            mode = "cluster"
        
        return await recommendation_service.get_recommendations(
            track_name, 
            top_k, 
            strategy=mapped_strategy,
            mode=mode
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{track_name}/compare")
async def compare_recommendation_strategies(
    track_name: str,
    top_k: int = Query(5, ge=1, le=20, description="Number of recommendations per strategy"),
    recommendation_service: AsyncRecommendationService = RecommendationServiceDep
):
    """
    Compare different recommendation strategies for a given track.
    
    Returns recommendations from all available strategies along with diversity metrics.
    """
    try:
        if not recommendation_service.is_available():
            raise HTTPException(status_code=503, detail="Recommender not available - features not loaded")
        
        return await recommendation_service.compare_strategies(track_name, top_k)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Strategy comparison error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")