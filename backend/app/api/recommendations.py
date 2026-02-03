from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import RecommendationResponse, RecommendationItem
from app.core.config import settings

router = APIRouter()


@router.get("/recommendations/{track_id}", response_model=RecommendationResponse)
async def get_recommendations(
    track_id: str,
    count: int = Query(
        default=settings.DEFAULT_RECOMMENDATION_COUNT,
        ge=1,
        le=settings.MAX_RECOMMENDATION_COUNT
    )
):
    # TODO: Implement FAISS similarity search
    # For now, return mock data
    mock_recommendations = [
        RecommendationItem(
            track_id=f"Bach_BWV{850 + i}-01_001_20090916-SMD",
            similarity_score=0.95 - (i * 0.05)
        )
        for i in range(count)
    ]

    return RecommendationResponse(
        query_track_id=track_id,
        recommendations=mock_recommendations
    )