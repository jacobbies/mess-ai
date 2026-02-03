from pydantic import BaseModel
from typing import List


class StreamResponse(BaseModel):
    url: str
    track_id: str
    expires_in: int = 600


class RecommendationItem(BaseModel):
    track_id: str
    similarity_score: float


class RecommendationResponse(BaseModel):
    query_track_id: str
    recommendations: List[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    s3_connected: bool
    version: str