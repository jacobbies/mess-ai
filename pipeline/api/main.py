"""
Simple ML Pipeline Service API
Minimal wrapper around our ML functionality.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

from query.layer_based_recommender import LayerBasedRecommender
from query.intelligent_query_engine import IntelligentQueryEngine

app = FastAPI(title="ML Pipeline", version="1.0.0")
logger = logging.getLogger(__name__)

# Global instances
recommender = None
query_engine = None


@app.on_event("startup")
async def startup():
    global recommender, query_engine
    recommender = LayerBasedRecommender()
    query_engine = IntelligentQueryEngine()


@app.get("/health")
async def health():
    return {"status": "ok"}


class RecommendRequest(BaseModel):
    track_id: str
    n_recommendations: int = 5


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """Get recommendations using empirically validated layers."""
    if not recommender:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        results = recommender.recommend_by_aspect(
            request.track_id, 
            "spectral_brightness",  # Use our best validated aspect
            request.n_recommendations
        )
        
        return {
            "reference_track": request.track_id,
            "recommendations": [
                {"track": track, "similarity": sim}
                for track, sim, _ in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def natural_query(request: QueryRequest):
    """Natural language query."""
    if not query_engine:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        intent, recommendations = query_engine.execute_query(request.query)
        return {
            "query": request.query,
            "recommendations": [
                {"track": track, "similarity": sim}
                for track, sim, _ in recommendations
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)