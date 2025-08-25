"""
Simple Natural Language Query Router
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.services.pipeline_client import get_pipeline_client, PipelineClient

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    query: str


@router.post("/")
async def natural_language_query(
    request: QueryRequest,
    pipeline: PipelineClient = Depends(get_pipeline_client)
):
    """Process natural language query for music recommendations."""
    return await pipeline.natural_language_query(request.query)