"""
Simple Pipeline Service Client
"""

import httpx
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PipelineClient:
    """Simple client for ML Pipeline Service."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")
    
    async def health_check(self) -> bool:
        """Check if pipeline service is healthy."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
            except:
                return False
    
    async def get_recommendations(
        self, 
        track_id: str, 
        strategy: str = "similarity",
        n_recommendations: int = 10
    ) -> Dict[str, Any]:
        """Get recommendations from pipeline service."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/recommend",
                    json={"track_id": track_id, "n_recommendations": n_recommendations}
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Pipeline service error: {e}")
                return {
                    "reference_track": track_id,
                    "recommendations": [],
                    "error": str(e)
                }
    
    async def natural_language_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/query",
                    json={"query": query}
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Query failed: {e}")
                return {"query": query, "recommendations": [], "error": str(e)}


# Global pipeline client instance
pipeline_client: Optional[PipelineClient] = None


def get_pipeline_client() -> PipelineClient:
    """Get or create pipeline client instance."""
    global pipeline_client
    if pipeline_client is None:
        pipeline_client = PipelineClient()
    return pipeline_client