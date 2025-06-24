"""
Metadata and system information API routes.
"""
from fastapi import APIRouter
import logging

from core.config import settings
from core.dependencies import MetadataServiceDep, AudioServiceDep, RecommendationServiceDep
from core.services.metadata_service import MetadataService
from core.services.audio_service import AudioService
from core.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["metadata"])


@router.get("/")
async def root(metadata_service: MetadataService = MetadataServiceDep):
    """API root endpoint with service information."""
    return {
        "service": settings.API_TITLE,
        "description": settings.API_DESCRIPTION,
        "version": settings.API_VERSION,
        "endpoints": {
            "tracks": "/tracks - Get all tracks with metadata",
            "recommend": "/recommend/{track_name} - Get similar tracks",
            "audio": "/audio/{filename} - Stream audio files", 
            "waveform": "/waveform/{filename} - Get waveform images",
            "composers": "/composers - List all composers",
            "tags": "/tags - List all tags",
            "health": "/health - Service health check",
            "docs": "/docs - Interactive API documentation"
        },
        "dataset": settings.DATASET_NAME,
        "total_tracks": metadata_service.get_track_count()
    }


@router.get("/composers")
async def get_composers(metadata_service: MetadataService = MetadataServiceDep):
    """Get list of all composers with track counts."""
    composers = metadata_service.get_composers()
    
    return {
        "composers": composers,
        "count": len(composers)
    }


@router.get("/tags")
async def get_tags(metadata_service: MetadataService = MetadataServiceDep):
    """Get all unique tags with counts."""
    tags = metadata_service.get_tags()
    
    return {
        "tags": tags,
        "count": len(tags)
    }


@router.get("/health")
async def health_check(
    metadata_service: MetadataService = MetadataServiceDep,
    audio_service: AudioService = AudioServiceDep,
    recommendation_service: RecommendationService = RecommendationServiceDep
):
    """Health check endpoint for monitoring."""
    audio_stats = audio_service.get_library_stats()
    faiss_status = recommendation_service.get_faiss_status()
    
    return {
        "status": "healthy",
        "library_loaded": audio_stats["library_loaded"],
        "recommender_loaded": recommendation_service.is_available(),
        "metadata_loaded": metadata_service.get_track_count() > 0,
        "tracks_count": audio_stats["total_audio_files"],
        "metadata_count": metadata_service.get_track_count(),
        "faiss_ready": faiss_status["available"],
        "faiss_details": faiss_status
    }