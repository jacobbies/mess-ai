"""
FastAPI dependency injection configuration.
Provides centralized dependency management for services.
"""
from typing import Dict
from fastapi import Depends
import logging

from .services.metadata_service import MetadataService
from .services.recommendation_service import RecommendationService
from .services.audio_service import AudioService
from .config import settings
from mess_ai.models.metadata import TrackMetadata

logger = logging.getLogger(__name__)

# Global service instances - will be set during app startup
_metadata_service: MetadataService = None
_recommendation_service: RecommendationService = None
_audio_service: AudioService = None


def initialize_services(
    metadata_dict: Dict[str, TrackMetadata],
    library,
    recommender
):
    """
    Initialize all services with their dependencies.
    Called during FastAPI startup.
    """
    global _metadata_service, _recommendation_service, _audio_service
    
    logger.info("Initializing services...")
    
    # Initialize services
    _metadata_service = MetadataService(metadata_dict)
    _recommendation_service = RecommendationService(recommender, metadata_dict)
    _audio_service = AudioService(library, settings.wav_dir, settings.waveforms_dir)
    
    logger.info("All services initialized successfully")


def get_metadata_service() -> MetadataService:
    """Dependency for metadata service."""
    if _metadata_service is None:
        raise RuntimeError("MetadataService not initialized")
    return _metadata_service


def get_recommendation_service() -> RecommendationService:
    """Dependency for recommendation service."""
    if _recommendation_service is None:
        raise RuntimeError("RecommendationService not initialized")
    return _recommendation_service


def get_audio_service() -> AudioService:
    """Dependency for audio service."""
    if _audio_service is None:
        raise RuntimeError("AudioService not initialized")
    return _audio_service


# Convenience functions for getting services
MetadataServiceDep = Depends(get_metadata_service)
RecommendationServiceDep = Depends(get_recommendation_service)
AudioServiceDep = Depends(get_audio_service)