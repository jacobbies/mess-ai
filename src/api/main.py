"""
FastAPI Music Similarity API - Main application entry point.
"""
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.config import settings
from core.dependencies import initialize_services
from mess_ai.audio.player import MusicLibrary
from mess_ai.models.recommender import MusicRecommender
from mess_ai.metadata.processor import MetadataProcessor
from mess_ai.models.metadata import TrackMetadata

# Import routers
from api.routers import tracks, recommendations, audio, metadata

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Global state
metadata_dict: Dict[str, TrackMetadata] = {}
library = None
recommender = None
executor = ThreadPoolExecutor(max_workers=settings.THREAD_POOL_MAX_WORKERS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global metadata_dict, library, recommender
    
    logger.info("Starting Music Similarity API...")
    
    # Initialize metadata processor
    try:
        metadata_processor = MetadataProcessor()
        metadata_dict = metadata_processor.load_metadata_dict()
        logger.info(f"Loaded metadata for {len(metadata_dict)} tracks")
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        metadata_dict = {}
    
    # Initialize audio library
    try:
        library = MusicLibrary(wav_dir=str(settings.wav_dir))
        logger.info("Audio library initialized")
    except Exception as e:
        logger.error(f"Failed to initialize audio library: {e}")
    
    # Initialize recommender
    try:
        # Try to use diverse recommender first
        try:
            from mess_ai.models.diverse_recommender import DiverseMusicRecommender
            recommender = DiverseMusicRecommender(features_dir=str(settings.features_dir))
            logger.info("Diverse music recommender initialized successfully with FAISS")
        except ImportError:
            # Fallback to standard recommender
            recommender = MusicRecommender(features_dir=str(settings.features_dir))
            logger.info("Standard music recommender initialized successfully with FAISS")
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {e}")
        recommender = None
    
    # Initialize services with dependency injection
    initialize_services(metadata_dict, library, recommender)
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    executor.shutdown(wait=True)
    logger.info("API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(metadata.router)  # Root endpoints (/, /health, etc.)
app.include_router(tracks.router)
app.include_router(recommendations.router)
app.include_router(audio.router)

# Health and metadata endpoints are included via metadata router

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=True
    )