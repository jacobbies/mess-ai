"""
FastAPI Music Similarity API - Main application entry point.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from core import settings
from core.dependencies import initialize_services
from mess_ai.audio import MusicLibrary
from mess_ai.models import AsyncUnifiedMusicRecommender, TrackMetadata
from mess_ai.datasets import DatasetFactory

# Import routers
from .routers import tracks_router, recommendations_router, audio_router, metadata_router, health_router

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
    
    # Initialize dataset
    try:
        dataset = DatasetFactory.create_dataset(settings.DATASET_TYPE, settings.data_dir)
        metadata_dict = dataset.load_metadata()
        logger.info(f"Loaded {dataset.name} with {len(metadata_dict)} tracks")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        metadata_dict = {}
        dataset = None
    
    # Initialize audio library
    try:
        wav_dir = dataset.get_wav_dir() if dataset else settings.wav_dir
        library = MusicLibrary(wav_dir=str(wav_dir))
        logger.info("Audio library initialized")
    except Exception as e:
        logger.error(f"Failed to initialize audio library: {e}")
    
    # Initialize async unified recommender
    try:
        features_dir = dataset.get_features_dir() if dataset else settings.features_dir
        recommender = AsyncUnifiedMusicRecommender(
            features_dir=str(features_dir),
            enable_cache=True
        )
        logger.info("Async unified music recommender initialized successfully")
        
        # Log available strategies
        strategies = ["similarity", "diverse", "popular", "random", "hybrid"]
        logger.info(f"Available recommendation strategies: {strategies}")
    except Exception as e:
        logger.error(f"Failed to initialize async unified recommender: {e}")
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
# Include "null" origin for file:// URLs (local HTML files)
allowed_origins = settings.ALLOWED_ORIGINS + ["null"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)  # Health and root endpoints
app.include_router(metadata_router)  # Metadata endpoints
app.include_router(tracks_router)
app.include_router(recommendations_router)
app.include_router(audio_router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "backend.api.main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=True
    )