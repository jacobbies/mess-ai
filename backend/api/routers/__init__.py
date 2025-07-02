"""API routers package."""
from .tracks import router as tracks_router
from .recommendations import router as recommendations_router
from .audio import router as audio_router
from .metadata import router as metadata_router
from .health import router as health_router

__all__ = [
    "tracks_router",
    "recommendations_router", 
    "audio_router",
    "metadata_router",
    "health_router"
]