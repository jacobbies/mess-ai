"""
Health monitoring API routes.
Provides health, readiness, and liveness endpoints for monitoring.
"""
from fastapi import APIRouter, HTTPException
import logging

from core.dependencies import HealthServiceDep
from core.services.health_service import HealthService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(health_service: HealthService = HealthServiceDep):
    """Basic health check endpoint."""
    try:
        health_status = health_service.get_basic_health()
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/health/detailed")
async def detailed_health_check(health_service: HealthService = HealthServiceDep):
    """Detailed health check with system metrics and service status."""
    try:
        health_status = health_service.get_detailed_health()
        return health_status
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail="Detailed health check failed")


@router.get("/health/ready")
async def readiness_check(health_service: HealthService = HealthServiceDep):
    """Kubernetes-style readiness probe."""
    try:
        readiness_status = health_service.get_readiness_check()
        
        if not readiness_status.get("ready", False):
            # Return 503 Service Unavailable if not ready
            raise HTTPException(
                status_code=503, 
                detail={
                    "message": "Service not ready",
                    "status": readiness_status
                }
            )
        
        return readiness_status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=500, detail="Readiness check failed")


@router.get("/health/live")
async def liveness_check(health_service: HealthService = HealthServiceDep):
    """Kubernetes-style liveness probe."""
    try:
        liveness_status = health_service.get_liveness_check()
        
        if not liveness_status.get("alive", False):
            # Return 503 Service Unavailable if not alive
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Service not alive", 
                    "status": liveness_status
                }
            )
        
        return liveness_status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        raise HTTPException(status_code=500, detail="Liveness check failed")


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MESS-AI Music Similarity API",
        "version": "1.0.0",
        "description": "AI-powered classical music discovery with MERT embeddings and FAISS similarity search",
        "docs": "/docs",
        "health": "/health"
    }