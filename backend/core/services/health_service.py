"""
Health monitoring service.
Handles system health checks and status reporting.
"""
import os
import psutil
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HealthService:
    """Service for system health monitoring and status checks."""
    
    def __init__(self, metadata_service=None, recommendation_service=None, audio_service=None):
        """Initialize with optional service dependencies for health checks."""
        self.metadata_service = metadata_service
        self.recommendation_service = recommendation_service  
        self.audio_service = audio_service
        self.start_time = datetime.utcnow()
        logger.info("HealthService initialized")
    
    def get_basic_health(self) -> Dict:
        """Get basic health status."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
        }
    
    def get_detailed_health(self) -> Dict:
        """Get detailed health status including system metrics."""
        basic_health = self.get_basic_health()
        
        # System metrics
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_health = {
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_percent": round((disk.used / disk.total) * 100, 2)
            },
            "cpu_percent": psutil.cpu_percent(interval=1)
        }
        
        # Service health checks
        services_health = self._check_services_health()
        
        return {
            **basic_health,
            "system": system_health,
            "services": services_health
        }
    
    def _check_services_health(self) -> Dict:
        """Check health of dependent services."""
        services = {}
        
        # Check metadata service
        if self.metadata_service:
            try:
                track_count = len(self.metadata_service.metadata_dict)
                services["metadata"] = {
                    "status": "healthy",
                    "tracks_loaded": track_count
                }
            except Exception as e:
                services["metadata"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        else:
            services["metadata"] = {"status": "not_configured"}
        
        # Check recommendation service
        if self.recommendation_service:
            try:
                has_recommender = self.recommendation_service.recommender is not None
                metrics = {}
                if hasattr(self.recommendation_service, 'get_metrics'):
                    metrics = self.recommendation_service.get_metrics()
                
                services["recommendations"] = {
                    "status": "healthy" if has_recommender else "degraded",
                    "recommender_loaded": has_recommender,
                    "type": "async_unified",
                    **metrics
                }
            except Exception as e:
                services["recommendations"] = {
                    "status": "unhealthy", 
                    "error": str(e)
                }
        else:
            services["recommendations"] = {"status": "not_configured"}
        
        # Check audio service
        if self.audio_service:
            try:
                has_library = self.audio_service.library is not None
                services["audio"] = {
                    "status": "healthy" if has_library else "degraded",
                    "library_loaded": has_library
                }
            except Exception as e:
                services["audio"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        else:
            services["audio"] = {"status": "not_configured"}
        
        return services
    
    def get_readiness_check(self) -> Dict:
        """Check if the application is ready to serve requests."""
        services_health = self._check_services_health()
        
        # Application is ready if critical services are healthy
        critical_services = ["metadata", "recommendations"]
        is_ready = all(
            services_health.get(service, {}).get("status") in ["healthy", "degraded"]
            for service in critical_services
        )
        
        return {
            "ready": is_ready,
            "timestamp": datetime.utcnow().isoformat(),
            "services": services_health
        }
    
    def get_liveness_check(self) -> Dict:
        """Check if the application is alive and responding."""
        try:
            # Simple liveness check - if we can respond, we're alive
            return {
                "alive": True,
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
            }
        except Exception as e:
            return {
                "alive": False,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }