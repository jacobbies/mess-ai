"""
Metadata and system information API routes.
"""
from fastapi import APIRouter
import logging

from core.config import settings
from core.dependencies import MetadataServiceDep
from core.services.metadata_service import MetadataService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["metadata"])




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


