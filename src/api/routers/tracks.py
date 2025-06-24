"""
Track management API routes.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

from core.dependencies import MetadataServiceDep
from core.services.metadata_service import MetadataService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tracks", tags=["tracks"])


@router.get("")
async def get_all_tracks(
    metadata_service: MetadataService = MetadataServiceDep,
    composer: Optional[str] = Query(None, description="Filter by composer"),
    era: Optional[str] = Query(None, description="Filter by era (Baroque, Classical, Romantic, Modern)"),
    form: Optional[str] = Query(None, description="Filter by form (Sonata, Prelude, etc.)"),
    search: Optional[str] = Query(None, description="Search in title, composer, or tags")
):
    """Get list of all available tracks with metadata and optional filtering."""
    try:
        tracks_with_metadata = metadata_service.get_all_tracks(
            composer=composer,
            era=era,
            form=form,
            search=search
        )
        
        return {
            "tracks": tracks_with_metadata,
            "count": len(tracks_with_metadata),
            "filters": {
                "composer": composer,
                "era": era,
                "form": form,
                "search": search
            }
        }
    except Exception as e:
        logger.error(f"Error getting tracks: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{track_id}/metadata")
async def get_track_metadata(track_id: str, metadata_service: MetadataService = MetadataServiceDep):
    """Get detailed metadata for a specific track."""
    metadata = metadata_service.get_track_metadata(track_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Metadata not found for track: {track_id}")
    
    return metadata