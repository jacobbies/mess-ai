"""
Audio and media file serving API routes.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
import logging

from core.config import settings
from core.dependencies import AudioServiceDep
from core.services.audio_service import AudioService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["media"])


@router.get("/audio/{filename}")
async def get_audio(filename: str, audio_service: AudioService = AudioServiceDep):
    """Serve audio files from local storage or redirect to S3."""
    # If using S3/CDN in production, redirect to the CDN URL
    if settings.AUDIO_BASE_URL:
        cdn_url = f"{settings.AUDIO_BASE_URL}{filename}"
        return RedirectResponse(url=cdn_url, status_code=302)
    
    # Serve from local file system
    file_path = audio_service.get_audio_file_path(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(str(file_path), media_type="audio/wav")


@router.get("/waveform/{filename}")
async def get_waveform(filename: str, audio_service: AudioService = AudioServiceDep):
    """Serve pre-generated waveform images from processed directory or S3."""
    # Remove .wav extension if present
    track_id = filename.replace('.wav', '')
    
    # If using S3/CDN in production, redirect to the CDN URL
    if settings.WAVEFORM_BASE_URL:
        cdn_url = f"{settings.WAVEFORM_BASE_URL}{track_id}.png"
        return RedirectResponse(url=cdn_url, status_code=302)
    
    # Check processed waveforms directory
    waveform_path = audio_service.get_waveform_file_path(filename)
    
    if waveform_path:
        # Serve pre-generated waveform from processed directory
        return FileResponse(
            str(waveform_path),
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable"  # 1 year cache
            }
        )
    
    # No fallback - return 404 if waveform not found
    raise HTTPException(status_code=404, detail="Waveform not found")