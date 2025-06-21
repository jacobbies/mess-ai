import sys
import os
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for asset URLs (prepare for S3)
WAVEFORM_BASE_URL = os.getenv('WAVEFORM_BASE_URL', None)  # e.g., https://cdn.example.com/waveforms/
AUDIO_BASE_URL = os.getenv('AUDIO_BASE_URL', None)  # e.g., https://cdn.example.com/audio/

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mess_ai.audio.player import MusicLibrary
from mess_ai.models.recommender import MusicRecommender

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    executor.shutdown(wait=True)
    logger.info("Shutdown complete")

app = FastAPI(title="Music Similarity Explorer", lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

wav_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/smd/wav-44'))
features_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/features'))
library = MusicLibrary(wav_dir=wav_dir)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# Initialize recommender
try:
    recommender = MusicRecommender(features_dir=features_dir)
    logger.info("Music recommender initialized successfully with FAISS")
except Exception as e:
    logger.error(f"Failed to initialize recommender: {e}")
    recommender = None

@app.get("/", response_class=None)
async def index(request: Request):
    wav_files = library.list_files()
    tracks = sorted([f.name for f in wav_files])
    
    logger.info(f"Rendering home page with {len(tracks)} tracks")
    if tracks:
        logger.info(f"First track: {tracks[0]}")
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "wav_files": tracks
    })

@app.get("/audio/{filename}")
async def audio(filename: str):
    """Serve audio files from local storage or redirect to S3."""
    # If using S3/CDN in production, redirect to the CDN URL
    if AUDIO_BASE_URL:
        cdn_url = f"{AUDIO_BASE_URL}{filename}"
        return RedirectResponse(url=cdn_url, status_code=302)
    
    # Serve from local file system
    file_path = os.path.join(library.wav_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav")

@app.get("/waveform/{filename}")
async def waveform(filename: str):
    """Serve pre-generated waveform images from processed directory or S3."""
    # Remove .wav extension if present
    track_id = filename.replace('.wav', '')
    
    # If using S3/CDN in production, redirect to the CDN URL
    if WAVEFORM_BASE_URL:
        cdn_url = f"{WAVEFORM_BASE_URL}{track_id}.png"
        return RedirectResponse(url=cdn_url, status_code=302)
    
    # Check processed waveforms directory
    processed_waveform_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "waveforms"
    waveform_path = processed_waveform_dir / f"{track_id}.png"
    
    if waveform_path.exists():
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

@app.get("/recommend/{track_name}")
async def get_recommendations(track_name: str, top_k: int = 5):
    """Get music recommendations based on MERT feature similarity."""
    try:
        if recommender is None:
            raise HTTPException(status_code=503, detail="Recommender not available - features not loaded")
        
        # Remove .wav extension if present for consistency
        clean_track_name = track_name.replace('.wav', '')
        
        logger.info(f"Getting {top_k} recommendations for {clean_track_name}")
        
        # Get similar tracks using FAISS
        similar_tracks = recommender.find_similar_tracks(clean_track_name, top_k=top_k)
        
        # Format response
        recommendations = []
    
        for track, score in similar_tracks:
            recommendations.append({
                "track_name": track,
                "similarity_score": round(score, 4),
                "display_name": track.replace('_', ' ').replace('-', ' ')
            })
        
        return {
            "reference_track": clean_track_name,
            "recommendations": recommendations,
            "total_tracks": len(recommender.get_track_names())
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/tracks")
async def get_all_tracks():
    """Get list of all available tracks for recommendations."""
    try:
        if recommender is None:
            raise HTTPException(status_code=503, detail="Recommender not available")
        
        tracks = recommender.get_track_names()
        return {
            "tracks": tracks,
            "count": len(tracks)
        }
    except Exception as e:
        logger.error(f"Error getting tracks: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    wav_files = list(library.list_files())
    return {
        "status": "healthy",
        "library_loaded": library is not None,
        "recommender_loaded": recommender is not None,
        "tracks_count": len(wav_files),
        "faiss_ready": recommender.search_engine.faiss_index is not None if recommender else False
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=False)