import sys
import os
from fastapi import FastAPI, Request, HTTPException, Query
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for asset URLs (prepare for S3)
WAVEFORM_BASE_URL = os.getenv('WAVEFORM_BASE_URL', None)  # e.g., https://cdn.example.com/waveforms/
AUDIO_BASE_URL = os.getenv('AUDIO_BASE_URL', None)  # e.g., https://cdn.example.com/audio/

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mess_ai.audio.player import MusicLibrary
from mess_ai.models.recommender import MusicRecommender
from mess_ai.metadata.processor import MetadataProcessor
from mess_ai.models.metadata import TrackMetadata

# Initialize metadata processor
metadata_processor = MetadataProcessor()
metadata_dict: Dict[str, TrackMetadata] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global metadata_dict
    try:
        metadata_dict = metadata_processor.load_metadata_dict()
        logger.info(f"Loaded metadata for {len(metadata_dict)} tracks")
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        metadata_dict = {}
    
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
    """Get music recommendations based on MERT feature similarity with metadata."""
    try:
        if recommender is None:
            raise HTTPException(status_code=503, detail="Recommender not available - features not loaded")
        
        # Remove .wav extension if present for consistency
        clean_track_name = track_name.replace('.wav', '')
        
        logger.info(f"Getting {top_k} recommendations for {clean_track_name}")
        
        # Get similar tracks using FAISS
        similar_tracks = recommender.find_similar_tracks(clean_track_name, top_k=top_k)
        
        # Get metadata for reference track
        ref_metadata = metadata_dict.get(clean_track_name)
        
        # Format response with metadata
        recommendations = []
    
        for track, score in similar_tracks:
            track_metadata = metadata_dict.get(track)
            
            if track_metadata:
                rec_data = {
                    "track_id": track,
                    "title": track_metadata.title,
                    "composer": track_metadata.composer,
                    "composer_full": track_metadata.composer_full,
                    "era": track_metadata.era,
                    "form": track_metadata.form,
                    "key_signature": track_metadata.key_signature,
                    "similarity_score": round(score, 4),
                    "filename": track_metadata.filename,
                    "tags": track_metadata.tags
                }
            else:
                # Fallback if metadata not found
                rec_data = {
                    "track_id": track,
                    "title": track.replace('_', ' ').replace('-', ' '),
                    "similarity_score": round(score, 4),
                    "filename": f"{track}.wav"
                }
            
            recommendations.append(rec_data)
        
        response = {
            "reference_track": clean_track_name,
            "recommendations": recommendations,
            "total_tracks": len(recommender.get_track_names())
        }
        
        # Add reference track metadata if available
        if ref_metadata:
            response["reference_metadata"] = {
                "title": ref_metadata.title,
                "composer": ref_metadata.composer,
                "composer_full": ref_metadata.composer_full,
                "era": ref_metadata.era,
                "form": ref_metadata.form,
                "key_signature": ref_metadata.key_signature,
                "tags": ref_metadata.tags
            }
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/tracks")
async def get_all_tracks(
    composer: Optional[str] = Query(None, description="Filter by composer"),
    era: Optional[str] = Query(None, description="Filter by era (Baroque, Classical, Romantic, Modern)"),
    form: Optional[str] = Query(None, description="Filter by form (Sonata, Prelude, etc.)"),
    search: Optional[str] = Query(None, description="Search in title, composer, or tags")
):
    """Get list of all available tracks with metadata and optional filtering."""
    try:
        if recommender is None:
            raise HTTPException(status_code=503, detail="Recommender not available")
        
        track_ids = recommender.get_track_names()
        tracks_with_metadata = []
        
        for track_id in track_ids:
            metadata = metadata_dict.get(track_id)
            
            # Apply filters if provided
            if composer and metadata:
                if composer.lower() not in metadata.composer.lower() and composer.lower() not in metadata.composer_full.lower():
                    continue
            
            if era and metadata:
                if metadata.era != era:
                    continue
            
            if form and metadata:
                if metadata.form != form:
                    continue
            
            if search and metadata:
                search_lower = search.lower()
                if not any([
                    search_lower in metadata.title.lower(),
                    search_lower in metadata.composer.lower(),
                    search_lower in metadata.composer_full.lower(),
                    any(search_lower in tag for tag in metadata.tags)
                ]):
                    continue
            
            if metadata:
                track_data = {
                    "track_id": track_id,
                    "title": metadata.title,
                    "composer": metadata.composer,
                    "composer_full": metadata.composer_full,
                    "era": metadata.era,
                    "form": metadata.form,
                    "key_signature": metadata.key_signature,
                    "opus": metadata.opus,
                    "movement": metadata.movement,
                    "filename": metadata.filename,
                    "tags": metadata.tags,
                    "recording_date": metadata.recording_date.isoformat() if metadata.recording_date else None
                }
            else:
                # Fallback if metadata not found
                track_data = {
                    "track_id": track_id,
                    "title": track_id.replace('_', ' ').replace('-', ' '),
                    "filename": f"{track_id}.wav"
                }
            
            tracks_with_metadata.append(track_data)
        
        # Sort by composer, then by opus/title
        tracks_with_metadata.sort(key=lambda x: (
            x.get('composer', ''),
            x.get('opus', ''),
            x.get('movement', ''),
            x.get('title', '')
        ))
        
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

@app.get("/tracks/{track_id}/metadata")
async def get_track_metadata(track_id: str):
    """Get detailed metadata for a specific track."""
    metadata = metadata_dict.get(track_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Metadata not found for track: {track_id}")
    
    return {
        "track_id": metadata.track_id,
        "filename": metadata.filename,
        "title": metadata.title,
        "composer": metadata.composer,
        "composer_full": metadata.composer_full,
        "opus": metadata.opus,
        "movement": metadata.movement,
        "movement_name": metadata.movement_name,
        "era": metadata.era,
        "form": metadata.form,
        "key_signature": metadata.key_signature,
        "tempo_marking": metadata.tempo_marking,
        "performer_id": metadata.performer_id,
        "performer_name": metadata.performer_name,
        "instrument": metadata.instrument,
        "recording_date": metadata.recording_date.isoformat() if metadata.recording_date else None,
        "year_composed": metadata.year_composed,
        "duration_seconds": metadata.duration_seconds,
        "dataset_source": metadata.dataset_source,
        "tags": metadata.tags
    }

@app.get("/composers")
async def get_composers():
    """Get list of all composers with track counts."""
    composer_stats = {}
    
    for metadata in metadata_dict.values():
        composer = metadata.composer
        if composer not in composer_stats:
            composer_stats[composer] = {
                "composer": composer,
                "composer_full": metadata.composer_full,
                "era": metadata.era,
                "track_count": 0
            }
        composer_stats[composer]["track_count"] += 1
    
    # Convert to list and sort by track count
    composers = list(composer_stats.values())
    composers.sort(key=lambda x: x["track_count"], reverse=True)
    
    return {
        "composers": composers,
        "count": len(composers)
    }

@app.get("/tags")
async def get_tags():
    """Get all unique tags with counts."""
    tag_counts = {}
    
    for metadata in metadata_dict.values():
        for tag in metadata.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    # Sort by count
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "tags": [{"tag": tag, "count": count} for tag, count in sorted_tags],
        "count": len(sorted_tags)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    wav_files = list(library.list_files())
    return {
        "status": "healthy",
        "library_loaded": library is not None,
        "recommender_loaded": recommender is not None,
        "metadata_loaded": len(metadata_dict) > 0,
        "tracks_count": len(wav_files),
        "metadata_count": len(metadata_dict),
        "faiss_ready": recommender.search_engine.faiss_index is not None if recommender else False
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=False)