import io
import sys
import os
import matplotlib
matplotlib.use('Agg')
from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging
from pydantic import BaseModel
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mess_ai.audio.player import MusicLibrary
from mess_ai.models.recommender import MusicRecommender

app = FastAPI(title = "Music Recommendation System")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

wav_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/smd/wav-44'))
features_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/features'))
library = MusicLibrary(wav_dir=wav_dir)

# Initialize recommender
try:
    recommender = MusicRecommender(features_dir=features_dir)
    logging.info("Music recommender initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize recommender: {e}")
    recommender = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    wav_files = library.list_files()
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "wav_files": [f.name for f in wav_files]
    })

@app.get("/audio/{filename}")
async def audio(filename: str):
    # Serve the audio file to the browser
    file_path = os.path.join(library.wav_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/waveform/{filename}")
def waveform(filename: str):
    try:
        file_path = os.path.join(library.wav_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        library.load_file(file_path)
        img_buf = library.plot_waveform(save_to_buffer=True)

        if img_buf is None:
            raise HTTPException(status_code=500, detail="Failed to generate waveform img")

        return Response(content=img_buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/{track_name}")
async def get_recommendations(track_name: str, top_k: int = 5):
    """Get music recommendations based on MERT feature similarity."""
    try:
        if recommender is None:
            raise HTTPException(status_code=503, detail="Recommender not available - features not loaded")
        
        # Remove .wav extension if present for consistency
        clean_track_name = track_name.replace('.wav', '')
        
        # Get similar tracks
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
        logging.error(f"Recommendation error: {e}")
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
        logging.error(f"Error getting tracks: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host='127.0.0.1', port=8000, reload=False)