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

app = FastAPI(title = "Music Recommendation System")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

wav_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/smd/wav-44'))
library = MusicLibrary(wav_dir=wav_dir)

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

@app.get("/recommend/{filename}")
async def recommend(filename: str, count: int = 5):
    try:
        file_path = os.path.join(library.wav_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
    # Implement in MusicLibrary class
        similar_tracks = library.find_similar_tracks(file_path, n=count)
        return {"recommendations": [Path(track).name for track, _ in similar_tracks]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host='127.0.0.1', port=8000, reload=False)