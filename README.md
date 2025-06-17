# mess-ai

A music similarity search system using deep learning to find musically similar passages based on expressive characteristics. Built with the Saarland Music Dataset (SMD) featuring 50 classical recordings.

## Features

- **Audio Playback**: Web-based music player with waveform visualization
- **Multi-modal Dataset**: Support for audio (WAV), MIDI, and performance annotations
- **Feature Extraction**: Comprehensive audio feature analysis (MFCC, mel spectrograms, chroma, etc.)
- **Web Interface**: Responsive Bootstrap-based player with track selection
- **Dataset Management**: Advanced utilities for the Saarland Music Dataset

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the web server:**
   ```bash
   cd src/api && python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Access the web interface:**
   Open http://localhost:8000 in your browser

## Architecture

### Core Components
- **MusicLibrary** - Audio file management and playback
- **FastAPI Server** - Web API with audio serving endpoints  
- **Feature Extraction** - Audio analysis using librosa
- **Dataset Utilities** - SMD dataset management and processing

### Data Flow
Audio files → Feature extraction → Web API → Interactive player interface

## Dataset Structure

The Saarland Music Dataset in `/data/smd/`:
- `wav-22/`, `wav-44/` - Audio recordings at different sample rates
- `midi/` - MIDI files for symbolic representation  
- `wav-midi/` - Synthesized MIDI audio
- `csv/` - Performance annotations and metadata

## Development Status

**Working:** Audio playback, feature extraction, dataset management, web interface

**In Development:** Similarity search algorithm, ML model training, recommendation system

## Tech Stack

- **Backend:** Python, FastAPI, PyTorch, librosa
- **Frontend:** HTML5, Bootstrap 5, JavaScript
- **Audio:** soundfile, pretty_midi, matplotlib
- **ML:** scikit-learn, NumPy, pandas

## Notebooks

Interactive Jupyter notebooks for experimentation:
```bash
jupyter notebook notebooks/
```