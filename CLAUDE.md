# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Running the web server:**
```bash
cd src/api && python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Installing dependencies:**
```bash
pip install -r requirements.txt
```

**Running Jupyter notebooks:**
```bash
jupyter notebook notebooks/
```

## Architecture Overview

**mess-ai** is a music similarity search system using deep learning to find musically similar passages based on expressive characteristics. The project uses the Saarland Music Dataset (SMD) with 50 classical recordings.

**Key architectural components:**

- **MusicLibrary** (`src/mess_ai/audio/player.py`) - Core audio file management with soundfile integration, waveform visualization, and metadata tracking
- **FastAPI Server** (`src/api/app.py`) - Web API with endpoints for audio serving, waveform generation, and similarity search (recommendation endpoint currently non-functional)
- **AudioFeatureExtractor** (`misc/feature_extraction.py`) - Comprehensive audio feature extraction using librosa (MFCC, mel spectrograms, chroma, spectral contrast, onset strength, tempograms)
- **SaarlandMusicDataset** (`misc/music_dataset_utils.py`) - Advanced multi-modal dataset manager supporting audio, MIDI, CSV annotations, and synthesized audio with visualization capabilities
- **AudioModelDatasetGenerator** (`misc/dataset_generation.py`) - ML dataset preparation toolkit for classification and sequence modeling (incomplete implementation)
- **MusicRecommender** (`src/mess_ai/models/recommender.py`) - Recommendation system skeleton (find_similar_tracks method not implemented)
- **Web Interface** (`src/api/templates/index.html`) - Bootstrap 5-based responsive music player with track selection, waveform display, and recommendation UI

**Data flow:**
1. SMD audio files (WAV at 22kHz/44kHz, MIDI, synthesized audio) stored in `/data/smd/`
2. AudioFeatureExtractor processes audio into feature representations  
3. Dataset utilities manage multi-modal data loading and preprocessing
4. FastAPI serves audio files, generates waveforms, and provides API endpoints
5. Web frontend provides interactive music player with similarity search interface (similarity functionality pending implementation)

## Technical Stack

- **Backend:** Python 3.x, FastAPI, PyTorch, librosa, NumPy
- **Frontend:** Jinja2 templates, Bootstrap 5, vanilla JavaScript
- **Audio:** soundfile, librosa, pretty_midi, torchaudio
- **ML:** PyTorch, scikit-learn, transformers (planned)

## Development Status

**Implemented:**
- Basic web server with audio playback
- Audio file loading and management
- HTML/JS music player interface
- Dataset structure and loading utilities

**Incomplete/Planned:**
- Core similarity search functionality (`find_similar_tracks` method not implemented)
- Deep learning model training pipeline (using MERT pretrained model, eventual finetuning of this model w dataset)
- Feature extraction integration with ML models
- Comprehensive testing infrastructure
- More loading utilities

## Dataset Structure

The Saarland Music Dataset is organized in `/data/smd/`:
- `wav-22/`, `wav-44/` - Audio recordings at different sample rates
- `midi/` - MIDI files for symbolic representation
- `wav-midi/` - Synthesized MIDI audio
- `csv/` - Performance annotations and metadata

## Important Notes
- We are only working with `wav-44/` because it is compatible with the MERT pretrained model
- The recommendation system (`src/mess_ai/models/recommender.py`) exists but `find_similar_tracks` is not implemented
- Audio processing requires system-level audio libraries (FFmpeg)
- No formal build system - pure Python project with manual server startup, eventually scale
- Jupyter notebooks in `/notebooks/` are used for experimentation and testing