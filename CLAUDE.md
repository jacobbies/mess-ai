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

**Processing full dataset features:**
```bash
# Run from notebooks/test_feature_extraction.ipynb
# OR programmatically:
from src.mess_ai.features.extractor import extract_features
extract_features()  # ~2.6 minutes on M3 Pro
```

## Architecture Overview

**mess-ai** is a production-ready music similarity search system using MERT (Music Understanding Model) embeddings to find musically similar classical pieces. The system processes the Saarland Music Dataset (SMD) with 50 classical recordings into 94GB of deep learning features.

**Key architectural components:**

- **FeatureExtractor** (`src/mess_ai/features/extractor.py`) - Complete MERT-based feature extraction with Apple Silicon MPS support, multi-scale output generation (raw/segments/aggregated)
- **MusicRecommender** (`src/mess_ai/models/recommender.py`) - Production similarity search using cosine similarity on precomputed MERT features, loads 94GB of embeddings into memory cache
- **FastAPI Server** (`src/api/app.py`) - Web API with endpoints for audio serving (`/audio/{track}`), waveform generation (`/waveform/{track}`), and MERT-powered recommendations (`/recommend/{track}`, `/tracks`)
- **MusicLibrary** (`src/mess_ai/audio/player.py`) - Core audio file management with soundfile integration and matplotlib waveform visualization
- **Web Interface** (`src/api/templates/index.html`) - Bootstrap 5-based responsive music player with background waveform loading, clickable similarity search, and user-controlled recommendation discovery

**Data flow:**
1. SMD audio files (WAV at 44kHz) stored in `/data/smd/wav-44/` → **50 tracks ready**
2. FeatureExtractor processes audio through MERT-v1-95M → **150 .npy files generated**
3. MusicRecommender loads precomputed features for real-time similarity search → **Sub-second queries**
4. FastAPI serves audio, generates waveforms, and provides MERT-based recommendations → **Production API**
5. Web frontend provides interactive player with AI-powered similarity search → **Complete UX**

## Technical Stack

- **Backend:** Python 3.11+, FastAPI, PyTorch 2.6+, transformers 4.38+, scikit-learn
- **Frontend:** Jinja2 templates, Bootstrap 5, vanilla JavaScript with async/await
- **Audio:** soundfile, librosa, torchaudio with Apple Silicon acceleration
- **ML:** MERT-v1-95M transformer, Wav2Vec2FeatureExtractor, cosine similarity
- **Storage:** NumPy .npy files, hierarchical feature organization (Plan A structure)

## Development Status

**✅ Completed (Production Ready):**
- **MERT Feature Extraction Pipeline** - Complete with MPS acceleration, 2.6min processing time
- **Similarity Search System** - Real cosine similarity using 13×768 MERT embeddings
- **Web Interface** - Interactive player with AI recommendations and background waveform loading
- **API Endpoints** - `/recommend/{track}`, `/tracks`, `/audio/{track}`, `/waveform/{track}`
- **Data Processing** - 94GB of precomputed MERT features (raw/segments/aggregated)
- **Apple Silicon Optimization** - MPS acceleration with CPU fallback
- **Smart UX** - User-controlled recommendations, background loading, caching

**🚧 In Progress:**
- Model fine-tuning on SMD dataset for domain-specific similarity
- Alternative similarity metrics (Euclidean, Manhattan, learned metrics)
- Performance optimization for larger datasets

**📋 Planned:**
- Docker containerization for deployment
- AWS S3 integration for cloud storage
- Expanded dataset support beyond classical music
- User preference learning and personalization
- Comprehensive testing suite with CI/CD

## Dataset Structure

The Saarland Music Dataset is organized with processed features:

```
data/
├── smd/                    # Original SMD dataset
│   ├── wav-44/            # 50 audio files at 44kHz (MERT compatible)
│   ├── csv/               # Performance annotations
│   └── midi/              # Symbolic representations
├── processed/features/     # MERT embeddings (94GB total)
│   ├── raw/               # Full temporal features [segments, 13, time, 768] 
│   ├── segments/          # Time-averaged [segments, 13, 768]
│   └── aggregated/        # Track-level [13, 768] - used for similarity search
└── models/                # Future training checkpoints
```

## Important Implementation Notes

- **MERT Requirements:** 24kHz audio input, trust_remote_code=True for model loading
- **Feature Loading:** MusicRecommender loads all 50 tracks into memory cache on initialization
- **Apple Silicon:** MPS acceleration reduces processing from hours to minutes
- **Similarity Search:** Uses flattened aggregated features (13×768=9984 dimensions) with cosine similarity
- **Background Processing:** Waveform generation and feature extraction happen asynchronously
- **Error Handling:** Graceful fallbacks for MPS→CPU, missing features, network errors
- **Memory Management:** ~2GB RAM for feature cache, streaming for large audio files

## Testing and Validation

- **Feature Extraction Testing:** `notebooks/test_feature_extraction.ipynb` - comprehensive validation
- **API Testing:** Manual testing via browser interface and curl commands
- **Performance Monitoring:** Built-in timing and logging throughout pipeline
- **Quality Assurance:** Feature integrity checks, similarity score validation