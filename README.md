# MESS-AI: Intelligent Music Similarity Search

A production-ready music recommendation system using empirically validated MERT layer specializations and natural language queries for classical music discovery.

## What Makes This Different

Unlike systems using arbitrary feature combinations, we **systematically discovered** which MERT layers encode specific musical aspects through rigorous cross-validation:

- **Layer 0**: Spectral brightness (R² = 0.944) - Best for timbral similarity
- **Layer 1**: Timbral texture (R² = 0.922) - Instrumental characteristics  
- **Layer 2**: Acoustic structure (R² = 0.933) - Resonance patterns

This replaces simple feature averaging (which causes 90%+ similarity between all tracks) with evidence-based recommendations.

## Quick Start

```bash
# Start ML Pipeline Service (port 8001)  
cd pipeline && python api/main.py

# Start Backend API Service (port 8000)
cd backend && python -m api.main

# Access: http://localhost:8000/docs
```

## Architecture

```
mess-ai/
├── backend/     # Web API service (port 8000)
├── pipeline/    # ML processing service (port 8001)
└── data/        # Audio files and MERT features
```

**Services:**
- **Backend**: REST API, metadata, audio streaming
- **Pipeline**: MERT processing, layer-based recommendations, natural language queries

## Key Features

- **Natural Language Queries**: "Find tracks with bright, sparkling timbre"
- **Evidence-Based Search**: Uses validated layer specializations, not guesswork
- **Sub-millisecond Search**: FAISS-powered similarity with proper layer mappings
- **Service Architecture**: Independent scaling for ML vs web workloads

## API Examples

```bash
# Natural language query
curl -X POST http://localhost:8000/query/ \
  -d '{"query": "Find pieces with warm timbral texture"}'

# Direct recommendations
curl -X POST http://localhost:8001/recommend \
  -d '{"track_id": "Beethoven_Op027No1-01", "n_recommendations": 5}'
```

## Tech Stack

- **ML**: PyTorch 2.6+, transformers 4.38+, scikit-learn, FAISS
- **Backend**: Python 3.11+, FastAPI, httpx
- **Audio**: librosa, soundfile, Apple Silicon optimization

## Performance

- **Similarity Search**: <1ms per query
- **Feature Extraction**: ~2.6 minutes for 50-track dataset (M3 Pro)  
- **API Response**: <100ms metadata, <500ms recommendations

Built with scientific rigor for music discovery and AI-powered recommendations.