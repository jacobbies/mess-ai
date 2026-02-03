# CLAUDE.md - ML Development Environment

## Project Overview

**MESS-AI** is a monorepo containing both the ML research library and production backend for music similarity using MERT (Music Understanding Model with Large-Scale Self-Supervised Training) embeddings.

### Architecture
- **mess/** - Core ML library (feature extraction, similarity search, layer discovery)
- **backend/** - FastAPI production API (imports `mess.search` for recommendations)
- **research/** - Jupyter notebooks and CLI scripts for experimentation

### Purpose
- **Local development** optimized for Apple Silicon (M3 Pro) for ML experimentation
- **Production backend** deployable to EC2 with lightweight dependencies (no torch)

### Core Focus
- Feature extraction from audio using MERT
- Layer discovery and validation (finding which layers encode which musical aspects)
- Similarity search algorithm development
- Dataset preprocessing and analysis
- Research experimentation via Jupyter notebooks

## Project Structure

```
mess-ai/
├── mess/                     # Core ML library (pip: mess-ai)
│   ├── extraction/           # MERT feature extraction (requires [ml])
│   ├── probing/              # Layer discovery & validation (requires [ml])
│   ├── search/               # FAISS similarity search (core deps only)
│   ├── datasets/             # Dataset loaders (SMD, MAESTRO)
│   └── config.py             # Global configuration
├── backend/                  # FastAPI API (pip: backend)
│   ├── app/
│   │   ├── api/              # Route handlers
│   │   ├── core/             # Settings, config
│   │   ├── models/           # Pydantic schemas
│   │   ├── services/         # S3, FAISS services
│   │   └── main.py           # FastAPI entry point
│   └── pyproject.toml        # Backend package config
├── research/                 # ML experimentation
│   ├── scripts/              # CLI workflow automation
│   └── notebooks/            # Jupyter notebooks
├── data/                     # Audio files & extracted features
│   ├── smd/                  # Saarland Music Dataset
│   ├── maestro/              # MAESTRO Dataset
│   └── processed/            # Pre-extracted MERT embeddings (~94GB)
├── docs/                     # Research documentation
└── pyproject.toml            # Root package config + uv workspace
```

## Dependency Architecture

The library has split dependencies to keep production images small:

```
mess-ai (core)          mess-ai[ml]              backend
─────────────────       ─────────────────        ─────────────────
numpy, scipy            + torch, torchaudio      mess-ai (core only)
scikit-learn            + transformers           fastapi, uvicorn
faiss-cpu               + librosa                boto3, pydantic
tqdm                    + jupyter, matplotlib    python-dotenv
~200MB                  ~3GB                     ~250MB
```

- **Core deps**: Everything needed for `mess.search` and `mess.datasets`
- **[ml] optional**: Full ML stack for `mess.extraction` and `mess.probing`
- **Backend**: Only depends on `mess-ai` core (no torch in Docker image)

## Key Scientific Discoveries

Through systematic layer discovery experiments, we've validated:

- **Layer 0**: Spectral brightness (R² = 0.944)
- **Layer 1**: Timbral texture (R² = 0.922)
- **Layer 2**: Acoustic structure (R² = 0.933)

These specializations replace naive feature averaging and enable evidence-based similarity search.

## Development Setup

### uv Workspace Commands

```bash
# Install everything (ML + backend) for full local development
uv sync --group all

# Install ML deps only (mess library development)
uv sync --group local

# Install backend deps only (API development)
uv sync --group backend

# Default: just core deps (mess.search, mess.datasets)
uv sync
```

### Dependency Groups

| Group | What it installs | Use case |
|-------|------------------|----------|
| `--group local` | `mess-ai[ml]` | ML research, feature extraction |
| `--group backend` | `backend` | API development |
| `--group all` | Both | Full development |

## Development Workflow

### 1. Feature Extraction
```bash
# Extract MERT embeddings from audio (requires [ml])
uv sync --group local
python research/scripts/extract_features.py --dataset smd

# Output: data/processed/features/aggregated/*.npy
# Format: [13 layers, 768 dims] per track
```

### 2. Layer Discovery
```bash
# Run probing experiments to validate layer specializations
python research/scripts/run_probing.py

# Output: mess/probing/layer_discovery_results.json
# Contains R² scores for layer/proxy target pairs
```

### 3. Similarity Search
```bash
# Test recommendations using validated layers
python research/scripts/demo_recommendations.py --track "Beethoven_Op027No1-01"

# Uses LayerBasedRecommender with empirically validated mappings
```

### 4. Experimentation
```bash
# Launch Jupyter for exploration
jupyter notebook research/notebooks/

# Suggested notebooks:
# - layer_discovery_analysis.ipynb
# - similarity_benchmarks.ipynb
# - feature_visualization.ipynb
```

### 5. Backend Development
```bash
# Run the API locally
uv sync --group all
uvicorn backend.app.main:app --reload

# Or from backend directory
cd backend && uvicorn app.main:app --reload
```

## Core Components

### MESS Library

The `mess/` directory is a Python library with these modules:

**extraction/** (requires `mess-ai[ml]`)
- `extractor.py`: MERT feature extraction from audio (with batching & caching)
- `config.py`: Extraction configuration (sample rate, segment duration, etc.)

**probing/** (requires `mess-ai[ml]`)
- `layer_discovery.py`: Systematic discovery of layer specializations
- `proxy_targets.py`: Musical aspect proxy targets for validation
- `layer_discovery_results.json`: Empirical validation results

**search/** (core deps only - production safe)
- `layer_based_recommender.py`: Recommendation engine using sklearn cosine_similarity
- `faiss_index.py`: FAISS index wrapper for similarity search
- `layer_indices.py`: Per-layer FAISS indices
- `similarity.py`: Similarity computation (cosine, euclidean, etc.)
- `diverse_similarity.py`: Diverse recommendation algorithms
- `cache.py`: Feature caching utilities

**datasets/** (core deps only)
- `base.py`: Base dataset class
- `smd.py`: Saarland Music Dataset loader
- `maestro.py`: MAESTRO dataset loader
- `factory.py`: Dataset factory pattern

### Backend

The `backend/` directory is a FastAPI application:

**app/api/** - Route handlers
- `health.py`: Health check endpoint
- `recommendations.py`: Recommendation API
- `stream.py`: Audio streaming

**app/services/** - Business logic
- `s3_service.py`: S3 audio file access
- `faiss_service.py`: FAISS index queries

**app/core/** - Configuration
- `config.py`: Environment-based settings

**app/models/** - Data models
- `schemas.py`: Pydantic request/response schemas

## Data Flow

```
Audio Files (.wav)
    ↓
MERT Feature Extraction (mess.extraction)  ← requires [ml]
    ↓
Embeddings [13 layers, 768 dims]
    ↓
Layer Discovery (mess.probing)             ← requires [ml]
    ↓
Validated Layer Mappings
    ↓
Similarity Search (mess.search)            ← core deps only
    ↓
Backend API (backend/)                     ← serves recommendations
    ↓
Recommendations
```

## Dataset Structure

```
data/
├── smd/                    # Saarland Music Dataset
│   ├── wav-44/            # 50 audio files at 44kHz (MERT compatible)
│   ├── csv/               # Performance annotations
│   └── midi/              # Symbolic representations
├── maestro/               # MAESTRO Dataset
├── processed/features/     # MERT embeddings (94GB total)
│   ├── raw/               # Full temporal features [segments, 13, time, 768]
│   ├── segments/          # Time-averaged [segments, 13, 768]
│   └── aggregated/        # Track-level [13, 768] - used for similarity search
└── models/                # Future training checkpoints
```

## Performance Characteristics

- **Feature Extraction**: ~2.6 minutes for 50-track dataset (M3 Pro)
- **Similarity Search**: <1ms per query (FAISS IndexFlatIP)
- **Layer Discovery**: ~10-15 minutes full validation
- **Dataset Size**: ~94GB processed features (SMD + MAESTRO)

## Tech Stack

- **Package Manager**: uv (workspaces for monorepo)
- **ML Framework**: PyTorch 2.2+ (MPS acceleration on Apple Silicon)
- **Transformers**: Hugging Face transformers 4.38+ (MERT model)
- **Audio**: librosa, soundfile (optimized for M3)
- **Search**: FAISS (CPU version, sub-millisecond queries)
- **Scientific**: scikit-learn, numpy, pandas
- **Backend**: FastAPI, uvicorn, boto3
- **Development**: Jupyter, matplotlib, seaborn

## Best Practices

### Code Organization
- Keep `mess/` as a clean Python library
- Keep `backend/` as a clean FastAPI application
- Use `research/scripts/` for CLI automation and batch processing
- Use `research/notebooks/` for exploration and visualization
- Document discoveries in `docs/`

### Development Patterns
- Run experiments in notebooks first
- Productionize proven code into `mess/` modules
- Use scripts for repeatable workflows
- Backend imports from `mess.search` (not `mess.extraction`)

### Data Management
- Keep raw audio in `data/{dataset}/wav-44/`
- Store processed features in `data/processed/features/`
- Never commit large binary files (use .gitignore)
- Document feature extraction parameters

### Research Workflow
1. **Explore** in Jupyter notebooks
2. **Validate** with probing experiments
3. **Productionize** proven code into `mess/`
4. **Deploy** via backend API

## Common Tasks

### Extract features from new audio
```bash
# Add audio to data/{dataset}/wav-44/
uv sync --group local
python research/scripts/extract_features.py --dataset {dataset}
```

### Validate new layer hypothesis
```bash
# Add proxy target to mess/probing/proxy_targets.py
# Run discovery
python research/scripts/run_probing.py
```

### Test new similarity metric
```bash
# Update mess/search/similarity.py
# Benchmark
python research/scripts/evaluate_similarity.py
```

### Experiment with recommendations
```bash
# Direct Python usage
python research/scripts/demo_recommendations.py --track {track_id} --aspect {aspect}

# Or in Jupyter for visualization
```

### Run backend locally
```bash
uv sync --group backend
uvicorn backend.app.main:app --reload --port 8000
```

## Development Status

**🚧 In Progress:**
- Model fine-tuning on SMD dataset for domain-specific similarity
- Advanced FAISS indices (IVF, HNSW) for even larger datasets
- Expanded proxy target validation

**📋 Planned:**
- Multi-modal fusion (audio + score + metadata)
- User preference learning
- Expanded dataset support beyond classical music

## Notes for Claude

- This is a **monorepo** with both ML library and production backend
- `mess/` is the core library - keep it clean and modular
- `backend/` is the FastAPI API - imports `mess.search` only
- `research/` contains experimentation code (notebooks, scripts)
- Use `uv sync --group all` for full development setup
- Backend should NEVER import `mess.extraction` or `mess.probing` (requires torch)
- Focus on scientific rigor in `mess/`, production readiness in `backend/`
