# CLAUDE.md - ML Research Environment

## Project Overview

**MESS-AI** is a Python library for MERT-embedding-based classical music passage retrieval.

### Architecture
- **mess/** - Core ML library (feature extraction, similarity search, layer discovery)
- **scripts/** - CLI workflow automation scripts
- **data/** - Audio datasets and extracted features (239GB total)

### Purpose
- Lean library for music passage retrieval using MERT embeddings
- Layer-wise feature analysis and evidence-based aspect search
- Platform-agnostic (works on Linux, macOS, Windows with appropriate GPU/CPU backends)

### Core Focus
- Feature extraction from audio using MERT
- Layer discovery and validation (which layers encode which musical aspects)
- FAISS cosine similarity search with aspect-weighted retrieval
- MLflow experiment tracking for reproducible research

## Project Structure

```
mess-ai/
├── mess/                     # Core ML library (pip install mess-ai)
│   ├── extraction/           # MERT feature extraction
│   ├── probing/              # Layer discovery & validation
│   ├── search/               # FAISS similarity search
│   ├── datasets/             # Dataset loaders (SMD, MAESTRO)
│   └── config.py             # Global configuration
├── tests/                    # pytest test suite (see tests/tests.md)
│   ├── datasets/             # Dataset loader tests
│   ├── extraction/           # Audio & storage tests
│   ├── probing/              # Discovery & registry tests
│   └── search/               # FAISS search tests
├── scripts/                  # CLI workflow automation
├── notebooks/                # Jupyter exploration
├── data/                     # Audio files & extracted features (239GB total)
│   ├── audio/                # Raw audio files (124GB)
│   ├── embeddings/           # Pre-extracted MERT embeddings (115GB)
│   ├── proxy_targets/        # Computed proxy target features (135MB)
│   ├── indices/              # FAISS index files
│   └── metadata/             # Dataset metadata and manifests
├── mlruns/                   # MLflow experiment tracking (gitignored)
├── docs/                     # Research documentation
└── pyproject.toml            # Package configuration
```

## Dependency Architecture

**Core Dependencies:**
- **Scientific**: numpy, scipy, scikit-learn, pandas
- **ML Framework**: PyTorch 2.10+ (with MPS/CUDA support), torchcodec
- **Transformers**: Hugging Face transformers 4.38+, safetensors
- **Audio Processing**: librosa 0.10+, soundfile, nnaudio
- **Search**: FAISS (CPU version)
- **Experiment Tracking**: MLflow 2.10+

**Installation:**
```bash
# Install all dependencies (including test deps)
uv sync --group dev --extra search

# Or using pip
pip install -e .
```

## Layer Discovery System

**15 audio-derived proxy targets across 6 musical dimensions** (see `mess/probing/discovery.py` SCALAR_TARGETS):
- **Timbre**: spectral_centroid, spectral_rolloff, spectral_bandwidth, zero_crossing_rate
- **Rhythm**: tempo, onset_density
- **Dynamics**: dynamic_range, dynamic_variance, crescendo_strength, diminuendo_strength
- **Harmony**: harmonic_complexity
- **Articulation**: attack_slopes, attack_sharpness
- **Phrasing**: phrase_regularity, num_phrases

**User-facing aspects** (in `mess/probing/discovery.py` ASPECT_REGISTRY):
- brightness, texture, warmth, tempo, rhythmic_energy, dynamics, crescendo, harmonic_richness, articulation, phrasing

**Methodology:**
- Ridge regression with 5-fold cross-validation on frozen MERT embeddings
- 13 layers probed against each proxy target
- R² > 0.8 -> "high confidence"
- R² > 0.5 -> "medium confidence"

**Example Validated Results** (from actual discovery runs):
- Layer 0 -> spectral_centroid (R² = 0.944) -> maps to "brightness" aspect
- Layer 1 -> spectral_rolloff (R² = 0.922) -> maps to "texture" aspect
- Layer 2 -> harmonic_complexity (R² = 0.933) -> maps to "harmonic_richness" aspect

These specializations replace naive feature averaging and enable evidence-based similarity search.

## Development Setup

### Installation

```bash
uv sync --group dev --extra search
```

### Running Tests

```bash
uv run pytest -v                          # all tests verbose
uv run pytest --cov=mess --cov-report=term-missing  # with coverage
uv run pytest -m unit                     # only fast unit tests
uv run pytest tests/test_config.py -v     # single module
```

Tests run in a few seconds with no model loading or real audio I/O. See `tests/tests.md` for full details.

## Development Workflow

### 1. Feature Extraction
```bash
python scripts/extract_features.py --dataset smd

# Output: data/embeddings/<dataset>-emb/
#   - raw/: Full temporal features [segments, 13, time, 768]
#   - segments/: Time-averaged [segments, 13, 768]
#   - aggregated/: Track-level [13, 768] - used for similarity search
# MLflow: Logs to "feature_extraction" experiment with timing metrics
```

### 2. Layer Discovery
```bash
python scripts/run_probing.py

# Output: mess/probing/layer_discovery_results.json
# Contains R² scores for 13 layers x N proxy targets
# MLflow: Logs to "layer_discovery" experiment with params, metrics, and artifact
```

### 3. Similarity Search
```bash
python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01"

# The recommender auto-loads layer_discovery_results.json and resolves aspects.
# Available aspects depend on what's been validated (run discovery first).
```

### 4. Experiment Tracking with MLflow
```bash
mlflow ui   # then open http://localhost:5000
```

**MLflow directory**: `mlruns/` (gitignored, stored locally)

## Core Components

### MESS Library

**extraction/**
- `extractor.py`: MERT feature extraction from audio (with batching & caching)
- `audio.py`: Audio loading and preprocessing
- `pipeline.py`: Feature extraction pipeline orchestration
- `storage.py`: Feature storage and caching utilities

**probing/**
- `discovery.py`: Systematic discovery of layer specializations via linear probing
- `targets.py`: Musical aspect proxy targets for validation

**search/**
- `search.py`: Simplified similarity search interface (track + clip + aspect-weighted)
- `faiss_index.py`: Artifact build/save/load and S3 publish helpers

**datasets/**
- `base.py`: Base dataset class with common functionality
- `smd.py`: Saarland Music Dataset loader
- `maestro.py`: MAESTRO dataset loader
- `factory.py`: Dataset factory pattern

## Data Flow

```
Audio Files (.wav)
    v
MERT Feature Extraction (mess.extraction)
    v
Embeddings [13 layers, 768 dims]
    v
Proxy Target Computation (mess.probing.targets)
    v
Layer Discovery (mess.probing.discovery)
    v
Validated Layer Mappings (layer_discovery_results.json)
    v
Similarity Search (mess.search)
    v
Recommendations
```

## Dataset Structure

```
data/
├── audio/                      # Raw audio files (124GB)
│   ├── smd/                    # Saarland Music Dataset
│   └── maestro/                # MAESTRO Dataset (~1200 recordings)
├── embeddings/                 # MERT embeddings (115GB)
│   ├── smd-emb/
│   ├── maestro-emb/
│   └── demo-emb/
├── proxy_targets/              # Computed proxy features (135MB)
├── indices/                    # FAISS index files
└── metadata/                   # Dataset manifests and track lists
```

## Performance Characteristics

- **Feature Extraction**: ~2.6 minutes for 50-track dataset (M3 Pro)
- **Similarity Search**: <1ms per query (FAISS IndexFlatIP)
- **Layer Discovery**: ~10-15 minutes full validation

## Tech Stack

- **Package Manager**: uv (fast Python package manager)
- **ML Framework**: PyTorch 2.10+ (MPS/CUDA/CPU), torchcodec 0.10+
- **Transformers**: Hugging Face transformers 4.38+ (MERT model)
- **Audio**: librosa 0.10+, soundfile, nnaudio 0.3.4+
- **Search**: FAISS (CPU version, sub-millisecond queries)
- **Scientific**: scikit-learn, numpy, scipy, pandas
- **Experiment Tracking**: MLflow 2.10+

## Best Practices

### Code Organization
- Keep `mess/` as a clean, well-documented Python library
- Use `scripts/` for CLI automation and batch processing
- Use `notebooks/` for exploration and visualization
- Document discoveries in `docs/`

### Development Patterns
- Productionize proven code into `mess/` modules
- Use scripts for repeatable workflows and batch processing
- Write tests for new code - mirror source structure in `tests/` (e.g., `mess/search/` -> `tests/search/`)
- Unit tests should avoid loading the MERT model; mock heavy deps, use `tmp_path` for I/O
- Run `uv run pytest` before committing

### Data Management
- Keep raw audio in `data/audio/{dataset}/`
- Store processed features in `data/embeddings/{dataset}-emb/`
- Proxy targets go in `data/proxy_targets/`
- Never commit large binary files (use .gitignore)
- Document feature extraction parameters in experiment tracking

## Common Tasks

### Extract features from new audio
```bash
python scripts/extract_features.py --dataset {dataset}
```

### Validate new layer hypothesis
```bash
# 1. Add proxy target to mess/probing/targets.py (if needed)
# 2. Add to SCALAR_TARGETS in mess/probing/discovery.py
# 3. Run discovery
python scripts/run_probing.py

# 4. Check MLflow for R² scores
mlflow ui

# 5. If R² > 0.5, the aspect is automatically available in the recommender
```

### Experiment with recommendations
```bash
python scripts/demo_recommendations.py --track {track_id} --aspect {aspect}
```

## Development Status

**Current baseline (working):**
- Frozen MERT embeddings + cosine similarity via FAISS IndexFlatIP
- Aspect-weighted search using layer discovery (proxy target R2 -> best layer -> cosine in that layer)
- Clip-level retrieval via FAISS artifact format

## Related Repo: classical-recsys

The visible demo application lives at `~/projects/classical-recsys` (FastAPI + Next.js 16). It consumes `mess-ai[search]` as a pip dependency and exposes passage similarity search via `search_by_clip()` over S3-hosted FAISS artifacts. Ongoing design doc for the aspect-slider demo: `docs/passage-search-demo.md` in that repo. Keep the public API in `mess/__init__.py` and `mess/search/__init__.py` stable for that consumer.

## Notes for Claude

- **`~/projects/neural-audio-codec` is READ-ONLY** - you may read files in that repo for reference but must NOT edit, write, or delete anything there.
- This is an **open source ML library** intended for public release
- `mess/` is the core library - keep it clean, well-documented, and modular
- `scripts/` contains CLI automation, `notebooks/` contains experimentation code
- Use `uv sync` for full development setup
- All experiments are tracked in MLflow - run `mlflow ui` to view results
- Focus on scientific rigor, reproducibility, and clean code
- Total storage requirement: ~239GB (124GB audio + 115GB embeddings)
