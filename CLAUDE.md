# CLAUDE.md - ML Research Environment

## Project Overview

**MESS-AI** is a Python library for music similarity research using MERT (Music Understanding Model with Large-Scale Self-Supervised Training) embeddings.

### Architecture
- **mess/** - Core ML library (feature extraction, similarity search, layer discovery)
- **research/** - Jupyter notebooks and CLI scripts for experimentation
- **data/** - Audio datasets and extracted features

### Purpose
- ML research library for music understanding and similarity
- Optimized for Apple Silicon (MPS acceleration) but works on any platform
- Focus on layer-wise feature analysis and evidence-based similarity search

### Core Focus
- Feature extraction from audio using MERT
- Layer discovery and validation (finding which layers encode which musical aspects)
- Similarity search algorithm development
- Dataset preprocessing and analysis
- Research experimentation via Jupyter notebooks
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
├── research/                 # ML experimentation
│   ├── scripts/              # CLI workflow automation
│   └── notebooks/            # Jupyter notebooks
├── data/                     # Audio files & extracted features
│   ├── smd/                  # Saarland Music Dataset
│   ├── maestro/              # MAESTRO Dataset
│   └── processed/            # Pre-extracted MERT embeddings (~94GB)
├── docs/                     # Research documentation
└── pyproject.toml            # Package configuration
```

## Dependency Architecture

The library has split dependencies to keep the core lightweight:

```
mess-ai (core)          mess-ai[ml]
─────────────────       ─────────────────
numpy, scipy            + torch, torchaudio
scikit-learn            + transformers
faiss-cpu               + librosa
~150MB                  + tqdm, mlflow
                        + jupyter, matplotlib
                        ~3GB
```

- **Core deps**: Everything needed for `mess.search` and `mess.datasets` (inference & analysis only)
- **[ml] optional**: Full ML stack for `mess.extraction` and `mess.probing` (feature extraction, research, experiment tracking)

## Key Scientific Discoveries

Through systematic layer discovery experiments, we've validated:

- **Layer 0**: Spectral brightness (R² = 0.944)
- **Layer 1**: Timbral texture (R² = 0.922)
- **Layer 2**: Acoustic structure (R² = 0.933)

These specializations replace naive feature averaging and enable evidence-based similarity search.

## Development Setup

### Installation

```bash
# Install core dependencies only (for using pre-extracted features)
uv sync

# Install with ML dependencies (for feature extraction and research)
uv sync --group dev

# Or using pip
pip install mess-ai              # Core only
pip install mess-ai[ml]          # Full ML stack
```

### Dependency Groups

| Command | What it installs | Use case |
|---------|------------------|----------|
| `uv sync` | Core deps | Using pre-extracted features, similarity search |
| `uv sync --group dev` | `mess-ai[ml]` | ML research, feature extraction |

## Development Workflow

### 1. Feature Extraction
```bash
# Extract MERT embeddings from audio (requires [ml])
uv sync --group dev
python research/scripts/extract_features.py --dataset smd

# Output: data/processed/features/aggregated/*.npy
# Format: [13 layers, 768 dims] per track
# MLflow: Logs to "feature_extraction" experiment with timing metrics
```

### 2. Layer Discovery
```bash
# Run probing experiments to validate layer specializations
python research/scripts/run_probing.py

# Output: mess/probing/layer_discovery_results.json
# Contains R² scores for layer/proxy target pairs
# MLflow: Logs to "layer_discovery" experiment with all metrics and artifacts
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

### 5. Experiment Tracking with MLflow
```bash
# View experiment history and metrics in browser UI
mlflow ui

# Then open http://localhost:5000 to browse:
# - Feature extraction runs (timing, cache hit rates)
# - Layer discovery runs (R² scores, best layers per target)
# - Compare hyperparameters across runs
# - Download artifacts (results JSON, model checkpoints)
```

**What's tracked:**
- **Feature Extraction**: Dataset, workers, timing, cache statistics
- **Layer Discovery**: Ridge alpha, CV folds, per-layer R² scores, best layer mappings
- **Artifacts**: Results JSON files automatically logged

**MLflow directory**: `mlruns/` (gitignored, stored locally)

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

**search/** (core deps only)
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

- **Package Manager**: uv (fast Python package manager)
- **ML Framework**: PyTorch 2.2+ (MPS acceleration on Apple Silicon)
- **Transformers**: Hugging Face transformers 4.38+ (MERT model)
- **Audio**: librosa, soundfile
- **Search**: FAISS (CPU version, sub-millisecond queries)
- **Scientific**: scikit-learn, numpy, pandas
- **Experiment Tracking**: MLflow (metrics, parameters, artifacts)
- **Development**: Jupyter, matplotlib, seaborn

## Best Practices

### Code Organization
- Keep `mess/` as a clean, well-documented Python library
- Use `research/scripts/` for CLI automation and batch processing
- Use `research/notebooks/` for exploration and visualization
- Document discoveries in `docs/`

### Development Patterns
- Run experiments in notebooks first
- Productionize proven code into `mess/` modules
- Use scripts for repeatable workflows
- Keep the core library (`mess.search`, `mess.datasets`) lightweight

### Data Management
- Keep raw audio in `data/{dataset}/wav-44/`
- Store processed features in `data/processed/features/`
- Never commit large binary files (use .gitignore)
- Document feature extraction parameters

### Research Workflow
1. **Explore** in Jupyter notebooks
2. **Validate** with probing experiments (tracked in MLflow)
3. **Productionize** proven code into `mess/`
4. **Review** experiment history with `mlflow ui`
5. **Share** findings through documentation and papers

## Common Tasks

### Extract features from new audio
```bash
# Add audio to data/{dataset}/wav-44/
uv sync --group dev
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

- This is an **open source ML research library** intended for public release
- `mess/` is the core library - keep it clean, well-documented, and modular
- `research/` contains experimentation code (notebooks, scripts)
- Use `uv sync --group dev` for full development setup
- All experiments are tracked in MLflow - run `mlflow ui` to view results
- Focus on scientific rigor, reproducibility, and clean code
- The library should be accessible to researchers and easy to extend
