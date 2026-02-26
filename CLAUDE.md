# CLAUDE.md - ML Research Environment

## Project Overview

**MESS-AI** is a Python library for music similarity research using MERT (Music Understanding Model with Large-Scale Self-Supervised Training) embeddings.

### Architecture
- **mess/** - Core ML library (feature extraction, similarity search, layer discovery)
- **scripts/** - CLI workflow automation scripts
- **data/** - Audio datasets and extracted features (239GB total)

### Purpose
- ML research library for music understanding and similarity
- Focus on layer-wise feature analysis and evidence-based similarity search
- Platform-agnostic (works on Linux, macOS, Windows with appropriate GPU/CPU backends)

### Core Focus
- Feature extraction from audio using MERT
- Layer discovery and validation (finding which layers encode which musical aspects)
- **Learned expressive similarity** — moving beyond raw cosine on frozen embeddings toward retrieval geometry optimized for expressive musical attributes
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
├── tests/                    # pytest test suite (see tests/tests.md)
│   ├── datasets/             # Dataset loader tests
│   ├── extraction/           # Audio & storage tests
│   ├── probing/              # Discovery & registry tests
│   └── search/               # FAISS search tests
├── scripts/                  # CLI workflow automation
├── notebooks/                # Jupyter exploration
├── data/                     # Audio files & extracted features (239GB total)
│   ├── audio/                # Raw audio files (124GB)
│   │   ├── smd/              # Saarland Music Dataset (50 tracks)
│   │   └── maestro/          # MAESTRO Dataset (~1200 tracks)
│   ├── embeddings/           # Pre-extracted MERT embeddings (115GB)
│   │   ├── smd-emb/          # 94GB (raw, segments, aggregated)
│   │   ├── maestro-emb/      # 21GB (raw, segments, aggregated)
│   │   └── demo-emb/         # Demo subset
│   ├── proxy_targets/        # Computed proxy target features (135MB)
│   ├── indices/              # FAISS index files
│   ├── metadata/             # Dataset metadata and manifests
│   ├── waveforms/            # Cached waveform arrays
│   └── models/               # Future training checkpoints
├── mlruns/                   # MLflow experiment tracking (gitignored)
├── docs/                     # Research documentation
└── pyproject.toml            # Package configuration
```

## Dependency Architecture

The library bundles all dependencies together for a complete ML research environment:

**Core Dependencies:**
- **Scientific**: numpy, scipy, scikit-learn, pandas
- **ML Framework**: PyTorch 2.10+ (with MPS/CUDA support), torchcodec
- **Transformers**: Hugging Face transformers 4.38+, safetensors
- **Audio Processing**: librosa 0.10+, soundfile, nnaudio
- **Search**: FAISS (CPU version)
- **Experiment Tracking**: MLflow 2.10+
- **Development**: Jupyter, matplotlib, seaborn, ipywidgets, tqdm

**Platform-Specific Features:**
- Linux: PyTorch with CUDA 12.8 support (via pytorch-cu128 index)
- macOS: PyTorch with MPS (Metal Performance Shaders) acceleration

**Testing:**
- pytest 8+, pytest-cov, pytest-mock (in `[dependency-groups] dev`)

**Installation:**
```bash
# Install all dependencies (including test deps)
uv sync --group dev

# Or using pip
pip install -e .
```

## Layer Discovery System

**22 Proxy Targets Across 7 Musical Dimensions:**
- **Timbre** (audio-derived): spectral_centroid, spectral_rolloff, spectral_bandwidth, zero_crossing_rate
- **Rhythm** (audio-derived): tempo, onset_density
- **Dynamics** (audio-derived): dynamic_range, dynamic_variance, crescendo_strength, diminuendo_strength
- **Harmony** (audio-derived): harmonic_complexity
- **Articulation** (audio-derived): attack_slopes, attack_sharpness
- **Phrasing** (audio-derived): phrase_regularity, num_phrases
- **Expression** (MIDI-derived): rubato, velocity_mean, velocity_std, velocity_range, articulation_ratio, tempo_variability, onset_timing_std

Expression targets are optional — computed only when MIDI data is available for a track. Both SMD (52 MIDI files) and MAESTRO (1,276 MIDI files) have full coverage.

**13 User-Facing Aspects** (in `mess/probing/discovery.py` ASPECT_REGISTRY):
- brightness, texture, warmth, tempo, rhythmic_energy, dynamics, crescendo, harmonic_richness, articulation, phrasing
- rubato, expressiveness, legato *(MIDI-derived)*

**Methodology:**
- Ridge regression with 5-fold cross-validation on frozen MERT embeddings
- 13 layers × 22 targets = 286 probing experiments per run (targets with missing data are probed on available tracks only)
- R² > 0.8 → "high confidence" (excellent for production use)
- R² > 0.5 → "medium confidence" (experimental)

**Example Validated Results** (from actual discovery runs):
- Layer 0 → spectral_centroid (R² = 0.944) → maps to "brightness" aspect
- Layer 1 → spectral_rolloff (R² = 0.922) → maps to "texture" aspect
- Layer 2 → harmonic_complexity (R² = 0.933) → maps to "harmonic_richness" aspect

These specializations replace naive feature averaging and enable evidence-based similarity search.

## Development Setup

### Installation

```bash
# Install all dependencies (including test deps)
uv sync --group dev

# Or using pip
pip install -e .
```

All dependencies are installed together. The library includes everything needed for:
- Feature extraction from audio
- Layer discovery and validation
- Similarity search and recommendations
- Jupyter notebook experimentation
- MLflow experiment tracking

### Running Tests

```bash
uv run pytest -v                          # all tests verbose
uv run pytest --cov=mess --cov-report=term-missing  # with coverage
uv run pytest -m unit                     # only fast unit tests
uv run pytest tests/test_config.py -v     # single module
```

99 tests, ~3.5s, no model loading or real audio I/O. See `tests/tests.md` for full details.

## Development Workflow

### 1. Feature Extraction
```bash
# Extract MERT embeddings from audio
python scripts/extract_features.py --dataset smd

# Output: data/embeddings/<dataset>-emb/
#   - raw/: Full temporal features [segments, 13, time, 768]
#   - segments/: Time-averaged [segments, 13, 768]
#   - aggregated/: Track-level [13, 768] - used for similarity search
# MLflow: Logs to "feature_extraction" experiment with timing metrics
```

### 2. Layer Discovery
```bash
# Run probing experiments to validate layer specializations
python scripts/run_probing.py

# Output: mess/probing/layer_discovery_results.json
# Contains R² scores for 13 layers × 15 proxy targets (195 experiments)
# MLflow: Logs to "layer_discovery" experiment with:
#   - Params: alpha, n_folds, n_samples, dataset
#   - Metrics: 195 R²/correlation/RMSE values
#   - Artifacts: layer_discovery_results.json
#   - Best layers per target with confidence ratings

# Hyperparameter tuning examples:
python scripts/run_probing.py --alpha 0.5 --folds 10 --samples 30
python scripts/run_probing.py --experiment "ridge_alpha_sweep"
```

### 3. Similarity Search
```bash
# Test recommendations using validated layers (dynamically loaded from discovery)
python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01"

# The recommender auto-loads layer_discovery_results.json and resolves aspects
# Available aspects depend on what's been validated (run discovery first)
# Examples: brightness, texture, warmth, dynamics, articulation, phrasing
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

**extraction/**
- `extractor.py`: MERT feature extraction from audio (with batching & caching)
- `audio.py`: Audio loading and preprocessing
- `pipeline.py`: Feature extraction pipeline orchestration
- `storage.py`: Feature storage and caching utilities

**probing/**
- `discovery.py`: Systematic discovery of layer specializations via linear probing
  - `LayerDiscoverySystem`: 13 layers × 15 proxy targets with Ridge regression + CV
  - Logs all params/metrics to MLflow if run is active
  - `discover_and_save()`: One-liner for discovery + save + return best layers
- `proxy_targets.py`: Musical aspect proxy targets for validation
- `layer_discovery_results.json`: Empirical validation results (gitignored, generated by probing)

**search/**
- `aspects.py`: Aspect registry mapping user concepts to probing targets
  - `ASPECT_REGISTRY`: 10 searchable aspects (brightness, texture, warmth, etc.)
  - `resolve_aspects()`: Auto-loads discovery results and maps aspects to best layers
- `layer_based_recommender.py`: Dynamic aspect-based recommendation
  - Loads validated layers from `layer_discovery_results.json` at init
  - Zero hardcoded layer mappings — fully data-driven
  - Graceful fallback when no results exist
- `search.py`: Simplified similarity search interface
- `cache.py`: Feature caching utilities

**datasets/**
- `base.py`: Base dataset class with common functionality
- `smd.py`: Saarland Music Dataset loader
- `maestro.py`: MAESTRO dataset loader
- `factory.py`: Dataset factory pattern

## Data Flow

```
Audio Files (.wav)
    ↓
MERT Feature Extraction (mess.extraction)
    ↓
Embeddings [13 layers, 768 dims]
    ↓
    ├── Proxy Target Computation (mess.probing.proxy_targets)
    │       ↓
    │   Layer Discovery (mess.probing.discovery)
    │       ↓
    │   Validated Layer Mappings (layer_discovery_results.json)
    │
    └── [Future] Learned Projection Head (mess.ranking)
            ↓  trained via contrastive loss using proxy targets as teacher
            ↓
        Projected Embeddings [d' dims, ANN-indexable]
    ↓
Similarity Search (mess.search)  ← cosine today, learned geometry future
    ↓
Recommendations
```

## Dataset Structure

```
data/
├── audio/                      # Raw audio files (124GB)
│   ├── smd/                    # Saarland Music Dataset
│   │   ├── wav-44/            # 50 audio files at 44kHz (MERT compatible)
│   │   ├── csv/               # Performance annotations
│   │   └── midi/              # Symbolic representations
│   └── maestro/               # MAESTRO Dataset (~1200 recordings)
│       ├── 2004-2018/         # Years of recordings
│       ├── *.csv              # Metadata manifests
│       └── *.json             # Dataset information
├── embeddings/                 # MERT embeddings (115GB)
│   ├── smd-emb/               # SMD embeddings (94GB)
│   │   ├── raw/               # Full temporal [segments, 13, time, 768]
│   │   ├── segments/          # Time-averaged [segments, 13, 768]
│   │   └── aggregated/        # Track-level [13, 768]
│   ├── maestro-emb/           # MAESTRO embeddings (21GB)
│   │   ├── raw/
│   │   ├── segments/
│   │   └── aggregated/
│   └── demo-emb/              # Small demo subset
├── proxy_targets/             # Computed proxy features (135MB)
│   └── *.npy                  # Per-dataset proxy target arrays
├── indices/                   # FAISS index files
├── metadata/                  # Dataset manifests and track lists
├── waveforms/                 # Cached waveform arrays
└── models/                    # Future training checkpoints
```

## Performance Characteristics

- **Feature Extraction**: ~2.6 minutes for 50-track dataset (M3 Pro)
- **Similarity Search**: <1ms per query (FAISS IndexFlatIP)
- **Layer Discovery**: ~10-15 minutes full validation (195 probing experiments)
- **Storage Requirements**:
  - Audio files: 124GB (SMD + MAESTRO)
  - Embeddings: 115GB (94GB SMD + 21GB MAESTRO)
  - Proxy targets: 135MB
  - Total: ~239GB

## Tech Stack

- **Package Manager**: uv (fast Python package manager)
- **ML Framework**: PyTorch 2.10+ (MPS/CUDA/CPU), torchcodec 0.10+
- **Transformers**: Hugging Face transformers 4.38+ (MERT model)
- **Audio**: librosa 0.10+, soundfile, nnaudio 0.3.4+
- **Search**: FAISS (CPU version, sub-millisecond queries)
- **Scientific**: scikit-learn, numpy, scipy, pandas
- **Experiment Tracking**: MLflow 2.10+ (metrics, parameters, artifacts)
- **Development**: Jupyter, matplotlib, seaborn, ipywidgets, tqdm

## Best Practices

### Code Organization
- Keep `mess/` as a clean, well-documented Python library
- Use `scripts/` for CLI automation and batch processing
- Use `notebooks/` for exploration and visualization
- Document discoveries in `docs/`

### Development Patterns
- Productionize proven code into `mess/` modules
- Use scripts for repeatable workflows and batch processing
- Keep the `mess/` library clean and well-tested
- Write tests for new code — mirror source structure in `tests/` (e.g., `mess/search/` → `tests/search/`)
- Unit tests should avoid loading the MERT model; mock heavy deps, use `tmp_path` for I/O
- Run `uv run pytest` before committing

### Data Management
- Keep raw audio in `data/audio/{dataset}/`
- Store processed features in `data/embeddings/{dataset}-emb/`
- Proxy targets go in `data/proxy_targets/`
- Never commit large binary files (use .gitignore)
- Document feature extraction parameters in experiment tracking

### Research Workflow
1. **Explore** in Jupyter notebooks
2. **Validate** with probing experiments (tracked in MLflow)
3. **Productionize** proven code into `mess/`
4. **Review** experiment history with `mlflow ui`
5. **Share** findings through documentation and papers

## Common Tasks

### Extract features from new audio
```bash
# Add audio to data/audio/{dataset}/
python scripts/extract_features.py --dataset {dataset}
```

### Validate new layer hypothesis
```bash
# 1. Add proxy target to mess/probing/proxy_targets.py (if needed)
# 2. Add to SCALAR_TARGETS in mess/probing/discovery.py
# 3. Run discovery
python scripts/run_probing.py

# 4. Check MLflow for R² scores
mlflow ui

# 5. If R² > 0.5, the aspect is automatically available in the recommender
# No manual updates needed — it's all dynamic
```

### Test new similarity metric
```bash
# Update mess/search/similarity.py
# Benchmark
python scripts/evaluate_similarity.py
```

### Experiment with recommendations
```bash
# Direct Python usage
python scripts/demo_recommendations.py --track {track_id} --aspect {aspect}

# Or in Jupyter for visualization
```

## Development Status

**Current baseline (working):**
- Frozen MERT embeddings + cosine similarity via FAISS IndexFlatIP
- Aspect-weighted search using layer discovery (proxy target R2 → best layer → cosine in that layer)

**Next research milestone: Learned Expressive Similarity**

The core research problem: cosine on frozen embeddings retrieves by *generic audio similarity* (timbre, content, recording conditions), not *expressive similarity* (rubato, articulation, dynamics, phrasing). Moving from cosine to a learned scoring function is the key unlock.

Experiment sequence (ordered by priority):
1. **MIDI-derived expression targets** — Add rubato, velocity dynamics, articulation ratio targets from MIDI data. These capture *how* notes are played rather than *what* is played, providing a cleaner teacher signal for everything downstream. Key empirical question: do MERT layers encode expression separately from content? **(in progress)**
2. **Human eval set** — Build 100-300 triplet judgments ("which clip is more expressively similar?") as a stable external ground truth. Without this, optimization against proxy targets risks circularity.
3. **Baseline recall measurement** — Measure cosine Recall@K against both proxy-derived and human labels. This establishes whether the bottleneck is retrieval geometry or candidate generation.
4. **Diagonal weighting** — Learn per-dimension or per-layer weights on the existing 768-dim embeddings. Simplest upgrade, extends current aspect-weighted search, highly interpretable.
5. **Linear projection head** — Train g(e) = Ae with InfoNCE/SupCon using proxy targets as teacher supervision. Precompute projected embeddings, serve with ANN. Still single-stage.
6. **Small MLP projection** — Only if linear saturates. Same training regime, slightly more capacity.
7. **Two-stage reranking** — Only if Recall@K is high but ranking quality inside top-K is poor. Pairwise reranker (RankNet-style) over interaction features on top-K candidates.

Future MIDI-enabled workstreams (after expression targets are validated):
- **Cross-performance passage pairing** — Use MIDI alignment to find same-passage-different-performer pairs for contrastive training (natural hard positives/negatives for expressive similarity).
- **Content-invariant embedding learning** — Train projection head to be invariant to content (conditioned on MIDI score) but sensitive to expression.

Key decision rules:
- If Recall@K is low → improve embedding geometry (steps 3-5), not reranking
- If proxy target quality is poor (human eval disagrees with proxy labels) → improve proxy targets before learning on them
- If linear projection saturates → try MLP, then consider reranking
- Two-stage is only justified when the scorer is non-decomposable AND Recall@K is high enough that reranking can matter

**Planned infrastructure:**
- `mess/ranking/` — Projection heads, contrastive training loops, evaluation harness
- `mess/evaluation/` — Leakage-safe splits, Recall@K / MRR@K / nDCG@K computation, paired bootstrap significance testing
- `data/eval/` — Human triplet judgments and proxy-derived relevance labels

## Known Issues

**Scripts needing updates** (see `scripts/_NEEDS_UPDATE.txt`):
- `build_layer_indices.py` - Uses old LayerIndexBuilder API
- `demo_layer_search.py` - Uses old LayerIndexBuilder API
- `evaluate_layer_indices.py` - Uses old LayerIndexBuilder API
- `evaluate_similarity.py` - Uses old FAISSIndex and SimilarityComputer APIs

These scripts were experimental and not validated. They can be:
1. Updated to use the new simplified search module
2. Deleted if not needed
3. Rewritten for specific research experiments

The main demo script (`demo_recommendations.py`) works correctly with the current API.

## Notes for Claude

- This is an **open source ML research library** intended for public release
- `mess/` is the core library - keep it clean, well-documented, and modular
- `scripts/` contains CLI automation, `notebooks/` contains experimentation code
- Use `uv sync` for full development setup (all deps bundled together)
- All experiments are tracked in MLflow - run `mlflow ui` to view results
- Focus on scientific rigor, reproducibility, and clean code
- The library should be accessible to researchers and easy to extend
- Total storage requirement: ~239GB (124GB audio + 115GB embeddings)
