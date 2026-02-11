# MESS-AI

**Music similarity research using empirically validated MERT layer specializations.**

Instead of naive feature averaging, we systematically discover which transformer layers encode specific musical aspects through linear probing with cross-validation.

## Quick Start

```bash
# Install with ML dependencies
uv sync --group dev

# Extract MERT features
python research/scripts/extract_features.py --dataset smd

# Run layer discovery (validates which layers encode what)
python research/scripts/run_probing.py

# View experiments
mlflow ui

# Get recommendations (uses validated layers)
python research/scripts/demo_recommendations.py --track "Beethoven_Op027No1-01"
```

## What This Does

**Layer Discovery**: Systematically probe MERT's 13 layers against 15 musical proxy targets using Ridge regression + 5-fold CV. Results show which layers encode brightness, texture, dynamics, articulation, etc.

**Dynamic Aspect System**: The recommender auto-loads validated layers from discovery results — no hardcoded mappings. Search by 10 user-facing aspects (brightness, texture, warmth, tempo, dynamics, etc.).

**MLflow Tracking**: All experiments logged automatically. Compare hyperparameters, R² scores, and layer validations across runs.

## Architecture

```
mess/                    # Core library
├── extraction/          # MERT feature extraction
├── probing/             # Layer discovery via linear probing
│   ├── discovery.py     # 13 layers × 15 targets = 195 experiments
│   └── proxy_targets.py # Musical aspect ground truth generation
├── search/              # Similarity search
│   ├── aspects.py       # 10 user-facing aspect → target mappings
│   └── layer_based_recommender.py  # Dynamic layer loading
└── datasets/            # SMD, MAESTRO loaders

research/scripts/        # Experiment automation
data/                    # Audio + extracted features
```

## Example Validated Results

From actual discovery runs on SMD dataset:

| Layer | Proxy Target | R² Score | User Aspect |
|-------|-------------|----------|-------------|
| 0 | spectral_centroid | 0.944 | brightness |
| 1 | spectral_rolloff | 0.922 | texture |
| 2 | harmonic_complexity | 0.933 | harmonic_richness |

**15 total targets** across timbre, rhythm, dynamics, harmony, articulation, phrasing.

## Usage

**Python API:**
```python
from mess.search.layer_based_recommender import LayerBasedRecommender

# Loads validated layers from discovery results
recommender = LayerBasedRecommender()

# See what aspects are available (depends on what's been validated)
print(recommender.get_available_aspects())
# ['brightness', 'texture', 'dynamics', ...]

# Search by aspect
recs = recommender.recommend_by_aspect(
    "Beethoven_Op027No1-01",
    aspect="brightness",
    n_recommendations=5
)

# Multi-aspect weighted search
recs = recommender.multi_aspect_recommendation(
    "Beethoven_Op027No1-01",
    aspect_weights={'brightness': 0.6, 'texture': 0.4},
    n_recommendations=5
)
```

**CLI:**
```bash
# Run probing with hyperparameter sweep
python research/scripts/run_probing.py --alpha 0.5 --folds 10 --samples 30

# Extract features with parallelization
python research/scripts/extract_features.py --dataset smd --workers 4

# Check MLflow for results
mlflow ui  # localhost:5000
```

## Dependencies

**Core** (for using pre-extracted features):
- numpy, scipy, scikit-learn, faiss-cpu

**[ml]** (for extraction + research):
- torch, transformers, librosa, mlflow, tqdm, jupyter

```bash
uv sync              # Core only
uv sync --group dev  # Full ML stack
```

## Tech Stack

- **ML**: PyTorch 2.2+, Hugging Face transformers (MERT-v1-95M)
- **Search**: FAISS (sub-millisecond queries), sklearn cosine similarity
- **Tracking**: MLflow (params, metrics, artifacts)
- **Audio**: librosa, torchaudio (24kHz resampling)
- **Dev**: uv (fast package manager), Jupyter

## Performance

- **Similarity search**: <1ms per query (FAISS IndexFlatIP)
- **Feature extraction**: ~2.6 min for 50 tracks (M3 Pro, parallel)
- **Layer discovery**: ~10-15 min full validation (195 experiments)

## Datasets

**SMD** (Saarland Music Dataset): 50 classical piano pieces @ 44kHz
**MAESTRO**: Larger dataset support (infrastructure exists)

Features stored in `data/processed/features/`:
- `raw/`: Full temporal `[segments, 13, time, 768]`
- `segments/`: Time-averaged `[segments, 13, 768]`
- `aggregated/`: Track-level `[13, 768]` ← used for search

## Documentation

See `CLAUDE.md` for:
- Full architecture details
- Development workflow
- Adding new proxy targets
- Research best practices
- MLflow experiment tracking guide

Built for reproducible music similarity research.
