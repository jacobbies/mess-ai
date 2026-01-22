# MESS-AI: ML Development for Music Similarity Search

Local ML development environment for MERT-based music similarity research using empirically validated layer specializations.

## What Makes This Different

Unlike systems using arbitrary feature combinations, we **systematically discovered** which MERT layers encode specific musical aspects through rigorous cross-validation:

- **Layer 0**: Spectral brightness (R² = 0.944) - Best for timbral similarity
- **Layer 1**: Timbral texture (R² = 0.922) - Instrumental characteristics
- **Layer 2**: Acoustic structure (R² = 0.933) - Resonance patterns

This replaces simple feature averaging (which causes 90%+ similarity between all tracks) with evidence-based recommendations.

## Quick Start

### Setup Environment

**Requirements:**
- Python 3.11.14 (managed via pyenv)
- UV package manager (faster than pip)

```bash
# Install pyenv (if not already installed)
brew install pyenv

# Install Python 3.11.14
pyenv install 3.11.14

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (fast!)
uv pip install -r requirements.txt
```

### Run ML Workflows

```bash
# Extract MERT features from audio
python scripts/extract_features.py --dataset smd

# Get recommendations
python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --aspect spectral_brightness

# Run layer discovery experiments
python scripts/run_probing.py

# Benchmark similarity metrics
python scripts/evaluate_similarity.py
```

## Project Structure

```
mess-ai/
├── pipeline/              # Core ML library
│   ├── extraction/       # MERT feature extraction
│   ├── probing/          # Layer discovery & validation
│   ├── query/            # Recommendation engine
│   ├── search/           # FAISS similarity search
│   ├── datasets/         # Dataset loaders (SMD, MAESTRO)
│   └── metadata/         # Metadata processing
├── scripts/              # Workflow automation
│   ├── extract_features.py
│   ├── demo_recommendations.py
│   ├── run_probing.py
│   └── evaluate_similarity.py
├── notebooks/            # Jupyter experimentation
├── data/                 # Audio files & features
│   ├── smd/             # Saarland Music Dataset
│   ├── maestro/         # MAESTRO Dataset
│   └── processed/       # Extracted MERT embeddings
└── docs/                # Documentation
```

## Usage Examples

### Feature Extraction

Extract MERT embeddings from audio files:

```python
from pipeline.extraction.extractor import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract("path/to/audio.wav")

# features contains:
# - 'raw': Full temporal [segments, 13, time, 768]
# - 'segments': Time-averaged [segments, 13, 768]
# - 'aggregated': Track-level [13, 768]
```

### Layer-Based Recommendations

Get recommendations using validated MERT layers:

```python
from pipeline.query.layer_based_recommender import LayerBasedRecommender

recommender = LayerBasedRecommender()

# Find tracks with similar spectral brightness
results = recommender.recommend(
    reference_track="Beethoven_Op027No1-01",
    aspect="spectral_brightness",
    n_recommendations=5
)

for rec in results['recommendations']:
    print(f"{rec['track_id']}: {rec['similarity']:.4f}")
```

### Layer Discovery

Discover which MERT layers encode specific musical aspects:

```python
from pipeline.probing.layer_discovery import LayerDiscovery

discovery = LayerDiscovery()
results = discovery.run_full_discovery()

# Prints R² scores for each layer/proxy target pair
# Identifies best layers for each musical aspect
```

## Datasets

### Saarland Music Dataset (SMD)
- 50 classical piano pieces at 44kHz
- Pre-extracted MERT features (~94GB)
- Used for layer discovery validation

### MAESTRO Dataset
- Larger dataset for expanded experiments
- Classical piano performances

## Tech Stack

- **Python**: 3.11.14 (via pyenv)
- **Package Manager**: UV (10-100x faster than pip)
- **ML**: PyTorch 2.6+, transformers 4.38+, scikit-learn
- **Search**: FAISS (sub-millisecond similarity queries)
- **Audio**: librosa, soundfile (Apple Silicon optimized)
- **Development**: Jupyter, matplotlib, seaborn

## Performance

- **Similarity Search**: <1ms per query (FAISS IndexFlatIP)
- **Feature Extraction**: ~2.6 minutes for 50-track dataset (M3 Pro)
- **Layer Discovery**: ~10-15 minutes full validation

## Workflow: Local to Production

This repo is optimized for **local ML development** on Apple Silicon (M3 Pro). The typical workflow:

1. **Local Development** (this repo)
   - Experiment with layer discovery
   - Test new similarity metrics
   - Extract features from datasets
   - Run probing experiments

2. **Sync to Production**
   - Push processed features to EC2/S3
   - Deploy validated models to production API
   - Share research findings

## Research Focus

Current areas of exploration:

- **Layer Specialization**: Systematic discovery of MERT layer functions
- **Proxy Target Validation**: Cross-validation of musical aspect mappings
- **Similarity Metrics**: Comparing cosine, euclidean, dot product
- **FAISS Optimization**: Testing IVF, HNSW indices for larger datasets
- **Model Fine-tuning**: Domain-specific training on SMD dataset

## Contributing

This is a personal research environment. For production API, see the EC2 deployment.

## Citation

If you use the layer discovery methodology or validated MERT layer mappings:

```
MESS-AI: Empirically Validated MERT Layer Specializations
Layer 0: Spectral Brightness (R² = 0.944)
Layer 1: Timbral Texture (R² = 0.922)
Layer 2: Acoustic Structure (R² = 0.933)
```

Built with scientific rigor for music similarity research.
