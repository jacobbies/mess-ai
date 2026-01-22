# CLAUDE.md - ML Development Environment

## Project Overview

**MESS-AI** is a local ML development environment for music similarity research using MERT (Music Understanding Model with Large-Scale Self-Supervised Training) embeddings.

### Purpose
This repo is optimized for **local ML experimentation** on Apple Silicon (M3 Pro), NOT for production deployment. Production API lives on EC2.

### Core Focus
- Feature extraction from audio using MERT
- Layer discovery and validation (finding which layers encode which musical aspects)
- Similarity search algorithm development
- Dataset preprocessing and analysis
- Research experimentation via Jupyter notebooks

## Project Structure

```
mess-ai/
├── pipeline/              # Core ML library
│   ├── extraction/       # MERT feature extraction
│   ├── probing/          # Layer discovery & validation
│   ├── query/            # Recommendation engine
│   ├── search/           # FAISS similarity search
│   ├── datasets/         # Dataset loaders (SMD, MAESTRO)
│   ├── metadata/         # Metadata processing
│   └── marble/           # External multi-task learning framework
├── scripts/              # CLI workflow automation
│   ├── extract_features.py
│   ├── demo_recommendations.py
│   ├── run_probing.py
│   └── evaluate_similarity.py
├── notebooks/            # Jupyter experimentation
├── data/                 # Audio files & extracted features
│   ├── smd/             # Saarland Music Dataset
│   ├── maestro/         # MAESTRO Dataset
│   └── processed/       # Pre-extracted MERT embeddings (~94GB)
└── docs/                # Research documentation
```

## Key Scientific Discoveries

Through systematic layer discovery experiments, we've validated:

- **Layer 0**: Spectral brightness (R² = 0.944)
- **Layer 1**: Timbral texture (R² = 0.922)
- **Layer 2**: Acoustic structure (R² = 0.933)

These specializations replace naive feature averaging and enable evidence-based similarity search.

## Development Workflow

### 1. Feature Extraction
```bash
# Extract MERT embeddings from audio
python scripts/extract_features.py --dataset smd

# Output: data/processed/features/aggregated/*.npy
# Format: [13 layers, 768 dims] per track
```

### 2. Layer Discovery
```bash
# Run probing experiments to validate layer specializations
python scripts/run_probing.py

# Output: pipeline/probing/layer_discovery_results.json
# Contains R² scores for layer/proxy target pairs
```

### 3. Similarity Search
```bash
# Test recommendations using validated layers
python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01"

# Uses LayerBasedRecommender with empirically validated mappings
```

### 4. Experimentation
```bash
# Launch Jupyter for exploration
jupyter notebook notebooks/

# Suggested notebooks:
# - layer_discovery_analysis.ipynb
# - similarity_benchmarks.ipynb
# - feature_visualization.ipynb
```

## Core Components

### Pipeline Library

The `pipeline/` directory is a Python library (not a service) with these modules:

**extraction/**
- `extractor.py`: MERT feature extraction from audio
- `config.py`: Extraction configuration (sample rate, segment duration, etc.)

**probing/**
- `layer_discovery.py`: Systematic discovery of layer specializations
- `proxy_targets.py`: Musical aspect proxy targets for validation
- `layer_discovery_results.json`: Empirical validation results

**query/**
- `layer_based_recommender.py`: Recommendation engine using validated layers
- `intelligent_query_engine.py`: Natural language query processing

**search/**
- `faiss_index.py`: FAISS index wrapper for similarity search
- `similarity.py`: Similarity computation (cosine, euclidean, etc.)
- `diverse_similarity.py`: Diverse recommendation algorithms
- `cache.py`: Feature caching utilities

**datasets/**
- `base.py`: Base dataset class
- `smd.py`: Saarland Music Dataset loader
- `maestro.py`: MAESTRO dataset loader
- `factory.py`: Dataset factory pattern

**metadata/**
- `processor.py`: Metadata extraction and processing
- `maestro_csv_parser.py`: MAESTRO CSV parsing

**marble/**
- External multi-task learning framework (1000+ files)
- Used for reference, may be separated in future refactors

## Data Flow

```
Audio Files (.wav)
    ↓
MERT Feature Extraction (extractor.py)
    ↓
Embeddings [13 layers, 768 dims]
    ↓
Layer Discovery (probing/)
    ↓
Validated Layer Mappings
    ↓
Similarity Search (FAISS)
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

- **ML Framework**: PyTorch 2.6+ (MPS acceleration on Apple Silicon)
- **Transformers**: Hugging Face transformers 4.38+ (MERT model)
- **Audio**: librosa, soundfile (optimized for M3)
- **Search**: FAISS (CPU version, sub-millisecond queries)
- **Scientific**: scikit-learn, numpy, pandas
- **Development**: Jupyter, matplotlib, seaborn

## Best Practices

### Code Organization
- Keep pipeline/ as a clean Python library (no API/service code)
- Use scripts/ for CLI automation and batch processing
- Use notebooks/ for exploration and visualization
- Document discoveries in docs/

### Development Patterns
- Run experiments in notebooks first
- Productionize proven code into pipeline/ modules
- Use scripts/ for repeatable workflows
- Sync validated features/models to EC2 for production

### Data Management
- Keep raw audio in data/{dataset}/wav-44/
- Store processed features in data/processed/features/
- Never commit large binary files (use .gitignore)
- Document feature extraction parameters

### Research Workflow
1. **Explore** in Jupyter notebooks
2. **Validate** with probing experiments
3. **Productionize** proven code into pipeline/
4. **Sync** to EC2/S3 for production use

## Local to Production Sync

This repo handles:
- Feature extraction (compute-intensive, needs M3 Pro)
- Layer discovery experiments
- Algorithm development
- Dataset preprocessing

Production EC2 handles:
- REST API serving
- Web interface
- Audio streaming
- Public-facing queries

**Sync artifacts**: Processed features, validated models, research findings

## Common Tasks

### Extract features from new audio
```bash
# Add audio to data/{dataset}/wav-44/
python scripts/extract_features.py --dataset {dataset}
```

### Validate new layer hypothesis
```python
# Add proxy target to pipeline/probing/proxy_targets.py
# Run discovery
python scripts/run_probing.py
```

### Test new similarity metric
```python
# Update pipeline/search/similarity.py
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

**🚧 In Progress:**
- Model fine-tuning on SMD dataset for domain-specific similarity
- Advanced FAISS indices (IVF, HNSW) for even larger datasets
- Expanded proxy target validation

**📋 Planned:**
- Multi-modal fusion (audio + score + metadata)
- User preference learning
- Expanded dataset support beyond classical music

## Notes for Claude

- This is an ML research environment, NOT a production service
- No backend/API code should exist in this repo
- Focus on experimentation, validation, and discovery
- Keep code clean, modular, and well-documented
- Prioritize scientific rigor over speed-to-market
