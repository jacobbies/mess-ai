# MESS-AI

MESS-AI is an open-source Python library for expressive, content-based music similarity research using MERT embeddings.

Core workflow:

`Audio WAV -> MERT features -> proxy targets -> layer discovery -> aspect-layer mapping -> FAISS retrieval`

## Capabilities

- Extract MERT features from local audio datasets
- Generate proxy musical targets and run layer discovery
- Retrieve by cosine similarity, single aspect, weighted multi-aspect, or 5-second clip query
- Build and publish FAISS artifacts for serving (local + S3)
- Train retrieval projection-head baselines on precomputed clip embeddings

## Scope

This is a research-first repository. The core library lives in `mess/`, and workflow scripts live in `scripts/`.

## Installation

Requirements:

- Python 3.11+
- `uv` for dependency and environment management

Use case 1: local development (full stack: extraction, probing, search, training, tests):

```bash
uv sync --group dev
```

Use case 2: slim runtime install:

```bash
pip install mess-ai
```

Runtime search install (EC2/service path):

```bash
pip install "mess-ai[search]"
```

Install from GitHub (search runtime):

```bash
pip install "mess-ai[search] @ git+https://github.com/jacobbies/mess-ai.git"
```

## Data Layout

Default data root is `data/`.

```text
data/
  audio/
    smd/wav-44/*.wav
    maestro/**/*.wav
  embeddings/
    smd-emb/
      raw/*.npy
      segments/*.npy
      aggregated/*.npy
    maestro-emb/
      raw/*.npy
      segments/*.npy
      aggregated/*.npy
  proxy_targets/
    *_targets.npz
  indices/
  metadata/
```

Feature shape contracts:

- `raw`: `[num_segments, 13, time_steps, 768]`
- `segments`: `[num_segments, 13, 768]`
- `aggregated`: `[13, 768]`
- `mess.search.load_features()` returns `(features, track_names)` where `features` is `(n_tracks, feature_dim)`

## Quickstart

1. Extract features:

```bash
uv run python scripts/extract_features.py --dataset smd
```

2. Run layer discovery:

```bash
uv run python scripts/run_probing.py --dataset smd
```

3. Run recommendations:

```bash
uv run python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01"
uv run python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --aspect brightness
uv run python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --aspects "brightness=0.7,phrasing=0.3"
uv run python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --clip-start 30 --k 10
```

Optional MLflow UI:

```bash
uv run mlflow ui
```

## Python API

Track-level retrieval:

```python
from mess import find_similar, load_features, search_by_aspect, search_by_aspects

features_dir = "data/embeddings/smd-emb/aggregated"
features, track_names = load_features(features_dir)

baseline = find_similar("Beethoven_Op027No1-01", features, track_names, k=5)
brightness = search_by_aspect("Beethoven_Op027No1-01", "brightness", features_dir, k=5)
weighted = search_by_aspects(
    query_track="Beethoven_Op027No1-01",
    aspect_weights={"brightness": 0.7, "phrasing": 0.3},
    features_dir=features_dir,
    k=5,
)
```

Clip-level retrieval:

```python
from mess import search_by_clip

results = search_by_clip(
    query_track="Beethoven_Op027No1-01",
    clip_start=30.0,
    features_dir="data/embeddings/smd-emb/segments",
    k=5,
)
```

## Retrieval Training Baseline

1. Build clip index metadata from segment embeddings:

```bash
uv run python scripts/build_clip_index.py \
  --dataset-id smd \
  --segments-dir data/embeddings/smd-emb/segments \
  --output data/metadata/smd_clip_index.csv \
  --assign-splits
```

2. Train projection head:

```bash
uv run python scripts/train_retrieval_ssl.py \
  --clip-index data/metadata/smd_clip_index.csv \
  --output-dir data/indices/retrieval_ssl/smd_run01 \
  --steps 500 --batch-size 64 --layer 0
```

## Production Artifact Workflow (EC2 + S3)

`mess-ai` is built to be imported as a dependency in serving systems where data/artifacts are retrieved from S3 at runtime.

Core helpers:

```python
from mess.search import (
    build_clip_artifact,
    save_artifact,
    upload_artifact_to_s3,
    download_artifact_from_s3,
    load_latest_from_s3,
)
```

Artifact integrity model:

- immutable `artifact_version_id`
- checksum validation for downloaded files
- `latest.json` pointer updated only after upload validation succeeds

Build + publish example:

```bash
uv run python scripts/publish_faiss_index.py \
  --dataset smd \
  --kind clip \
  --s3-bucket <BUCKET> \
  --s3-prefix mess/faiss
```

## Script Status

Maintained scripts:

- `scripts/build_clip_index.py`
- `scripts/extract_features.py`
- `scripts/run_probing.py`
- `scripts/demo_recommendations.py`
- `scripts/publish_faiss_index.py`
- `scripts/train_retrieval_ssl.py`

Research utility (not part of production path):

- `scripts/evaluate_clip_retrieval.py`

Retired scripts (removed from repository):

- `build_layer_indices.py`
- `demo_layer_search.py`
- `evaluate_layer_indices.py`
- `evaluate_similarity.py`

See `scripts/_NEEDS_UPDATE.txt` for retirement details and replacement guidance.

## Drift Corrections (Current Canonical Modules)

When docs drift, treat implementation in `mess/` and tests in `tests/` as source of truth.

Current canonical module locations:

- Aspect registry + resolver: `mess/probing/discovery.py` (`ASPECT_REGISTRY`, `resolve_aspects`)
- Primary search implementation: `mess/search/search.py`
- Public search/runtime exports: `mess/search/__init__.py`

Outdated references you may still see in older notes:

- `mess/search/aspects.py`
- `mess/search/layer_based_recommender.py`

## Development Checks

Tests:

```bash
uv run pytest -v
```

Lint and format checks:

```bash
uv run ruff check .
uv run ruff format --check .
```

Type checking:

```bash
uv run mypy mess
```

## Repository Layout

```text
mess/
  extraction/
  probing/
  search/
  datasets/
  training/
scripts/
tests/
docs/
```

## Notes

- `AGENTS.md` contains implementation-oriented operational guidance for maintainers/agents.
- `docs/PUBLIC_RELEASE_CHECKLIST.md` contains reproducible public-release audit steps.
- `LICENSE` contains project license terms.
