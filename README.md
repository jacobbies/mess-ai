# MESS-AI

MESS-AI is a Python library for music similarity research with MERT embeddings.
It focuses on expressive, content-based retrieval: extract features, probe layer behavior, and run similarity search with FAISS.

## What You Can Do

- Extract MERT features from local audio datasets
- Build proxy musical targets and run layer discovery
- Search by cosine similarity, single aspect, or weighted multi-aspect queries
- Track extraction and probing runs with MLflow

## Project Scope

This is a research-first repository. The core library is in `mess/`, with scripts in `scripts/` for common workflows.

Current pipeline:

`Audio WAV -> MERT features -> proxy targets -> layer discovery -> aspect-layer mapping -> FAISS search`

## Requirements

- Python 3.11+
- `uv` for environment and dependency management
- Local access to supported datasets (SMD and/or MAESTRO)

## Installation

```bash
uv sync --group dev
```

Install from another repository (pinned Git commit):

```bash
pip install "mess-ai @ git+https://github.com/jacobbies/mess-ai.git@ac675b6"
```

Run commands with `uv run` so the project environment is used:

```bash
uv run python --version
```

## Data Layout

Default data root is `data/`. Expected structure:

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
```

## Quickstart

1. Extract features:

```bash
uv run python scripts/extract_features.py --dataset smd
```

2. Run probing / layer discovery:

```bash
uv run python scripts/run_probing.py
```

3. Start MLflow UI (optional):

```bash
uv run mlflow ui
```

4. Run recommendations:

```bash
uv run python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01"
uv run python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --aspect brightness
uv run python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --aspects "brightness=0.7,phrasing=0.3"
```

## Python API Example

```python
from mess.search.search import find_similar, load_features, search_by_aspect, search_by_aspects

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

Search-only import path (lightest runtime surface for recommendation services):

```python
from mess.search.search import load_features, find_similar, search_by_clip
```

## Repository Layout

```text
mess/
  extraction/   # Audio loading, segmentation, and MERT feature extraction
  probing/      # Proxy targets and layer discovery
  search/       # FAISS search and aspect-aware retrieval
  datasets/     # Dataset abstractions (SMD, MAESTRO)
scripts/
  extract_features.py
  run_probing.py
  demo_recommendations.py
tests/
```

## Script Status

Maintained:
- `scripts/extract_features.py`
- `scripts/run_probing.py`
- `scripts/demo_recommendations.py`

Experimental / needs update:
- `scripts/build_layer_indices.py`
- `scripts/demo_layer_search.py`
- `scripts/evaluate_layer_indices.py`
- `scripts/evaluate_similarity.py`
- `scripts/evaluate_clip_retrieval.py` (research evaluation utility)

See `scripts/_NEEDS_UPDATE.txt` for details.

## Testing

```bash
uv run pytest -v
```

For focused runs:

```bash
uv run pytest tests/search/test_search.py -v
uv run pytest tests/probing/test_discovery.py -v
```

## Linting and Type Checking

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy mess
```

To auto-fix lint/import issues and format code:

```bash
uv run ruff check . --fix
uv run ruff format .
```

## Notes

- `scripts/_NEEDS_UPDATE.txt` lists older scripts that are currently outdated.
- `AGENTS.md` is the implementation-oriented guide and source of operational conventions for this repo.
- `docs/PUBLIC_RELEASE_CHECKLIST.md` contains reproducible public-release audit steps.

## License

Add your chosen open-source license in a `LICENSE` file before publishing.
