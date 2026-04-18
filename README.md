# MESS-AI

MESS-AI is a Python library for content-based classical music passage retrieval using MERT embeddings.

Pipeline:
`WAV audio -> embeddings -> proxy targets -> layer discovery -> aspect-aware retrieval`

## Start Here (Step-by-Step)

### 1. Install

Requirements:
- Python 3.11+
- `uv`

Full local workflow (recommended for this repo):
```bash
uv sync --group dev --extra search --extra ml
```

Runtime-only installs:
```bash
pip install mess-ai
pip install "mess-ai[search]"
pip install "mess-ai[ml]"
pip install "mess-ai[search,ml]"
```

### 2. Add Demo Audio (One Command)

Fastest path:

```bash
uv run python scripts/setup_demo_data.py --data-root data
```

This generates 3 synthetic WAV tracks and `data/metadata/smd_metadata.csv`.
Details: see [`data/README.md`](data/README.md).

### 3. Put Real Audio In Expected Paths (Optional)

Default data root is `data/`:

```text
data/
  audio/
    smd/wav-44/*.wav
    maestro/**/*.wav
```

Track IDs come from filename stems (for example `Beethoven_Op027No1-01`).

### 4. Extract Embeddings

```bash
uv run python scripts/extract_features.py --dataset smd
```

First run will download MERT weights (`m-a-p/MERT-v1-95M`).
Outputs are written under:

```text
data/embeddings/smd-emb/
  raw/
  segments/
  aggregated/
```

### 5. Run Layer Discovery (Recommended)

```bash
uv run python scripts/run_probing.py --dataset smd
```

This computes proxy targets and writes validated aspect/layer mappings used by aspect search.

### 6. Run Retrieval

Baseline track-level similarity:
```bash
uv run python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01"
```

Single-aspect retrieval:
```bash
uv run python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --aspect brightness
```

Weighted multi-aspect retrieval:
```bash
uv run python scripts/demo_recommendations.py \
  --track "Beethoven_Op027No1-01" \
  --aspects "brightness=0.7,phrasing=0.3"
```

Clip query:
```bash
uv run python scripts/demo_recommendations.py \
  --track "Beethoven_Op027No1-01" \
  --clip-start 30 \
  --k 10
```

Optional experiment UI:
```bash
uv run mlflow ui
```

## Python API (Minimal)

```python
from mess import find_similar, load_features, search_by_aspect, search_by_clip
from mess.search import find_latest_artifact_dir

features_dir = "data/embeddings/smd-emb/aggregated"
features, track_names = load_features(features_dir)

baseline = find_similar("Beethoven_Op027No1-01", features, track_names, k=5)
brightness = search_by_aspect("Beethoven_Op027No1-01", "brightness", features_dir, k=5)

clip_artifact_dir = find_latest_artifact_dir("data/indices", artifact_name="clip_index")
clip_results = search_by_clip(
    artifact=clip_artifact_dir,
    track_id="Beethoven_Op027No1-01",
    start_sec=30.0,
    k=5,
)
```

## Data Contracts (Do Not Break)

- `raw`: `[num_segments, 13, time_steps, 768]`
- `segments`: `[num_segments, 13, 768]`
- `aggregated`: `[13, 768]`
- `mess.search.load_features()` returns `(features, track_names)` where `features` is `(n_tracks, feature_dim)` for FAISS search

## Repository Map

- `mess/`: library modules (`datasets`, `extraction`, `probing`, `search`)
- `scripts/`: CLI workflows
- `tests/`: contract and behavior tests
- `docs/`: architecture and research context

When docs drift, trust implementation in `mess/` and tests in `tests/`.

## Script Status

Source of truth for lifecycle status:
- `scripts/script_status.json`

Maintained scripts:
- `scripts/setup_demo_data.py`
- `scripts/extract_features.py`
- `scripts/run_probing.py`
- `scripts/demo_recommendations.py`

## Development Checks

```bash
uv run ruff check .
uv run pytest -v
uv run mypy --follow-imports skip \
  mess/search/faiss_index.py \
  mess/search/__init__.py \
  mess/datasets/clip_index.py
```
