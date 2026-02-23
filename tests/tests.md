# Testing Infrastructure

## Quick Reference

```bash
uv run pytest -v                          # all tests verbose
uv run pytest --cov=mess --cov-report=term-missing  # with coverage
uv run pytest -m unit                     # only unit markers
uv run pytest tests/test_config.py -v     # single module
uv run pytest --run-gpu                   # include GPU-marked tests
uv run ruff check .                       # lint checks
uv run ruff format --check .              # format checks
uv run mypy mess                          # type checks
```

## Structure

```
tests/
├── conftest.py              # Root fixtures + GPU skip hook
├── test_config.py           # MESSConfig (23 tests)
├── datasets/
│   ├── conftest.py          # autouse: saves/restores DatasetFactory._datasets
│   ├── test_base.py         # BaseDataset via ConcreteTestDataset subclass (13)
│   ├── test_factory.py      # DatasetFactory registration/lookup (7)
│   ├── test_smd.py          # SMDDataset properties (4)
│   └── test_maestro.py      # MAESTRODataset properties (4)
├── extraction/
│   ├── conftest.py          # long_audio_array fixture (10s sine)
│   ├── test_audio.py        # segment_audio pure logic, load_audio mocked (10)
│   ├── test_extractor.py    # cache paths, OOM recovery, delegation, safe errors (6)
│   ├── test_pipeline.py     # discovery fallback, worker statuses, run orchestration (7)
│   └── test_storage.py      # path helpers, save/load roundtrip (16)
├── probing/
│   ├── conftest.py
│   ├── test_discovery.py    # _probe_single, discover flow, best_layers, resolve_aspects, registries (17)
│   └── test_targets.py      # create_target_dataset nested/case-insensitive discovery (1)
└── search/
    └── test_search.py       # track/clip/aspect FAISS search behavior (21)
```

## Root Fixtures (tests/conftest.py)

| Fixture | Returns | Used by |
|---|---|---|
| `sample_audio_array` | 1s 24kHz f32 sine wave | audio segmentation |
| `sample_embeddings` | Dict of 5 tracks, each `[13, 768]` f32 | search tests |
| `sample_discovery_results` | 13-layer x 3-target dict with known best layers (L0=0.95, L5=0.85, L12=0.45) | probing tests |
| `isolated_config` | `MESSConfig` with `project_root=tmp_path` | config/path tests |

## Custom Markers

Defined in `pyproject.toml [tool.pytest.ini_options]`:
- `unit` — fast, no I/O or external deps
- `integration` — filesystem or multi-module
- `slow` — >5s (model loading, large data)
- `gpu` — requires CUDA/MPS; auto-skipped unless `--run-gpu`

## Design Principles

- **No model loading in unit tests**: Heavy MERT model paths are mocked or bypassed
- **tmp_path everywhere**: Tests never touch real `data/` directory
- **No `__init__.py`**: pytest discovers test dirs without them
- **Factory isolation**: `datasets/conftest.py` auto-restores `_datasets` dict per test
- **`_probe_single` tested via `object.__new__`**: Bypasses `LayerDiscoverySystem.__init__` (which needs real dataset files)
- **Nested dataset safety**: discovery and target-generation tests cover recursive `.wav` discovery (MAESTRO-style directory trees)

## Dependencies

In `pyproject.toml` `[dependency-groups] dev`:
- `pytest>=8.0`
- `pytest-cov>=5.0`
- `pytest-mock>=3.14`
- `ruff`
- `mypy`

Install: `uv sync --group dev`
