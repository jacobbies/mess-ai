# Testing Strategy

MESS-AI uses two confidence layers:

1. **Pure library tests** (`library`, typically `unit` + focused `integration`) for deterministic logic, contracts, and error handling.
2. **Workflow/system tests** (`system`, usually `workflow`) for end-to-end retrieval behavior and artifact integrity.

To prevent silent drift, deterministic **regression tests** (`regression`) lock key retrieval outputs and tiny-benchmark metrics.

## Quick Reference

```bash
uv run pytest -v                                   # full suite
uv run pytest -m library                           # pure library layer
uv run pytest -m system                            # workflow/system layer
uv run pytest -m unit                              # pure fast unit tests
uv run pytest -m integration                       # multi-module / filesystem slices
uv run pytest -m regression                        # drift / golden checks
uv run pytest -m workflow                          # end-to-end workflow/system tests
uv run pytest --run-gpu                            # include GPU-marked tests
uv run pytest tests/workflows -v                   # workflow folder only
uv run ruff check tests                            # lint tests
uv run mypy mess                                   # type checks for library code
```

## Layer Mapping

### Pure Library Layer

Use this layer to answer: “Does this code do what it claims?”

- Deterministic logic and invariants:
  - path/config parsing, chunk bounds, vector normalization, scoring/ranking helpers
- Interface contracts:
  - required fields, output shapes, dtypes, timestamp alignment, schema stability
- Error behavior:
  - malformed config, missing files, invalid shapes, unsupported modes
- Integration slices:
  - dataset -> extractor -> storage
  - features -> index -> search
  - probing targets -> discovery/evaluation pipeline inputs

### Workflow/System Layer

Use this layer to answer: “Does retrieval behave correctly end to end?”

- Mini pipeline smoke:
  - fixture embeddings -> clip index build -> FAISS artifact -> retrieval query
- Artifact integrity:
  - manifest/checksum/index/vectors/metadata persistence and load validation
- Rerun stability:
  - deterministic outputs for same inputs/seed
- Retrieval usefulness guardrails:
  - tiny curated benchmark with metric thresholds (MRR/Recall@K)

## Key Files

```
tests/
├── fixtures/
│   └── retrieval_tiny.json                 # deterministic tiny retrieval corpus + positives
├── search/
│   ├── test_search.py                      # core search contracts
│   ├── test_faiss_index.py                 # artifact and S3 integrity
│   └── test_retrieval_regression.py        # golden ranking + drift checks
├── evaluation/
│   ├── test_metrics.py                     # metric primitives
│   ├── test_evaluate_script_smoke.py       # report schema smoke
│   └── test_retrieval_mini_benchmark.py    # usefulness guardrail metrics
├── test_setup_demo_data_script.py          # data setup CLI workflow smoke
├── training/
│   └── test_end_to_end_projection_artifact.py  # train->artifact workflow path
└── workflows/
    └── test_retrieval_pipeline_workflow.py  # retrieval workflow + corruption/resume checks
```

## Root Fixtures (`tests/conftest.py`)

| Fixture | Returns | Used by |
|---|---|---|
| `sample_audio_array` | 1s 24kHz f32 sine wave | audio segmentation |
| `sample_embeddings` | Dict of 5 tracks, each `[13, 768]` f32 | search tests |
| `sample_discovery_results` | 13-layer x 3-target dict with known best layers | probing tests |
| `isolated_config` | `MESSConfig` with `project_root=tmp_path` | config/path tests |

## Markers

Defined in `pyproject.toml [tool.pytest.ini_options]`:

- `unit`: fast, isolated, no filesystem/external deps
- `integration`: multi-module or filesystem slices
- `library`: pure-library confidence layer (auto-assigned unless test is system)
- `system`: workflow/system confidence layer (auto-assigned for workflow tests and key CLI flows)
- `regression`: deterministic drift/golden checks
- `workflow`: end-to-end system behavior tests
- `slow`: longer-running checks (>5s)
- `gpu`: requires CUDA/MPS (`--run-gpu`)

## Fixture Policy

- Keep fixtures tiny and deterministic (`tests/fixtures/`).
- Do not use full datasets for CI tests.
- Use tolerance-based assertions for floating-point values.
- Lock shapes, IDs, and ranking identities where possible.

## Dependencies

Install dev/test dependencies and optional runtime extras:

```bash
uv sync --group dev --extra search --extra ml
```
