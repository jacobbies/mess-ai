# PR/TASK HANDOFF
- Task/PR: Refactor testing into two confidence layers and add drift/usefulness guards
- Branch: `main`
- Status: `completed`
- Depends on: current worktree state

## Objective
- Strengthen confidence in both library correctness and workflow behavior by explicitly separating:
  - pure library tests (contracts/invariants/error handling)
  - workflow/system tests (end-to-end retrieval pipeline behavior)
- Add regression coverage that detects silent retrieval drift and schema drift on deterministic tiny fixtures.

## Scope
- In:
  - Test strategy documentation and marker taxonomy
  - New deterministic tiny retrieval fixtures
  - Regression tests for retrieval ranking/stability contracts
  - Workflow tests covering mini end-to-end retrieval path and artifact integrity
  - Pytest marker registration for new layers
- Out:
  - Changes to production retrieval/training algorithms
  - Large benchmark/evaluation datasets
  - CI workflow redesign beyond enabling the new test slices

## Files To Inspect
- `pyproject.toml`
- `tests/tests.md`
- `tests/conftest.py`
- `tests/search/test_search.py`
- `tests/evaluation/test_evaluate_script_smoke.py`
- `mess/search/search.py`
- `mess/search/faiss_index.py`
- `scripts/build_clip_index.py`
- `scripts/evaluate_retrieval.py`

## Contracts To Preserve
- Data shape contracts:
  - `raw`: `[num_segments, 13, time_steps, 768]`
  - `segments`: `[num_segments, 13, 768]`
  - `aggregated`: `[13, 768]`
- Search contract:
  - `load_features()` returns `(features, track_names)` where `features` is `(n_tracks, feature_dim)`.
- Artifact integrity contract:
  - immutable `artifact_version_id`, strict checksum validation, and schema validation on load.
- Retrieval contract:
  - normalized cosine behavior and score-sorted output ordering.

## Plan
1. Update this task record before editing.
2. Introduce explicit test-layer markers and update testing docs to reflect the two-layer strategy.
3. Add deterministic fixture-driven regression tests for ranking identities, score sanity, and reproducibility tolerance.
4. Add workflow/system tests that run a compact end-to-end retrieval pipeline and assert artifact/schema integrity.
5. Run focused lint/tests for changed files and report residual risk.

## Validation
- `uv run ruff check tests/conftest.py tests/test_setup_demo_data_script.py tests/evaluation/test_evaluate_script_smoke.py tests/training/test_end_to_end_projection_artifact.py tests/workflows/test_retrieval_pipeline_workflow.py tests/search/test_retrieval_regression.py tests/evaluation/test_retrieval_mini_benchmark.py`
- `uv run pytest -q tests/search/test_retrieval_regression.py`
- `uv run pytest -q tests/workflows/test_retrieval_pipeline_workflow.py`
- `uv run pytest -q tests/evaluation/test_retrieval_mini_benchmark.py`
- `uv run pytest -q tests/workflows/test_retrieval_pipeline_workflow.py tests/evaluation/test_evaluate_script_smoke.py tests/test_setup_demo_data_script.py tests/training/test_end_to_end_projection_artifact.py tests/search/test_retrieval_regression.py tests/evaluation/test_retrieval_mini_benchmark.py`
- `uv run pytest -q tests/search/test_search.py -m library`
- `uv run pytest -q tests/workflows/test_retrieval_pipeline_workflow.py -m system`

## Risks / Open Questions
- Regression expectations must be deterministic but not overfit to exact floating-point outputs; tolerances must be calibrated.
- Tiny benchmark checks improve drift detection but are not a substitute for full research evaluation on real datasets.
