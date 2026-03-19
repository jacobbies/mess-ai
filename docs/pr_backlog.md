# PR/TASK HANDOFF
- Task/PR: Full merge-readiness cleanup for contextualizer + repo baseline
- Branch: `chore/pr-backlog-template-reset`
- Status: `completed`

## Objective
- Clean up the full current worktree so it can merge to `main` without avoidable lint, typing, test, documentation, or local validation blockers.

## Scope
- In:
  - Contextualizer training/export/reranker additions and their tests
  - Training package export surface changes
  - README/demo-data/script-status onboarding changes
  - CI and lint coverage updates needed to enforce the new files
  - Repo-baseline fixes required for `uv run pytest -q` and `uv run mypy mess`
- Out:
  - Broad refactors unrelated to the current changed files
  - Contract changes outside what is already introduced in this branch

## Files To Inspect
- `.github/workflows/ci.yml`
- `.gitignore`
- `README.md`
- `data/README.md`
- `mess/search/reranker.py`
- `mess/training/__init__.py`
- `mess/training/context_config.py`
- `mess/training/context_export.py`
- `mess/training/context_trainer.py`
- `mess/training/contextualizer.py`
- `pyproject.toml`
- `scripts/_NEEDS_UPDATE.txt`
- `scripts/contextualize_embeddings.py`
- `scripts/evaluate_clip_retrieval.py`
- `scripts/inspect_embeddings.py`
- `scripts/script_status.json`
- `scripts/setup_demo_data.py`
- `scripts/train_contextualizer.py`
- `tests/search/test_reranker.py`
- `tests/test_script_status_contract.py`
- `tests/test_setup_demo_data_script.py`
- `tests/training/test_context_trainer_smoke.py`
- `tests/training/test_contextualizer.py`
- `tests/training/test_training_init_exports.py`
- `mess/config.py`
- `mess/extraction/audio.py`
- `mess/extraction/extractor.py`
- `mess/probing/discovery.py`
- `mess/probing/midi_targets.py`
- `mess/probing/segment_targets.py`
- `mess/probing/targets.py`
- `mess/search/faiss_index.py`
- `mess/search/hybrid.py`
- `mess/search/search.py`
- `tests/search/test_faiss_index.py`

## Contracts To Preserve
- Embedding/search contracts:
  - `load_features()` shape/API behavior stays intact
  - FAISS retrieval remains normalized cosine via `IndexFlatIP` + L2 normalization
  - Query self-exclusion and deterministic result ordering stay intact
- Training/data safety:
  - Clip-index split handling remains recording-level leakage-safe
  - Contextualizer training/export keeps checkpoint -> inference -> artifact flow coherent
- Repo hygiene:
  - Demo data remains synthetic and git-safe
  - Script lifecycle status has one source of truth and matches repository contents

## Plan
1. Reproduce the remaining full-suite and full-type-check blockers against the current branch.
2. Apply minimal fixes for the FAISS full-suite crash path without changing artifact/search contracts.
3. Resolve repo-wide `mypy` errors with narrow typing/import fixes rather than refactors.
4. Re-run full validation plus the changed-surface checks to confirm merge readiness.
5. Report final readiness, validation coverage, and any residual risks that still require follow-up.

## Validation
- Lint:
  - `uv run ruff check .`
- Types:
  - `uv run mypy --follow-imports skip mess/search/faiss_index.py mess/search/__init__.py mess/datasets/clip_index.py mess/training/config.py mess/training/index.py`
  - `uv run mypy --follow-imports skip mess/search/reranker.py mess/training/context_config.py mess/training/context_export.py mess/training/context_trainer.py mess/training/contextualizer.py`
  - `uv run mypy mess`
- Tests:
  - `uv run pytest -q tests/search/test_reranker.py tests/test_script_status_contract.py tests/test_setup_demo_data_script.py tests/training/test_context_trainer_smoke.py tests/training/test_contextualizer.py tests/training/test_training_init_exports.py`
  - `uv run pytest -q`

## Validation Results
- `uv run ruff check .`: pass
- `uv run mypy mess`: pass
- `uv run pytest -q`: pass
- Focused contextualizer/search/script-status tests: pass

## Risks / Open Questions
- Full-suite FAISS behavior on local macOS may still be sensitive to small-sample IVF training and native-library stability.
- Some `mypy` failures are in older probing/extraction modules, so the cleanup must stay narrow and avoid behavioral changes while tightening types.
