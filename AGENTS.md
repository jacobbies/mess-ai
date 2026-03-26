# MESS-AI AGENT EXECUTION CONTRACT

This file defines repo-local execution policy. Use it for workflow rules and hard constraints; code and tests remain the source of truth.

## 1) Precedence

- Source of truth: `mess/` and `tests/`.
- When docs disagree with code/tests, code/tests win.
- This file defines execution policy; ignore `CLAUDE.md`.
- Load `docs/` only when the task needs extra context.

## 2) Working Defaults

- Default to `main`; create a branch only when the user asks, when work is stacked on unmerged changes, or when isolation materially reduces risk.
- Keep scope narrow; avoid unrelated edits and opportunistic refactors.
- Preserve existing behavior unless the task explicitly requires change.
- If an interface changes, update callers, tests, and docs together.
- If work depends on another unmerged change, state that explicitly and treat it as stacked work.
- For non-trivial PR/debug work, update `docs/pr_backlog.md` before editing with objective, scope, files, contracts, plan, validation, and open risks.

## 3) Core Objectives

- Keep `mess/` modular, test-backed, and reproducible.
- Preserve data, shape, path, and API contracts.
- Prefer contract stability and reproducibility over convenience refactors.
- Support both local research workflows and remote artifact serving.

## 4) Hard Contracts (Do Not Regress)

### 4.1 Data Layout And Shapes

- Default data root: `mess_config.data_root -> <project>/data`
- Embedding feature shapes:
  - `raw`: `[num_segments, 13, time_steps, 768]`
  - `segments`: `[num_segments, 13, 768]`
  - `aggregated`: `[13, 768]`
- Search contract: `load_features()` returns `(features, track_names)` and `features` is `(n_tracks, feature_dim)` for FAISS.

### 4.2 Clip Index And Leakage Safety

- Clip index rows must preserve required columns and stable alignment.
- Split assignment for train/val/test must be by `recording_id`, never by clip.
- Retrieval training must preserve self-exclusion, near-time exclusion, and recording-level positive/negative policies.

### 4.3 Proxy Targets / Discovery

- Proxy target NPZ files must keep nested category payloads.
- Discovery must read targets through nested category/field access.
- Missing required scalar targets must trigger omission or warnings.
- Optional categories (for example MIDI expression) may use NaN and must be filtered before probing.

### 4.4 Artifact Integrity (Local + S3)

- Artifact integrity is mandatory: immutable `artifact_version_id`, checksum validation on load/download, and `latest.json` updates only after upload validation succeeds.
- Serving must support `load_latest_from_s3` and `download_artifact_from_s3`.
- Publishing must support `upload_artifact_to_s3`.

## 5) Module Guardrails

- Extraction: check config impact in `mess/config.py`; preserve shape/path behavior in `mess/extraction/storage.py`; preserve throughput and memory behavior in `mess/extraction/pipeline.py`.
- Probing/aspects: update `mess/probing/targets.py` when adding targets; keep `SCALAR_TARGETS` and optional-category handling consistent in `mess/probing/discovery.py`; ensure `ASPECT_REGISTRY` references only existing scalar targets; rerun probing when aspect mapping changes.
- Search: preserve normalized cosine behavior (`IndexFlatIP` + L2 normalization); keep query existence checks and result ordering stable; validate layer-specific and aspect-weighted paths.
- Training/projection: keep recording-level split boundaries leakage-safe; preserve mining guardrails and index refresh determinism; keep projection-head-first as default; only consider MERT unfreezing after head-only plateaus and metric justification.

## 6) Execution Workflow

For code changes:
1. Read the task and inspect affected code/tests first.
2. Implement the minimal coherent change.
3. Add or update tests for behavior changes.
4. Run quality checks for the affected scope.
5. Report what changed, how it was validated, and any residual risks.

Recommended checks:

```bash
uv run ruff check .
uv run pytest -v
uv run mypy mess
```

Scope checks to touched modules when appropriate.

## 7) Testing Conventions

- Do not load the real MERT model in unit tests.
- Mock heavy model or dependency calls.
- Use `tmp_path` for filesystem tests.
- Mirror tests by domain:
  - `mess/extraction` -> `tests/extraction`
  - `mess/probing` -> `tests/probing`
  - `mess/search` -> `tests/search`
  - `mess/training` -> `tests/training`

## 8) Script Status Policy

Stable scripts:
- `scripts/build_clip_index.py`
- `scripts/extract_features.py`
- `scripts/run_probing.py`
- `scripts/demo_recommendations.py`
- `scripts/train_retrieval_ssl.py`

Treat scripts listed in `scripts/_NEEDS_UPDATE.txt` as outdated or experimental unless the task explicitly updates them.

## 9) Git, Data, And Context Hygiene

- Do not commit large generated artifacts from `data/`, `mlruns/`, or generated embeddings.
- Keep one logical change per commit and use short imperative commit messages.
- Default load order: this file, then relevant code/tests, then only the `docs/*.md` files the task needs.
- Use `docs/context_index.md` as the pointer router.
- Do not bulk-load `docs/`.
- If a pointer doc conflicts with this contract or code/tests, this contract and code/tests win.
