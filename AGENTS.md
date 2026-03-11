# MESS-AI AGENT EXECUTION CONTRACT

This file is the binding execution contract for coding agents in this repo.
Use it for operational behavior and hard constraints only.

## 1) Precedence And Scope

1. Source of truth is implementation and tests:
   - `mess/`
   - `tests/`
2. When docs disagree with code/tests, code/tests win.
3. This file defines execution policy.
4. Narrative context is pointer-loaded from `docs/` only when needed.
5. Ignore `CLAUDE.md`.

## 2) Core Objectives

- Keep `mess/` modular, test-backed, and reproducible.
- Preserve data/shape/path/API contracts.
- Make minimal coherent changes with explicit validation.

## 2.1 Project Intent And Stack Rationale

- MESS-AI is research-first infrastructure with production artifact compatibility as a hard requirement.
- The architecture split is intentional:
  - `datasets/` for stable metadata and clip contracts
  - `extraction/` for model inference + persistence/orchestration separation
  - `probing/` for measurable layer semantics
  - `search/` for retrieval primitives and deployable FAISS artifacts
  - `training/` for embedding-geometry iteration with retrieval compatibility
- Historical operating constraint: local research workflows and remote artifact serving must both work.
- Change policy implication: prefer contract stability and reproducibility over convenience refactors.

## 3) Hard Contracts (Do Not Regress)

### 3.1 Data Layout And Shapes

- Default data root: `mess_config.data_root -> <project>/data`
- Embedding feature contracts:
  - `raw`: `[num_segments, 13, time_steps, 768]`
  - `segments`: `[num_segments, 13, 768]`
  - `aggregated`: `[13, 768]`
- Search contract:
  - `load_features()` returns `(features, track_names)`
  - `features` is `(n_tracks, feature_dim)` for FAISS

### 3.2 Clip Index And Leakage Safety

- Clip index rows must preserve required columns and stable alignment.
- Split assignment for train/val/test must be by `recording_id`, never by clip.
- Retrieval training must keep leakage guards:
  - self exclusion
  - near-time exclusion
  - recording-level positive/negative policies

### 3.3 Proxy Targets / Discovery

- Proxy target NPZ contract: nested category payloads.
- Discovery reads targets via nested category/field access.
- Missing required scalar targets cause omission/warnings.
- Optional categories (for example MIDI expression) may use NaN and be filtered pre-probe.

### 3.4 Artifact Integrity (Local + S3)

- Artifact integrity is mandatory:
  - immutable `artifact_version_id`
  - checksum validation on load/download
  - pointer `latest.json` updated only after upload validation succeeds
- Serving path must support:
  - `load_latest_from_s3`
  - `download_artifact_from_s3`
- Publishing path must support:
  - `upload_artifact_to_s3`

## 4) High-Level Module Guardrails

### 4.1 Extraction Changes

1. Check config impact in `mess/config.py`.
2. Preserve extraction shape/path behavior in `mess/extraction/storage.py`.
3. Preserve throughput and memory behavior in `mess/extraction/pipeline.py`.

### 4.2 Probing / Aspect Changes

1. Update target generation in `mess/probing/targets.py` when adding targets.
2. Keep `SCALAR_TARGETS` and optional-category handling consistent in `mess/probing/discovery.py`.
3. Ensure `ASPECT_REGISTRY` only references existing scalar target names.
4. Re-run probing workflow when aspect mapping behavior changes.

### 4.3 Search Changes

1. Preserve normalized cosine behavior (`IndexFlatIP` + L2 normalization).
2. Keep query existence checks and result ordering stable.
3. Validate layer-specific and aspect-weighted paths.

### 4.4 Training / Projection Changes

1. Keep clip split boundaries leakage-safe (recording-level).
2. Preserve mining guardrails and index refresh determinism.
3. Keep projection-head-first workflow as default.
4. Only consider MERT unfreezing after head-only plateaus and metrics justify it.

## 5) Execution Workflow

For code changes:
1. Read task and inspect affected code/tests first.
2. Implement minimal coherent change.
3. Add/update tests for behavior changes.
4. Run quality checks for affected scope.
5. Summarize changes, validations, and residual risks.

Recommended checks:

```bash
uv run ruff check .
uv run pytest -v
uv run mypy mess
```

Scope checks to touched modules when appropriate.

## 6) Testing Conventions

- Do not load real MERT model in unit tests.
- Mock heavy model/dependency calls.
- Use `tmp_path` for filesystem tests.
- Keep tests mirrored by domain:
  - `mess/extraction` -> `tests/extraction`
  - `mess/probing` -> `tests/probing`
  - `mess/search` -> `tests/search`
  - `mess/training` -> `tests/training`

## 7) Script Status Policy

Stable scripts:
- `scripts/build_clip_index.py`
- `scripts/extract_features.py`
- `scripts/run_probing.py`
- `scripts/demo_recommendations.py`
- `scripts/train_retrieval_ssl.py`

Outdated/experimental scripts are listed in `scripts/_NEEDS_UPDATE.txt`.
Do not treat those as production behavior unless explicitly refactoring them.

## 8) Git And Data Hygiene

- Do not commit large generated artifacts from `data/`, `mlruns/`, or generated embeddings.
- Keep one logical change per commit.
- Use short imperative commit messages.

## 9) PR Backlog Handoff (Required For Non-Trivial PR/Debug Work)

Use `docs/pr_backlog.md` as the branch-local handoff file for non-trivial PR/debug tasks.
Before implementation, update it with objective, scope, files to inspect, contracts, plan, validation, and open risks.

## 10) Smart Context Pointer Method (Markdown)

Default load behavior:
1. Load this file.
2. Load code/tests required by task.
3. Only then load additional `docs/*.md` via pointer routing.

Pointer routing entrypoint:
- `docs/context_index.md`

Rules:
- Do not bulk-load `docs/`.
- Load only files whose trigger matches the current task.
- If a pointer doc conflicts with this contract or code/tests, this contract and code/tests win.
