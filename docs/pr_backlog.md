# MESS-AI PR Backlog (Non-Stacked)

Last updated: 2026-03-05

This file is the persistent execution ledger for proposed PR work.  
Goal: each PR is independently reviewable, opened from `main`, and not stacked on another open PR branch.

## Current Contract Snapshot

1. Clip index contract is now strict:
`build_clip_records()` requires segment embeddings shaped like `[segments, 13, embedding_dim]` and fails fast otherwise (`mess/datasets/clip_index.py`).
2. Mapping manifest parsing is hardened:
`_load_id_maps()` strips whitespace and ignores blank IDs (`scripts/build_clip_index.py`, `tests/datasets/test_build_clip_index_script.py`).
3. Artifact deployment contract is explicit:
FAISS artifacts require immutable `artifact_version_id`, checksum validation, and pointer-based latest resolution (`mess/search/faiss_index.py`, README production section).

These are baseline assumptions for all PRs below.

## Non-Stacked Execution Rules

1. Start every PR branch from current `main`.
2. Do not open a PR that depends on commits from another unmerged PR.
3. If a planned PR touches files already modified by an open PR, wait until merge, then rebase from latest `main`.
4. Keep one logical change per PR; move any spillover into the next queued PR.
5. After PR submission, mark it complete in this file and remove detail sections if desired.

## Queue

- [ ] PR-01: Fix extras + CI + install/error-message consistency (Small-Medium)
- [ ] PR-02: Add IVF `nprobe` tuning in artifacts + runtime search (Medium)
- [ ] PR-03: Expand index types with FAISS `index_factory` (Large)
- [ ] PR-04: Export projection retriever artifact from training outputs (Large)
- [ ] PR-05: Add retrieval evaluation harness + JSON reports (Medium)
- [ ] PR-06: Add hybrid semantic + keyword metadata search (Medium-Large)
- [ ] PR-07: Unify probing audio decode with extraction decode utilities (Medium)
- [ ] PR-08: Resolve/retire scripts listed in `_NEEDS_UPDATE` (Small-Medium)

## PR-01: Fix Extras + CI + Install/Error Consistency

Status: Planned  
Branch: `pr01-packaging-extras-consistency`  
Primary objective: remove install-path drift between CI, pyproject extras, and runtime guidance.

### Current Evidence

1. CI installs `--extra probing`, but `pyproject.toml` currently defines only `search`.
2. `mess/probing/discovery.py` error hint says `mess-ai[probing]`, which is currently invalid.
3. README install guidance omits a probing extra path.

### Implementation Plan

1. Add `probing` optional dependency group to `pyproject.toml` (or remove all probing extra references; choose one path and make all files consistent).
2. Align `.github/workflows/ci.yml` sync command with actual extras.
3. Update probing error message in `mess/probing/discovery.py` to match chosen install path.
4. Update README install section with canonical commands.
5. Add contract tests:
`tests/test_packaging_contracts.py` to verify CI `--extra X` values exist in pyproject optional dependencies and runtime hints are accurate.

### Acceptance Criteria

1. `uv sync --group dev --extra search --extra probing` either works or is replaced everywhere with valid equivalent.
2. Runtime probing dependency error points to a real install path.
3. CI extra references and docs remain in lockstep via tests.

### Out of Scope

1. Broad dependency pruning.
2. Changing dev-group package choices unrelated to extras drift.

## PR-02: Add IVF `nprobe` Tuning (Artifacts + Runtime)

Status: Planned  
Branch: `pr02-faiss-nprobe-config`  
Primary objective: expose IVF recall/latency knob without changing default flat-index behavior.

### Current Evidence

1. `mess/search/faiss_index.py` supports `flatip` and `ivfflat`, with `nlist` only.
2. Manifest schema has no field for default IVF search params.
3. `scripts/publish_faiss_index.py` exposes `--nlist` but not `--nprobe`.

### Implementation Plan

1. Extend artifact manifest model to include optional `default_nprobe`.
2. Apply `default_nprobe` when loading or searching IVF indexes.
3. Add `nprobe` override path for runtime search calls on `FAISSArtifact.search(...)`.
4. Add CLI option `--nprobe` in `scripts/publish_faiss_index.py` and plumb into artifact builders.
5. Keep backward compatibility for existing artifacts that do not include `default_nprobe`.

### Tests

1. Extend `tests/search/test_faiss_index.py`:
assert `default_nprobe` persists in manifest for IVF artifacts and affects loaded index behavior.
2. Add search-time override test ensuring explicit `nprobe` takes effect for IVF.

### Acceptance Criteria

1. IVF artifacts can encode sane default `nprobe`.
2. Runtime can override per-query/per-call without rebuilding artifacts.
3. Existing flat and legacy artifacts continue to load.

### Out of Scope

1. New FAISS index families (`HNSW`, `PQ`, `OPQ`) in this PR.

## PR-03: Expand Index Types with `index_factory`

Status: Planned  
Branch: `pr03-faiss-index-factory`  
Primary objective: support additional ANN configurations (`HNSW`, `IVFPQ`, `OPQ+IVF`) through FAISS-native factory strings.

### Current Evidence

1. `IndexType` is limited to `"flatip" | "ivfflat"` in `mess/search/faiss_index.py`.
2. No factory-string path exists in build/publish flow.

### Implementation Plan

1. Add index creation mode supporting FAISS `index_factory` strings.
2. Persist factory metadata in artifact manifest (`factory_string`, training metadata if needed).
3. Update `scripts/publish_faiss_index.py` with CLI support for factory mode.
4. Keep existing index_type paths stable (`flatip`, `ivfflat`) for backward compatibility.
5. Validate train-required factory indexes before add/search.

### Tests

1. New `tests/search/test_index_factory_build.py` with synthetic vectors:
build, train (if required), add, save/load, and search.
2. Include representative factory examples that are dimension-compatible in tests.

### Acceptance Criteria

1. Factory-based index build path works for at least one HNSW and one IVF/PQ-style config.
2. Manifest fully describes how index was built.
3. Existing artifact loading and search contracts stay intact.

### Out of Scope

1. Automatic hyperparameter tuning.
2. Benchmark suite for all factory variants.

## PR-04: Export Projection Retriever Artifact from Training

Status: Planned  
Branch: `pr04-training-artifact-export`  
Primary objective: connect retrieval training output to deployable FAISS artifact packaging.

### Current Evidence

1. `scripts/train_retrieval_ssl.py` outputs checkpoint + metrics but not a deployable artifact.
2. FAISS artifact tooling exists (`mess/search/faiss_index.py`) but currently builds from feature directories, not projection head checkpoints.

### Implementation Plan

1. Add training export helper module (`mess/training/export.py`) to:
load clip vectors, apply trained projection head, produce projected embeddings aligned to clip metadata.
2. Extend `scripts/train_retrieval_ssl.py` with:
`--export-artifact`, `--artifact-root`, `--artifact-name`, and artifact build settings.
3. Reuse existing artifact persistence/checksum/versioning contract.
4. Optionally add S3 upload flags to publish directly after export.
5. Keep export optional so current training workflow remains valid.

### Tests

1. New `tests/training/test_end_to_end_projection_artifact.py`:
small synthetic dataset, short training, export artifact, load artifact, smoke-search.
2. Validate manifest fields and checksum files exist.

### Acceptance Criteria

1. One command can produce checkpoint + deployable FAISS artifact.
2. Exported artifact is loadable with existing `load_artifact`/`load_latest_from_s3` contract.
3. Training script remains backward compatible when export flags are absent.

### Out of Scope

1. Full orchestration pipeline that automatically triggers evaluation.

## PR-05: Retrieval Evaluation Harness + JSON Reports

Status: Planned  
Branch: `pr05-retrieval-evaluation-harness`  
Primary objective: make cosine baseline vs learned retriever comparisons reproducible and reportable.

### Current Evidence

1. `scripts/evaluate_clip_retrieval.py` provides sanity-style recall checks, not full standardized reports.
2. No dedicated `mess/evaluation` module currently exists.

### Implementation Plan

1. Add `mess/evaluation/metrics.py` with ranking metrics:
Recall@K, MRR, nDCG.
2. Add `scripts/evaluate_retrieval.py` that:
loads clip index + vectors, evaluates baseline and optionally projection-head output, writes report JSON.
3. Include both clip->clip and clip->track protocol modes.
4. Optional MLflow logging only as additive behavior.

### Tests

1. `tests/evaluation/test_metrics.py` with deterministic known rankings.
2. `tests/evaluation/test_evaluate_script_smoke.py` on tiny synthetic index and vectors.

### Acceptance Criteria

1. Script emits deterministic JSON report with metric blocks and run config.
2. Baseline and learned retriever can be compared in one output artifact.
3. Metrics module is unit-tested independently from scripts.

### Out of Scope

1. Human-eval annotation tooling.
2. Dashboard/UI work.

## PR-06: Hybrid Semantic + Keyword Metadata Search

Status: Planned  
Branch: `pr06-hybrid-search`  
Primary objective: add metadata-aware retrieval controls without replacing embedding similarity.

### Current Evidence

1. `mess/datasets/metadata_table.py` already provides canonical row loading and lookup helpers.
2. Search stack currently ranks purely by vector similarity.
3. Demo script has no metadata filter/keyword path.

### Implementation Plan

1. Add `mess/search/hybrid.py`:
combine vector score with lightweight token-match score and optional hard filters.
2. Extend metadata helpers minimally in `mess/datasets/metadata_table.py` for text-field access.
3. Update `scripts/demo_recommendations.py` with optional:
`--keyword`, `--filter key=value`, and weighting controls.
4. Keep existing search APIs unchanged; hybrid path is additive.

### Tests

1. New `tests/search/test_hybrid_search.py` using tiny synthetic embeddings + metadata table.
2. Validate filtering and score-boost behavior are deterministic.

### Acceptance Criteria

1. Users can constrain/boost by metadata fields while preserving semantic ranking.
2. No regression in existing `mess/search/search.py` behavior.

### Out of Scope

1. Full BM25 dependency integration unless justified by measured gain.

## PR-07: Unify Probing Decode Path with Extraction Utilities

Status: Planned  
Branch: `pr07-probing-audio-decode-unification`  
Primary objective: ensure probing target generation uses the same decode/segmentation contracts as extraction.

### Current Evidence

1. `mess/probing/segment_targets.py` currently uses `torchaudio.load` + local resample path.
2. `mess/probing/targets.py` currently decodes independently via torchaudio.
3. `mess/extraction/audio.py` already provides canonical `load_audio()` and `load_audio_segments()`.

### Implementation Plan

1. In `mess/probing/segment_targets.py`, replace direct torchaudio decode with `load_audio_segments()`.
2. In `mess/probing/targets.py`, replace direct torchaudio decode with `load_audio()`.
3. Ensure segment boundaries remain identical to extraction contract.
4. Preserve fallback behavior when TorchCodec is unavailable.

### Tests

1. Extend `tests/probing/test_segment_targets.py` with boundary-alignment assertions against extraction utilities.
2. Add/extend fallback-path tests (TorchCodec present vs absent simulation).

### Acceptance Criteria

1. Probing and extraction use one decode contract.
2. Segment-level target counts/timestamps remain stable.
3. Tests cover both primary and fallback decode paths.

### Out of Scope

1. Redesign of target formulas.

## PR-08: Resolve or Retire `_NEEDS_UPDATE` Scripts

Status: Planned  
Branch: `pr08-needs-update-script-resolution`  
Primary objective: remove confusion around unsupported scripts and align docs with maintained entry points.

### Current Evidence

1. `scripts/_NEEDS_UPDATE.txt` lists four outdated scripts.
2. README already marks them experimental, but scripts remain runnable and drift-prone.

### Implementation Plan

1. Choose one policy and apply consistently:
either retire (delete + doc cleanup) or refactor each script to current API.
2. Update README + `_NEEDS_UPDATE.txt` to reflect final state.
3. If retiring, replace with explicit guidance pointing to maintained scripts.

### Tests

1. If scripts are refactored: add focused smoke tests for each retained script.
2. If retired: add a small docs/status consistency test to avoid stale references.

### Acceptance Criteria

1. No ambiguous or misleading script status in repo.
2. Maintained scripts list is accurate in docs.

### Out of Scope

1. Adding new research scripts beyond what is needed to replace retired behavior.

## Completion Log

Use this section when a PR is submitted/merged:

- `PR-ID`:
- GitHub PR:
- Branch:
- Commit SHA:
- Date:
- Notes:

