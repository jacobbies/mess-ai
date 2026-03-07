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

- [x] PR-01: Fix extras + CI + install/error-message consistency (Small-Medium, PR #11)
- [x] PR-02: Add IVF `nprobe` tuning in artifacts + runtime search (Medium, PR #12)
- [x] PR-03: Expand index types with FAISS `index_factory` (Large, PR #13)
- [x] PR-04: Export projection retriever artifact from training outputs (Large, PR #14)
- [x] PR-05: Add retrieval evaluation harness + JSON reports (Medium, PR #15)
- [x] PR-06: Add hybrid semantic + keyword metadata search (Medium-Large, PR #16)
- [x] PR-07: Unify probing audio decode with extraction decode utilities (Medium, PR #17)
- [x] PR-08: Resolve/retire scripts listed in `_NEEDS_UPDATE` (Small-Medium, PR #18)

## PR-01: Fix Extras + CI + Install/Error Consistency

Status: Completed (submitted)  
Branch: `pr01-packaging-extras-consistency`  
GitHub PR: https://github.com/jacobbies/mess-ai/pull/11

Shipped scope:

1. Added `probing` optional extra in `pyproject.toml` so CI `--extra probing` is valid.
2. Updated runtime install docs in README for probing.
3. Added `tests/test_packaging_contracts.py` to keep CI extras and probing install hints aligned.
4. Added and initialized this persistent backlog ledger (`docs/pr_backlog.md`).

## PR-02: Add IVF `nprobe` Tuning (Artifacts + Runtime)

Status: Completed (submitted)  
Branch: `pr02-faiss-nprobe-config`  
GitHub PR: https://github.com/jacobbies/mess-ai/pull/12

Shipped scope:

1. Added optional `default_nprobe` to FAISS artifact manifests.
2. Applied manifest-level `default_nprobe` during artifact load for IVF indexes.
3. Added runtime `nprobe` override support in `FAISSArtifact.search(...)`.
4. Added `--nprobe` CLI support to `scripts/publish_faiss_index.py`.
5. Extended FAISS artifact integration tests for nprobe persistence + override behavior.

## PR-03: Expand Index Types with `index_factory`

Status: Completed (submitted)  
Branch: `pr03-faiss-index-factory`  
GitHub PR: https://github.com/jacobbies/mess-ai/pull/13

Shipped scope:

1. Added `index_type="factory"` support in artifact builders.
2. Added manifest metadata for `factory_string` plus training metadata fields.
3. Updated publish CLI to support `--index-type=factory` with `--factory-string`.
4. Added integration tests covering HNSW, IVFPQ, and OPQ+IVF factory builds/searches.

## PR-04: Export Projection Retriever Artifact from Training

Status: Completed (submitted)  
Branch: `pr04-training-artifact-export`  
GitHub PR: https://github.com/jacobbies/mess-ai/pull/14  
Primary objective: connect retrieval training output to deployable FAISS artifact packaging.

Shipped scope:

1. Added `mess/training/export.py` to project clip vectors with trained projection-head weights and emit clip artifact payloads.
2. Extended `scripts/train_retrieval_ssl.py` with optional `--export-artifact` flow, configurable artifact build options, and optional S3 upload.
3. Added in-memory clip artifact builder (`build_clip_artifact_from_vectors`) to reuse artifact persistence contracts without intermediate files.
4. Added integration coverage for training-to-artifact export plus training package export surface updates.

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

Status: Completed (submitted)  
Branch: `pr05-retrieval-evaluation-harness`  
GitHub PR: https://github.com/jacobbies/mess-ai/pull/15  
Primary objective: make cosine baseline vs learned retriever comparisons reproducible and reportable.

Shipped scope:

1. Added `mess/evaluation/metrics.py` with deterministic Recall@K, MRR, and nDCG aggregation helpers.
2. Added `scripts/evaluate_retrieval.py` to evaluate baseline vectors and optional projection-checkpoint vectors in one JSON report.
3. Added support for both `clip_to_clip` and `clip_to_track` protocol blocks.
4. Added unit tests for metric calculations and a script smoke test over synthetic clip-index data.

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

Status: Completed (submitted)  
Branch: `pr06-hybrid-search`  
GitHub PR: https://github.com/jacobbies/mess-ai/pull/16  
Primary objective: add metadata-aware retrieval controls without replacing embedding similarity.

Shipped scope:

1. Added `mess/search/hybrid.py` with semantic cosine scoring fused with keyword matching and optional hard metadata filters.
2. Added `DatasetMetadataTable.text_for_track(...)` for lightweight metadata text extraction.
3. Extended `scripts/demo_recommendations.py` with `--keyword`, repeatable `--filter key=value`, and semantic/keyword weighting controls.
4. Added deterministic hybrid search tests plus metadata-table text helper coverage.

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

Status: Completed (submitted)  
Branch: `pr07-probing-audio-decode-unification`  
GitHub PR: https://github.com/jacobbies/mess-ai/pull/17  
Primary objective: ensure probing target generation uses the same decode/segmentation contracts as extraction.

Shipped scope:

1. Replaced direct torchaudio decode in `mess/probing/targets.py` with canonical `mess.extraction.audio.load_audio(...)`.
2. Replaced manual decode + segmentation in `mess/probing/segment_targets.py` with `mess.extraction.audio.load_audio_segments(...)`.
3. Added delegation tests to verify probing paths use extraction audio utilities.
4. Preserved segment-boundary behavior and fallback coverage through existing extraction-audio tests.

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

Status: Completed (submitted)  
Branch: `pr08-needs-update-script-resolution`  
GitHub PR: https://github.com/jacobbies/mess-ai/pull/18  
Primary objective: remove confusion around unsupported scripts and align docs with maintained entry points.

Shipped scope:

1. Retired and removed four outdated scripts that depended on removed search/index APIs.
2. Updated `README.md` script-status section to clearly mark retired scripts as removed.
3. Updated `scripts/_NEEDS_UPDATE.txt` to resolved status with replacement guidance.
4. Added `tests/test_script_status_contract.py` to keep maintained/retired script status aligned with the repository state.

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

- `PR-ID`: PR-01
- GitHub PR: https://github.com/jacobbies/mess-ai/pull/11
- Branch: `pr01-packaging-extras-consistency`
- Commit SHA: `d2ca697`
- Date: 2026-03-05
- Notes: CI extras/install-path contract aligned; backlog ledger initialized.

- `PR-ID`: PR-02
- GitHub PR: https://github.com/jacobbies/mess-ai/pull/12
- Branch: `pr02-faiss-nprobe-config`
- Commit SHA: `9c5e62b`
- Date: 2026-03-05
- Notes: Added IVF nprobe defaults/overrides in artifacts + publish CLI support.

- `PR-ID`: PR-03
- GitHub PR: https://github.com/jacobbies/mess-ai/pull/13
- Branch: `pr03-faiss-index-factory`
- Commit SHA: `bce7bc0`
- Date: 2026-03-05
- Notes: Added index_factory mode, factory metadata, and HNSW/IVFPQ/OPQ+IVF tests.

- `PR-ID`: PR-04
- GitHub PR: https://github.com/jacobbies/mess-ai/pull/14
- Branch: `pr04-training-artifact-export`
- Commit SHA: `e7794a4`
- Date: 2026-03-05
- Notes: Added projection-head artifact export path in training script with optional S3 publish and end-to-end export tests.

- `PR-ID`: PR-05
- GitHub PR: https://github.com/jacobbies/mess-ai/pull/15
- Branch: `pr05-retrieval-evaluation-harness`
- Commit SHA: `87446d9`
- Date: 2026-03-05
- Notes: Added reusable retrieval metric module plus JSON evaluation script covering clip-to-clip and clip-to-track protocols.

- `PR-ID`: PR-06
- GitHub PR: https://github.com/jacobbies/mess-ai/pull/16
- Branch: `pr06-hybrid-search`
- Commit SHA: `bd544f7`
- Date: 2026-03-05
- Notes: Added additive hybrid semantic+metadata retrieval path and demo script controls for keyword boosting and hard filters.

- `PR-ID`: PR-07
- GitHub PR: https://github.com/jacobbies/mess-ai/pull/17
- Branch: `pr07-probing-audio-decode-unification`
- Commit SHA: `b30e149`
- Date: 2026-03-05
- Notes: Unified probing decode with extraction audio helpers and added delegation tests for both track-level and segment-level target generation.

- `PR-ID`: PR-08
- GitHub PR: https://github.com/jacobbies/mess-ai/pull/18
- Branch: `pr08-needs-update-script-resolution`
- Commit SHA: `a1b2687`
- Date: 2026-03-05
- Notes: Retired obsolete scripts, updated docs/script-status ledger, and added a status consistency test.
