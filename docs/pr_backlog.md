# PR/TASK HANDOFF
- Task/PR: Unify clip retrieval around ClipIndex-backed FAISS artifacts
- Branch: `pr15-clip-search-artifact`
- Status: `in_progress`

## Objective
- Make clip retrieval use one contract end-to-end: clip artifacts built from ClipIndex metadata and clip search executed against prebuilt FAISS artifacts.
- Preserve clip identity and provenance in artifacts so clip search can return full metadata, not just reduced location fields.

## Scope
- In:
  - Clip artifact metadata schema and persistence
  - Projection export path for clip artifacts
  - Clip artifact build path used by publishing/search workflows
  - One canonical artifact-backed clip search API
  - Tests and repo callers directly affected by the contract change
- Out:
  - Track retrieval API changes
  - Unrelated probing/search refactors
  - Broader CLI redesign outside what is needed to support the new clip artifact/search contract

## Files To Inspect
- `mess/datasets/clip_index.py`
- `mess/search/faiss_index.py`
- `mess/search/search.py`
- `mess/search/__init__.py`
- `mess/__init__.py`
- `mess/training/export.py`
- `mess/training/context_export.py`
- `scripts/publish_faiss_index.py`
- `scripts/demo_recommendations.py`
- `tests/search/test_faiss_index.py`
- `tests/search/test_search.py`
- `tests/training/test_end_to_end_projection_artifact.py`
- `tests/search/test_search_init_exports.py`
- `tests/test_public_api.py`

## Contracts To Preserve
- Clip artifact row order must stay aligned with FAISS vector row order.
- Clip search must stay cosine-based (`IndexFlatIP`/normalized vectors) and preserve self-exclusion and near-time dedupe behavior unless explicitly changed.
- ClipIndex remains the source of truth for clip identity/provenance fields, especially `clip_id`, `recording_id`, `work_id`, and recording-level `split`.
- Artifact integrity behavior stays intact: immutable `artifact_version_id`, checksum validation, `latest.json` upload ordering, and local/S3 load paths.

## Plan
1. Replace reduced clip-location artifact metadata with a richer clip metadata row aligned to ClipIndex identity/provenance.
2. Update clip artifact save/load/build helpers and projection export to persist that metadata.
3. Add a canonical artifact-backed clip search API that resolves queries by `clip_id` or `(track_id, start_sec)`.
4. Update repo callers/exports to use the new clip artifact/search contract.
5. Run focused validation for artifact round-trips, clip search behavior, and public API exports.

## Validation
- `uv run pytest -q tests/search/test_faiss_index.py tests/search/test_search.py tests/training/test_end_to_end_projection_artifact.py tests/search/test_search_init_exports.py tests/test_public_api.py`
- `uv run ruff check mess/search mess/training scripts/publish_faiss_index.py scripts/demo_recommendations.py`

## Risks / Open Questions
- Persisted clip artifact schema is changing; loader compatibility for older clip artifacts needs an explicit decision and tests.
- `mess/training/context_export.py` currently uses clip artifact helpers for track-level outputs and may need a compatibility adjustment if clip metadata becomes stricter.
- The repo currently exposes `search_by_clip`; decide whether to replace it outright or keep it as a thin compatibility wrapper during the transition.
