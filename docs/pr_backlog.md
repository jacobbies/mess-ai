# PR/TASK HANDOFF
- Task/PR: Simplify dataset definitions and dataset factory
- Branch: `pr09-simplify-dataset-layer`
- Status: `completed`

## Objective
- Reduce `mess.datasets.base` and `mess.datasets.factory` complexity without changing dataset path behavior, clip-index helpers, or public dataset names.

## Scope
- In:
  - `BaseDataset` contract shape
  - `SMDDataset` and `MAESTRODataset` definitions
  - `DatasetFactory` registry/lookup behavior
  - Dataset tests directly covering those contracts
- Out:
  - Search/training/probing refactors
  - Clip index or metadata schema changes
  - Changes to dataset consumer behavior outside what is required for the simpler dataset contract

## Files To Inspect
- `mess/datasets/base.py`
- `mess/datasets/factory.py`
- `mess/datasets/__init__.py`
- `tests/datasets/test_base.py`
- `tests/datasets/test_factory.py`
- `tests/datasets/test_smd.py`
- `tests/datasets/test_maestro.py`
- `tests/datasets/conftest.py`
- `mess/probing/discovery.py`
- `scripts/extract_features.py`
- `scripts/inspect_embeddings.py`

## Contracts To Preserve
- `DatasetFactory.get_dataset("smd"|"maestro")` continues to return the same public dataset types.
- Dataset instances keep the same resolved paths for `audio_dir`, `embeddings_dir`, `aggregated_dir`, `segments_dir`, `metadata_table_path`, and `clip_index_path`.
- `BaseDataset.build_clip_index()` and metadata-loading behavior stay unchanged.
- Custom dataset registration via `DatasetFactory.register_dataset()` continues to work.
- Public exports from `mess.datasets` and `mess` remain stable.

## Plan
1. Replace abstract-property-heavy dataset definitions with a smaller concrete base contract.
2. Convert built-in datasets to declarative class-level definitions.
3. Simplify factory lookup/registration without changing public method names.
4. Update tests to cover the simpler dataset definition style and preserved compatibility.
5. Run focused validation for dataset contracts and key callers.

## Validation
- `uv run pytest -q tests/datasets/test_base.py tests/datasets/test_factory.py tests/datasets/test_smd.py tests/datasets/test_maestro.py`
- `uv run pytest -q tests/probing/test_discovery.py tests/test_public_api.py`

## Validation Results
- `uv run ruff check mess/datasets/base.py mess/datasets/factory.py tests/datasets/test_base.py tests/datasets/test_factory.py tests/datasets/test_smd.py tests/datasets/test_maestro.py tests/datasets/test_init.py tests/probing/test_discovery.py tests/test_public_api.py`: pass
- `uv run pytest -q tests/datasets/test_base.py tests/datasets/test_factory.py tests/datasets/test_smd.py tests/datasets/test_maestro.py tests/datasets/test_init.py tests/probing/test_discovery.py tests/test_public_api.py`: pass

## Risks / Open Questions
- Direct third-party subclassing of `BaseDataset` may rely on overriding properties instead of class attributes; compatibility should be preserved where practical.
- The factory registry is still mutable by design for tests and extensions, so simplification should not remove that capability.
