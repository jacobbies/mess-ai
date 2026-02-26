# MESS-AI — Codex Agent Technical Guide

This file is the operational reference for coding agents working in this repo.
Primary goals:
- Keep `mess/` clean, modular, and test-backed.
- Preserve reproducible ML research workflows.
- Prefer implementation truth over stale docs when conflicts appear.

## 1) Project Identity

MESS-AI is an open-source Python 3.11+ ML library for music similarity research using MERT embeddings.
Core representation:
- 13 transformer layers
- 768 dimensions per layer
- Track-level embedding view is `[13, 768]`

Primary workflow:
`Audio WAV -> MERT features -> proxy targets -> layer discovery -> aspect-layer mapping -> FAISS similarity search`

## 2) Source-Of-Truth Rule

When docs disagree:
1. Trust code in `mess/` and tests in `tests/`.
2. Treat markdown docs as guidance, not contract.
3. Update docs and comments when you fix drift.

Important drift currently present:
- Some docs mention `mess/search/aspects.py` and `mess/search/layer_based_recommender.py`.
- Current implementation keeps aspect registry and resolver in `mess/probing/discovery.py`.
- Active search module is `mess/search/search.py`.

## 3) Repository Map

```text
mess/
  config.py
  datasets/
    base.py
    factory.py
    smd.py
    maestro.py
  extraction/
    __init__.py
    audio.py
    extractor.py
    pipeline.py
    storage.py
  probing/
    __init__.py
    discovery.py
    targets.py
    DISCOVERY_REFERENCE.md
  search/
    faiss_index.py
    search.py
scripts/
  extract_features.py
  run_probing.py
  demo_recommendations.py
  publish_faiss_index.py
  _NEEDS_UPDATE.txt
tests/
  test_config.py
  datasets/
  extraction/
  probing/
  search/
  tests.md
```

## 4) Data Layout And Contracts

Default root:
- `mess_config.data_root -> <project>/data`

Expected layout:
```text
data/
  audio/
    smd/wav-44/*.wav
    maestro/**/*.wav
  embeddings/
    smd-emb/
      raw/*.npy
      segments/*.npy
      aggregated/*.npy
    maestro-emb/
      raw/*.npy
      segments/*.npy
      aggregated/*.npy
  proxy_targets/
    *_targets.npz
  indices/
  metadata/
  waveforms/
```

Feature shape contracts:
- raw: `[num_segments, 13, time_steps, 768]`
- segments: `[num_segments, 13, 768]`
- aggregated: `[13, 768]`

Search view contract:
- `load_features()` returns `(features, track_names)`
- `features` shape should be `(n_tracks, feature_dim)` for FAISS.

Proxy target contract (`targets.py` -> `discovery.py`):
- NPZ file contains nested dict-like categories.
- Discovery reads with `data[category].item()[field]`.
- Missing required scalar targets causes target omission or warnings.

Deployment/runtime contract (EC2 dependency mode):
- This repo is imported as a Python dependency in the production music-recsys service running on EC2.
- Production workers may not have a full local dataset checkout; audio, embeddings, and FAISS artifacts are retrieved from S3 at runtime.
- Artifact flow must support both directions:
  - download for serving (`load_latest_from_s3`, `download_artifact_from_s3`)
  - upload/publish for training/index build jobs (`upload_artifact_to_s3`)
- Treat S3 artifact integrity as mandatory:
  - immutable `artifact_version_id`
  - checksum validation for downloaded files
  - pointer (`latest.json`) updated only after upload validation passes.

## 5) Core Config (`mess/config.py`)

Class: `MESSConfig`
- Default device is CPU (`MERT_DEVICE='cpu'`).
- Model default: `m-a-p/MERT-v1-95M`.
- Target sample rate: 24kHz.
- Segment duration: 5.0s.
- Overlap ratio: 0.5.
- Device batch defaults: CUDA 16, MPS 8, CPU 4.

CUDA flags:
- `MERT_CUDA_PINNED_MEMORY`
- `MERT_CUDA_NON_BLOCKING`
- `MERT_CUDA_MIXED_PRECISION`
- `MERT_CUDA_AUTO_OOM_RECOVERY`

Env overrides:
- `MESS_DEVICE`
- `MESS_WORKERS`
- `MESS_BATCH_SIZE`
- `MESS_CUDA_MIXED_PRECISION`

Key paths:
- `probing_results_file -> mess/probing/layer_discovery_results.json`
- `proxy_targets_dir -> data/proxy_targets`

Validation:
- `validate_config()` enforces positive durations/batch/workers and valid device in `{cpu,cuda,mps}`.

## 6) Extraction Stack (`mess/extraction`)

### `audio.py`
Responsibilities:
- Load audio (`torchaudio.load`)
- Convert stereo to mono
- Resample to 24kHz via thread-safe resampler cache
- Segment overlapping windows
- Validate audio integrity and minimum duration

Functions:
- `_get_resampler(orig_sr, target_sr)`
- `load_audio(audio_path, target_sr=24000)`
- `segment_audio(audio, segment_duration=5.0, overlap_ratio=0.5, sample_rate=24000)`
- `validate_audio_file(audio_path, check_corruption=True, min_duration=1.0)`

### `storage.py`
Responsibilities:
- Feature existence checks
- Selected or full feature loads
- Atomic writes with lock files

Functions:
- `_resolve_base_dir(output_dir, dataset=None)`
- `_resolve_filename(audio_path, track_id=None)`
- `features_exist(...)`
- `features_exist_for_types(...)`
- `load_features(...)`
- `load_selected_features(...)`
- `save_features(...)`

Persistence behavior:
- Uses `.locks/<track>.lock` with non-blocking `fcntl.flock`.
- Writes temp file then atomic move.
- Returns gracefully when file is currently locked by another worker.

### `extractor.py`
Class: `FeatureExtractor`

Responsibilities:
- Load MERT processor/model
- Move model to device with fallback chain
- Batched inference
- Optional mixed precision on CUDA
- OOM recovery by halving batch size
- Build feature views and optionally save

Key methods:
- `_load_model()`
- `_move_to_device()`
- `_extract_mert_features_batched()`
- `_extract_mert_features_batched_with_oom_recovery()`
- `_extract_feature_views_from_segments()`
- `extract_track_features(...)`
- `extract_track_features_safe(...)`
- `clear_gpu_cache()`
- `extract_dataset_features(...)` (delegates to pipeline)
- `estimate_extraction_time(...)` (delegates to pipeline)

### `pipeline.py`
Class: `ExtractionPipeline`

Responsibilities:
- Dataset-level orchestration
- Recursive fallback discovery for nested datasets
- Threaded CPU preprocessing + serialized GPU inference
- Bounded in-flight futures to control memory

Methods:
- `_discover_audio_files(audio_dir, file_pattern)`
- `run(...)`
- `run_parallel(...)`
- `_preprocess_worker(...)`
- `estimate_time(...)`

Parallel strategy:
- Worker threads: load/resample/segment
- Main thread: MERT inference + save
- Bounded queue: `max_in_flight = num_workers * 2`

## 7) Probing Stack (`mess/probing`)

### `discovery.py`
Global constants:
- `NUM_LAYERS = 13`
- `EMBEDDING_DIM = 768`

Model inspection utilities:
- `inspect_model(model_name=None)`
- `trace_activations(audio_path, model_name=None, segment_duration=5.0)`

Class: `LayerDiscoverySystem(dataset_name='smd', alpha=1.0, n_folds=5)`

Core methods:
- `load_features(audio_files)` reads raw embeddings and reduces over segments/time.
- `load_targets(audio_files)` reads scalar proxy targets from NPZ.
- `_probe_single(X, y)` runs Ridge + StandardScaler with CV.
- `discover(n_samples=50)` runs full probing grid.
- `best_layers(results)` returns best layer per target + confidence.
- `discover_and_save(n_samples=50, path=None)`
- `save(results, path=None)`

Confidence policy:
- R2 > 0.8 -> `high`
- R2 > 0.5 -> `medium`
- else -> `low`

Aspect mapping:
- `ASPECT_REGISTRY` currently lives here.
- `resolve_aspects(min_r2=0.5, results_path=None)` maps user-facing aspects to validated layers.

### `targets.py`
Class: `MusicalAspectTargets`

Responsibilities:
- Generate proxy targets from audio for categories:
  - rhythm
  - harmony
  - timbre
  - articulation
  - dynamics
  - phrasing

Key methods:
- `generate_all_targets(audio_path)`
- `validate_target_structure(targets)`
- `_generate_rhythm_targets(audio)`
- `_generate_harmony_targets(audio)`
- `_generate_timbre_targets(audio)`
- `_generate_articulation_targets(audio)`
- `_generate_dynamics_targets(audio)`
- `_generate_phrasing_targets(audio)`

Dataset-level helper:
- `create_target_dataset(audio_dir, output_dir, validate=True, use_mlflow=True, dataset_id=None)`
- When `dataset_id` is provided, also generates MIDI expression targets via `midi_targets.py`

CLI:
- module `main()` accepts `--dataset`, `--no-validate`, `--no-mlflow`.

### `midi_targets.py`
Class: `MidiExpressionTargets`

Responsibilities:
- Generate expression proxy targets from MIDI performance data (category: `expression`)
- Computes: rubato, velocity_mean, velocity_std, velocity_range, articulation_ratio, tempo_variability, onset_timing_std
- All targets are pre-reduced scalars (single-element arrays with `'first'` reduction)
- Lazy `import pretty_midi` — graceful when library is not installed

Key methods:
- `generate_expression_targets(midi_path)` → `{'expression': {field: np.ndarray}}`
- `_collect_notes(pm)` — all non-drum notes sorted by onset
- `_compute_rubato(ioi_positive)` — std of consecutive IOI ratios
- `_compute_articulation_ratio(durations, ioi)` — mean(duration/IOI)
- `_compute_tempo_variability(ioi_positive)` — std of local BPM
- `_compute_onset_timing_std(onsets)` — std of deviations from median-IOI grid

Helper function:
- `resolve_midi_path(audio_path, dataset_id)` → `Path | None`
  - SMD: sibling `midi/` directory, same stem
  - MAESTRO: co-located, same stem with `.midi` extension
  - Returns `None` if no MIDI file found

Design notes:
- Expression targets are **optional** — `OPTIONAL_CATEGORIES = {'expression'}` in discovery.py
- Tracks without MIDI are excluded from expression probing but included in audio-derived probing
- `load_targets` uses NaN for missing optional targets; `_probe_single` filters NaN rows before CV

## 8) Search Stack (`mess/search/search.py`)

Functions:
- `load_features(features_dir, layer=None)`
- `build_index(features)`
- `find_similar(query_track, features, track_names, k=10, exclude_self=True)`
- `search_by_aspect(query_track, aspect, features_dir, k=10)`
- `search_by_aspects(query_track, aspect_weights, features_dir, k=10, min_r2=0.5, scale_by_r2=True)`

Search mechanics:
- Uses `faiss.IndexFlatIP`.
- Applies L2 normalization for cosine similarity equivalence.
- `search_by_aspect` imports `resolve_aspects` from probing package.
- `search_by_aspects` resolves multiple aspects to layers and fuses normalized layer vectors via weighted combination.

Operational note:
- No `mess/search/__init__.py` currently.
- Import directly from `mess.search.search`.

## 9) Dataset Layer (`mess/datasets`)

Base class:
- `BaseDataset` defines `dataset_id`, `audio_dir`, `embeddings_dir`, `name`, `description`.
- Provides `aggregated_dir`, `get_audio_files()`, `get_feature_path()`, `exists()`, `__len__`.

Factory:
- `DatasetFactory.get_dataset(name, data_root=None)`
- `DatasetFactory.create_dataset(...)` alias
- `DatasetFactory.get_available_datasets()`
- `DatasetFactory.register_dataset(name, cls)`

Concrete datasets:
- `SMDDataset`
  - audio: `data/audio/smd/wav-44`
  - embeddings: `data/embeddings/smd-emb`
- `MAESTRODataset`
  - audio: `data/audio/maestro`
  - embeddings: `data/embeddings/maestro-emb`

## 10) CLI Scripts And Status

Stable scripts:
- `scripts/extract_features.py`
- `scripts/run_probing.py`
- `scripts/demo_recommendations.py`

Known outdated scripts (`scripts/_NEEDS_UPDATE.txt`):
- `build_layer_indices.py`
- `demo_layer_search.py`
- `evaluate_layer_indices.py`
- `evaluate_similarity.py`

Interpret outdated scripts as experimental; do not rely on them for production behavior without refactor.
Linting scope note:
- Ruff checks are intentionally scoped to maintained paths (`mess/`, `tests/`, and stable scripts). Linting for outdated scripts should be addressed only when those scripts are actively refactored.

## 11) Tests: Expectations And Conventions

Reference: `tests/tests.md`

Current suite:
- 99 tests
- fast runtime (~3.5s target)
- pytest markers: `unit`, `integration`, `slow`, `gpu`

Hard testing principles:
- Do not load real MERT model in unit tests.
- Use mocks for heavy deps and model calls.
- Use `tmp_path` for filesystem I/O.
- Keep tests mirrored by module domain:
  - `mess/extraction` -> `tests/extraction`
  - `mess/probing` -> `tests/probing`
  - `mess/search` -> `tests/search`

Important fixture patterns:
- Root fixtures in `tests/conftest.py`.
- Dataset factory registry restoration in `tests/datasets/conftest.py`.
- Discovery unit tests construct object via `object.__new__` to bypass heavy init.

## 12) Dependency And Runtime Stack

Package + build:
- `hatchling` build backend
- `uv` dependency management

Core libs:
- numpy, scipy, scikit-learn, pandas
- torch 2.10.0, torchaudio 2.10.0, torchcodec 0.10.0
- transformers >= 4.38, safetensors
- librosa, soundfile, nnaudio
- faiss-cpu
- mlflow >= 2.10
- tqdm

Dev/test:
- pytest >= 8
- pytest-cov
- pytest-mock
- ruff
- mypy

Platform notes:
- Linux pulls CUDA wheels via `tool.uv.sources` and `pytorch-cu128` index.
- macOS can use MPS backend.

## 13) Coding Conventions

Language/style:
- Python 3.11+
- PEP 8
- 4-space indentation
- Type hints on public interfaces
- Concise, accurate docstrings for public functions/classes

Code organization:
- Put reusable logic in `mess/`.
- Put user workflows and orchestration in `scripts/`.
- Keep notebook code exploratory, not library-critical.
- Keep user-facing documentation in `docs/`; keep implementation-oriented references near module code when they primarily serve maintainers.

Git/data hygiene:
- Never commit large binary artifacts from `data/`, `mlruns/`, or generated embeddings.
- Keep one logical change per commit.
- Use short imperative commit messages.

## 14) Agentic Workflow For Codex

Default execution protocol for code changes:
1. Read task + inspect affected modules/tests first.
2. Confirm implementation truth and identify doc drift.
3. Implement minimal coherent change.
4. Add or update tests when behavior changes.
5. Run quality checks for affected scope:
   - `uv run ruff check .`
   - `uv run pytest -v` (or targeted pytest commands)
   - `uv run mypy mess` when touching typed core interfaces or before merge/release
6. Run broader suite as needed.
7. Summarize what changed, what was validated, and residual risks.

For extraction-related tasks:
1. Check config impacts (`mess/config.py`).
2. Verify shape/path contracts in `storage.py` and tests.
3. Confirm parallel behavior in `pipeline.py` when touching throughput logic.

For probing/aspect tasks:
1. Update target generation in `targets.py` when adding proxy signals.
2. Update `SCALAR_TARGETS` in `discovery.py`.
3. Verify `ASPECT_REGISTRY` references only existing scalar target names.
4. Re-run probing workflow if mapping behavior is changed.

For search tasks:
1. Preserve normalized cosine behavior in FAISS.
2. Validate query-track existence checks and result ordering.
3. Test layer-specific loading paths when changing aspect search.

## 15) Operational Command Cheatsheet

Environment:
```bash
uv sync --group dev
```

Tests:
```bash
uv run pytest -v
uv run pytest --cov=mess --cov-report=term-missing
uv run pytest -m unit
uv run pytest tests/probing/test_discovery.py -v
```

Linting and type checks:
```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy mess
```

Feature extraction:
```bash
uv run python scripts/extract_features.py --dataset smd
uv run python scripts/extract_features.py --dataset maestro --device cuda --workers 4
uv run python scripts/extract_features.py --dataset smd --feature-level aggregated
```

Layer discovery:
```bash
uv run python scripts/run_probing.py
uv run python scripts/run_probing.py --samples 30 --alpha 0.5 --folds 10
```

Recommendations:
```bash
uv run python scripts/demo_recommendations.py --track "<TRACK_ID>"
uv run python scripts/demo_recommendations.py --track "<TRACK_ID>" --aspect brightness --k 10
uv run python scripts/demo_recommendations.py --track "<TRACK_ID>" --aspects "brightness=0.7,phrasing=0.3" --k 10
```

MLflow:
```bash
uv run mlflow ui
```

## 16) Pre-Commit Checklist For Agents

- Did you preserve data shape contracts?
- Did you preserve path contracts (`audio_dir`, `embeddings_dir`, proxy targets)?
- Did you update/add tests for behavioral changes?
- Did `uv run pytest` pass for affected scope?
- Did `uv run ruff check .` pass for affected scope?
- Did `uv run mypy mess` pass when typed core interfaces changed or before merge/release?
- Did you avoid touching outdated scripts unless explicitly requested?
- Did you avoid committing generated large files?

## 0) North-Star Product Goal

MESS-AI should evolve toward a modern AI music streaming experience centered on expressive, content-based similarity search over MERT embeddings.

North-star capability statement:
MESS should allow a user to select a 5-second musical gesture and retrieve passages across performances and composers that match its expressive character, optionally constrained by interpretable aspects (rubato, dynamics, articulation, harmony), and eventually guided by natural-language descriptions of mood or structure.

North-star query suite (the system should eventually dominate these):

1. Segment-level expressive search (core primitive)
- "Find performances that feel like this 5-second cadenza."
- "Find passages with similar left-hand arpeggiated swell."
- "Find a similar emotional build-up but more restrained."
- "Find something like this but darker in tone."

2. Aspect-constrained similarity
- "Find pieces similar in phrasing but not tempo."
- "Similar rubato profile, ignore tempo differences."
- "Same dynamic contour but different harmonic density."
- "More legato than this passage."

3. Cross-performance comparison
- "How does Argerich's phrasing differ from Zimerman here?"
- "Compare the same timestamp across two performances."
- "Who stretches this cadence the most?"

4. Text-guided expressive retrieval (future multimodal)
- "Find something that feels like a rainy Paris evening."
- "Melancholic but hopeful."
- "Heroic but restrained."

5. Structure-aware search
- "Find climaxes similar to this one."
- "Find cadences with similar harmonic resolution."
- "Find transitions that modulate similarly."

6. Research/debug queries (internal quality gate)
- "Which MERT layer best encodes articulation?"
- "Are dynamics separable from harmonic density?"
- "What's the R2 stability across folds?"
- "Does fine-tuning collapse expressive variance?"

## Research Roadmap

Ordered experiment sequence toward learned expressive similarity:

1. **MIDI-derived expression targets** — rubato, velocity dynamics, articulation from MIDI. Tests whether MERT layers encode expression separately from content. **(in progress)**
2. **Human eval set** — 100-300 triplet judgments as external ground truth.
3. **Baseline recall@K** — Measure cosine retrieval against both proxy and human labels.
4. **Diagonal weighting** → **linear projection** → **MLP projection** — progressive geometry learning, each validated before advancing.
5. **Two-stage reranking** — only if recall@K is high but ranking within top-K is poor.

Future MIDI workstreams (after expression targets validated):
- Cross-performance passage pairing via MIDI alignment (contrastive training pairs)
- Content-invariant embedding learning (conditioned on MIDI score)
- Segment-level retrieval (5-second gesture search — the north-star primitive)

Key decision rules:
- If Recall@K is low → improve embedding geometry, not reranking
- If proxy targets disagree with human eval → fix proxy quality first
- If linear projection saturates → try MLP, then consider reranking
- Two-stage only justified when scorer is non-decomposable AND recall ceiling is high enough

Strategic direction:
- Prioritize embedding-first retrieval and expressive musical understanding over metadata-first or collaborative-filtering-first approaches.
- MIDI data is an anchor for disentangling content from expression — use it as ground truth for what "expressive similarity" means, not just as another feature source.
- Keep this repository research-first: robust dataset handling, extraction, probing, and search infrastructure in `mess/` that enables rapid experimentation.

Assessment heuristic for roadmap decisions:
1. Does this change improve expressive user-intent retrieval quality or enable experiments that can improve it?
2. Does this keep the core `mess/` library modular, testable, and reproducible?
3. Does this avoid over-coupling the core system to metadata-only assumptions?

Ignore CLAUDE.md

This document is intended to keep Codex aligned with real code behavior and reproducible ML workflows in this repository.
