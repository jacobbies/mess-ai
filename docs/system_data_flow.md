# MESS-AI System Data Flow

This document describes how data moves through the system end-to-end, from raw audio to training outputs and production retrieval artifacts.

## 1) Dataset Resolution And Paths

The dataset layer resolves source and destination paths for each dataset (`smd`, `maestro`) and provides canonical access to audio files, embedding directories, metadata, and clip index locations.

Key modules:
- `mess/datasets/factory.py`
- `mess/datasets/base.py`
- `mess/config.py`

## 2) Audio To MERT Features (Extraction Lane)

The extraction stack performs:
1. Audio decode/load.
2. Mono conversion and resampling to 24kHz.
3. 5-second overlapping segmentation (default overlap ratio `0.5`).
4. MERT forward pass.
5. Feature view construction and persistence.

Primary feature contracts:
- `raw`: `[num_segments, 13, time_steps, 768]`
- `segments`: `[num_segments, 13, 768]`
- `aggregated`: `[13, 768]`

Persistence is lock-protected and atomic to avoid partial writes during parallel extraction.

Key modules:
- `mess/extraction/audio.py`
- `mess/extraction/extractor.py`
- `mess/extraction/pipeline.py`
- `mess/extraction/storage.py`

## 3) Segment Embeddings To Clip Index Contract

Segment embedding files (`segments/*.npy`) are converted into clip-level rows with:
- `clip_id`, `dataset_id`, `recording_id`, `track_id`
- `segment_idx`, `start_sec`, `end_sec`
- `split`, `embedding_path`

Splits are assigned at `recording_id` level (never clip-level) to prevent leakage.

Key modules:
- `mess/datasets/clip_index.py`
- `scripts/build_clip_index.py`

## 4) Proxy Targets Generation (Probing Inputs)

Targets are generated from audio and saved as NPZ nested categories (for example `timbre.*`, `dynamics.*`).
Optional MIDI-derived expression targets can also be generated when MIDI data is available.

Key modules:
- `mess/probing/targets.py`
- `mess/probing/midi_targets.py`

## 5) Layer Discovery And Aspect Resolution

Discovery loads embeddings and targets, runs cross-validated Ridge probes for each (layer, target), and writes discovery results.
Aspect resolution maps user-facing aspects (for example `brightness`, `phrasing`) to validated layers from discovery outputs.

Key module:
- `mess/probing/discovery.py`

## 6) Baseline Retrieval/Search Lane

Search code:
1. Loads vectors from embeddings.
2. Normalizes vectors for cosine similarity.
3. Builds FAISS index (`IndexFlatIP`).
4. Returns nearest neighbors for:
   - track-level retrieval,
   - clip-level retrieval,
   - aspect-weighted retrieval.

Key module:
- `mess/search/search.py`

## 7) Projection Training Lane (`mess/training`)

Training consumes clip-indexed vectors and metadata, then:
1. Builds a FAISS retrieval index on target-encoder embeddings.
2. Mines positives/negatives with guardrails (time separation, recording constraints).
3. Optimizes a projection head with multi-positive InfoNCE.
4. EMA-updates the target encoder.
5. Periodically refreshes retrieval index geometry.
6. Saves config, metrics, and checkpoint state dicts.

Key modules:
- `mess/datasets/stores.py`
- `mess/training/config.py`
- `mess/training/mining.py`
- `mess/training/losses.py`
- `mess/training/index.py`
- `mess/training/trainer.py`
- `scripts/train_retrieval_ssl.py`

## 8) Projection Export To Production Artifact

After training, projected clip vectors are exported as a deployable FAISS artifact with:
- `index.faiss`
- `manifest.json`
- lookup metadata (`clip_locations`)
- `checksums.json`

Artifacts include schema/version metadata and checksums for integrity validation.

Key modules:
- `mess/training/export.py`
- `mess/search/faiss_index.py`

## 9) S3 Publish And Serving Runtime Load

Publish path:
1. Upload artifact files.
2. Validate uploaded object metadata.
3. Write `latest.json` pointer only after validation succeeds.

Serving/load path:
1. Read `latest.json`.
2. Download artifact.
3. Validate checksums and version consistency.
4. Load FAISS artifact for runtime retrieval.

Key module:
- `mess/search/faiss_index.py`

## 10) Evaluation And Gate Decisions

Evaluation scripts compute retrieval metrics and expressive checks (sanity, aspect faithfulness, counterfactual behavior), then apply gate thresholds for pass/warn/fail decisions.

Key modules/scripts:
- `mess/evaluation/retrieval.py`
- `mess/evaluation/aspects.py`
- `mess/evaluation/gates.py`
- `scripts/evaluate_retrieval.py`
- `scripts/evaluate_expressive_retrieval.py`

## Summary

Primary end-to-end path:

`Audio -> Extraction -> Embeddings -> Clip Index -> (Probing/Aspects, Search, Training) -> Projection Artifact -> S3 Pointer -> Serving Retrieval`
