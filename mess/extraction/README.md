# mess/extraction/ — MERT Feature Extraction

Extracts hierarchical embeddings from audio using the MERT transformer model (13 layers, 768 dims per layer).

## Pipeline

```
Audio (.wav) → load + mono + 24kHz → 5s segments (50% overlap) → MERT inference → aggregate
```

Output formats per track:
- **raw**: `[segments, 13, time, 768]` — full temporal resolution
- **segments**: `[segments, 13, 768]` — time-averaged per segment
- **aggregated**: `[13, 768]` — track-level vector (used for similarity search)

## Module Structure

| File | Lines | Purpose |
|------|-------|---------|
| `extractor.py` | ~400 | `FeatureExtractor` — model lifecycle, MERT inference, track-level extraction |
| `audio.py` | ~170 | Standalone audio loading, segmentation, validation (no model dependency) |
| `cache.py` | ~130 | `FeatureCache` — .npy feature save/load/exists with consolidated path resolution |
| `pipeline.py` | ~370 | `ExtractionPipeline` — dataset-level batch processing (sequential + parallel) |
| `__init__.py` | ~25 | Public API re-exports |

## Usage

```python
# Standard: extract features for a track
from mess.extraction.extractor import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_track_features("audio.wav", output_dir="data/processed/features")

# Batch: extract entire dataset (backward-compatible convenience method)
extractor.extract_dataset_features(audio_dir, output_dir, num_workers=4)

# Lightweight: preprocess audio without loading MERT
from mess.extraction.audio import load_audio, segment_audio

audio = load_audio("track.wav", target_sr=24000)
segments = segment_audio(audio, segment_duration=5.0)

# Direct pipeline control
from mess.extraction.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline(extractor)
results = pipeline.run(audio_dir, output_dir, num_workers=4)
```

## Design Decisions

- **audio.py functions are standalone** — no `self`, no model. Researchers can preprocess audio or validate files without loading a 95M-parameter model into memory.
- **FeatureCache consolidates path logic** — the `track_id or stem` + `dataset subdirectory` pattern was duplicated across three methods; now it lives in `resolve_path()`.
- **ExtractionPipeline uses composition** — wraps a `FeatureExtractor` instance rather than inheriting. The pipeline orchestrates; the extractor does inference.
- **Backward compatibility** — `FeatureExtractor.extract_dataset_features()` still exists as a convenience delegator to `ExtractionPipeline`, so `scripts/extract_features.py` works unchanged.
