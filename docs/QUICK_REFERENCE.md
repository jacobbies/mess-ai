# MESS-AI Quick Reference

Quick lookup guide for common tasks, layer mappings, and decision trees.

---

## Embedding Shape Cheat Sheet

```python
# MERT Output Formats
raw_features:        [segments, 13, time_steps, 768]  # Full temporal resolution
segment_features:    [segments, 13, 768]              # Time-averaged per segment
aggregated_features: [13, 768]                        # Track-level (used for similarity)
```

**Interpretation:**
- `13` = Number of MERT layers (0-12)
- `768` = Embedding dimensionality
- `time_steps` = Variable (depends on audio length)
- `segments` = Audio split into 5-second chunks

---

## Layer-to-Aspect Mapping Table

| Layer | Musical Aspect | R² Score | Use Case |
|-------|---------------|----------|----------|
| **0** | Spectral brightness | **0.944** | Find tracks with similar timbral brightness (piano vs cello) |
| **1** | Timbral texture | **0.922** | Find tracks with similar surface texture (smooth vs rough) |
| **2** | Acoustic structure | **0.933** | Find tracks with similar structural patterns |
| 3 | Melodic contour | 0.867 | (Experimental) Similar melodic shapes |
| 4 | Temporal patterns | 0.673 | (Lower confidence) Rhythmic patterns |
| 7 | Phrasing | 0.781 | (Promising) Musical phrase structure |
| 8-12 | Unknown | <0.5 | Not yet validated |

**Quick Decision:**
- Want **timbral similarity**? → Use Layer 0
- Want **textural similarity**? → Use Layer 1
- Want **structural similarity**? → Use Layer 2
- Want **multi-aspect**? → Combine layers with weights

---

## Similarity Metrics Decision Tree

```
Do you care about vector direction or magnitude?
│
├─ Direction (angle between vectors)
│  └─ USE: Cosine Similarity
│     - Range: -1 to 1 (embeddings: ~0 to 1)
│     - Formula: cos_sim(A, B) = (A · B) / (||A|| × ||B||)
│     - Best for: MERT embeddings (normalized)
│
└─ Magnitude (absolute distance)
   └─ USE: Euclidean Distance
      - Range: 0 to ∞
      - Formula: sqrt(Σ(A_i - B_i)²)
      - Best for: Physical measurements
```

**For MERT embeddings**: **Always use Cosine Similarity** (empirically validated)

---

## FAISS Index Types

| Index Type | Speed | Accuracy | Use When |
|------------|-------|----------|----------|
| **IndexFlatIP** | Fast | 100% exact | <100K tracks (our default) |
| IndexIVF | Faster | ~95% | 100K-1M tracks |
| IndexHNSW | Fastest | ~98% | >1M tracks |
| IndexPQ | Very fast | ~90% | Memory-constrained |

**Current MESS-AI**: `IndexFlatIP` (exact search, perfect for our 50-1000 track scale)

**Upgrade path**:
- At 100K tracks → Switch to `IndexIVFFlat`
- At 1M tracks → Switch to `IndexHNSW`

---

## Common Tasks

### Get Recommendations by Aspect

```python
from pipeline.query.layer_based_recommender import LayerBasedRecommender

recommender = LayerBasedRecommender()

# Single aspect (using validated layer)
results = recommender.recommend_by_aspect(
    "Beethoven_Op027No1-01",
    aspect="spectral_brightness",  # or "timbral_texture", "acoustic_structure"
    n_recommendations=5
)
```

### Get Recommendations by Layer

```python
# Direct layer access (if you know which layer you want)
results = recommender.recommend_by_layer(
    "Beethoven_Op027No1-01",
    layer=0,  # 0=brightness, 1=texture, 2=structure
    n_recommendations=5
)
```

### Multi-Aspect Recommendations

```python
# Combine multiple aspects with custom weights
results = recommender.multi_aspect_recommendation(
    "Beethoven_Op027No1-01",
    aspect_weights={
        'spectral_brightness': 0.6,  # 60% importance
        'timbral_texture': 0.4,      # 40% importance
    },
    n_recommendations=5
)
```

**When to use each:**
- **One clear aspect** → `recommend_by_layer()` or `recommend_by_aspect()`
- **Multiple aspects with priorities** → `multi_aspect_recommendation()`

---

## Audio Processing Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sample Rate | 24,000 Hz | MERT pre-training rate (Nyquist: 12kHz max frequency) |
| Segment Duration | 5 seconds | Balances context window vs memory |
| Overlap Ratio | 0.5 (50%) | Smooth temporal transitions |
| Batch Size | 8 | Optimized for M3 Pro memory |
| Max Workers | 4 | CPU cores for parallel processing |

**Why 24kHz?**
- MERT was pre-trained on 24kHz audio
- Higher rates waste computation (model can't use extra frequencies)
- Lower rates lose information (Nyquist: max frequency = sample_rate / 2)

---

## File Path Quick Reference

```python
from pipeline.extraction.config import pipeline_config

# Key paths (auto-detected, no hardcoding needed)
pipeline_config.project_root              # /path/to/mess-ai
pipeline_config.data_dir                  # /path/to/mess-ai/data
pipeline_config.audio_dir                 # data/audio/smd/wav-44
pipeline_config.smd_embeddings_dir        # data/embeddings/smd-emb
pipeline_config.aggregated_features_dir   # data/embeddings/smd-emb/aggregated
pipeline_config.probing_results_file      # pipeline/probing/layer_discovery_results.json
pipeline_config.proxy_targets_dir         # data/proxy_targets
```

**Never hardcode paths!** Always use `pipeline_config` for portability.

---

## Common Pitfalls and Solutions

### Pitfall 1: Averaging All Layers

```python
# ❌ BAD: Destroys layer specialization
avg_embedding = features.mean(axis=0)
```

```python
# ✅ GOOD: Use validated layer for specific aspect
brightness_layer = features[0]  # R²=0.944 for brightness
```

**Why**: Layer 0 alone (R²=0.944) beats averaged layers (R²~0.65) for brightness.

### Pitfall 2: Using Euclidean Distance

```python
# ❌ Suboptimal for normalized embeddings
dist = np.linalg.norm(emb1 - emb2)
```

```python
# ✅ Correct for MERT embeddings
similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

**Why**: MERT embeddings are roughly normalized; direction matters more than magnitude.

### Pitfall 3: Hardcoded Paths

```python
# ❌ BAD: Breaks on other machines
data_dir = "/Users/username/mess-ai/data"
```

```python
# ✅ GOOD: Portable configuration
from pipeline.extraction.config import pipeline_config
data_dir = pipeline_config.data_dir
```

### Pitfall 4: Combining Layers Before Comparison

```python
# ❌ BAD: Loses layer-specific information
combined = 0.6 * features[0] + 0.4 * features[1]  # Mix embeddings first
similarity = cosine(combined_query, combined_candidate)
```

```python
# ✅ GOOD: Compare per-layer, then combine similarities
sim_layer0 = cosine(query[0], candidate[0])
sim_layer1 = cosine(query[1], candidate[1])
final_score = 0.6 * sim_layer0 + 0.4 * sim_layer1  # Combine similarities
```

**Why**: Preserves layer specialization and allows weighted aspect control.

---

## R² Score Interpretation

| R² Score | Interpretation | Action |
|----------|---------------|--------|
| **0.9 - 1.0** | Excellent fit | **Use this layer confidently** |
| **0.8 - 0.9** | Good fit | Use with awareness |
| 0.7 - 0.8 | Promising | Experimental use |
| 0.5 - 0.7 | Weak correlation | Avoid for production |
| <0.5 | No correlation | Do not use |

**Our validated layers:**
- Layer 0: 0.944 → Excellent
- Layer 1: 0.922 → Excellent
- Layer 2: 0.933 → Excellent

---

## Command Line Scripts

### Extract Features from Audio
```bash
python scripts/extract_features.py --dataset smd
# Output: data/embeddings/smd-emb/aggregated/*.npy
```

### Run Layer Discovery
```bash
python scripts/run_probing.py
# Output: pipeline/probing/layer_discovery_results.json
```

### Inspect Embeddings
```bash
python scripts/inspect_embeddings.py
# Shows: shapes, statistics, layer specializations, cosine similarities
```

### Get Recommendations
```bash
# Default (Layer 0 - brightness)
python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01"

# Specific aspect
python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --aspect timbral_texture

# More results
python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --n 10
```

---

## Dataset Information

### Saarland Music Dataset (SMD)
- **Tracks**: 50 classical piano performances
- **Format**: WAV, 44.1kHz (resampled to 24kHz for MERT)
- **Duration**: ~5 hours total
- **Location**: `data/audio/smd/wav-44/`
- **Embeddings**: `data/embeddings/smd-emb/aggregated/` (50 files × 40KB = ~2MB)

### Naming Convention
```
Bach_BWV849-01_001_20090916-SMD.npy
│    │        │   │   │          │
│    │        │   │   │          └─ Dataset tag
│    │        │   │   └─ Recording date
│    │        │   └─ Take number
│    │        └─ Movement/section
│    └─ Catalog number (BWV, Op, KV, etc.)
└─ Composer
```

---

## Typical Workflow

```
1. Add audio files to data/audio/{dataset}/wav-44/
2. Extract features: python scripts/extract_features.py --dataset {dataset}
3. (Optional) Validate layers: python scripts/run_probing.py
4. Explore embeddings: python scripts/inspect_embeddings.py
5. Get recommendations: python scripts/demo_recommendations.py --track {name}
6. (For research) Use Jupyter notebooks for visualization
```

---

## When to Re-run Layer Discovery

Re-run `python scripts/run_probing.py` when:
- ✅ You add a new proxy target (new musical aspect to validate)
- ✅ You fine-tune the MERT model
- ✅ You significantly expand the dataset (>2x tracks)
- ❌ You just add more tracks of the same type (no need)
- ❌ You change similarity search parameters (no need)

**Current validated results**: `pipeline/probing/layer_discovery_results.json`

---

## Memory Usage Estimates

| Data Type | Per Track | 50 Tracks | 1000 Tracks |
|-----------|-----------|-----------|-------------|
| Audio (44.1kHz WAV) | ~50 MB | 2.5 GB | 50 GB |
| Raw features | ~10 MB | 500 MB | 10 GB |
| Aggregated features | ~40 KB | 2 MB | 40 MB |
| FAISS index (aggregated) | ~40 KB | 2 MB | 40 MB |

**Recommendation**: For >10K tracks, use compressed storage (HDF5, Zarr) for raw features.

---

## Performance Benchmarks (M3 Pro)

| Operation | Time | Notes |
|-----------|------|-------|
| Feature extraction | ~3 seconds/track | MPS acceleration |
| Layer discovery (full) | ~10-15 minutes | 50 tracks, 10 proxy targets |
| FAISS index build | <1 second | 50 tracks |
| Similarity search | <1 ms | Single query, 50 tracks |
| Recommendation (multi-aspect) | <5 ms | Weighted combination |

**Bottleneck**: Feature extraction is compute-intensive; everything else is fast.

---

## Debugging Tips

### Check if embeddings exist
```bash
ls -lh data/embeddings/smd-emb/aggregated/ | head -10
```

### Verify embedding shape
```python
import numpy as np
emb = np.load("data/embeddings/smd-emb/aggregated/Bach_BWV849-01_001_20090916-SMD.npy")
print(emb.shape)  # Should be (13, 768)
```

### Test layer discovery results
```python
import json
with open("pipeline/probing/layer_discovery_results.json") as f:
    results = json.load(f)
print(results["layer_0"]["spectral_centroid"]["r2_score"])  # Should be ~0.944
```

### Verify FAISS index
```python
from pipeline.query.layer_based_recommender import LayerBasedRecommender
rec = LayerBasedRecommender()
print(f"Loaded {rec.index.ntotal} tracks")  # Should match number of .npy files
```

---

## Further Reading

- **Full domain concepts**: [docs/CONCEPTS.md](CONCEPTS.md)
- **Project architecture**: [README.md](../README.md)
- **Project instructions for Claude**: [CLAUDE.md](../CLAUDE.md)
- **MERT paper**: https://arxiv.org/abs/2306.00107
- **FAISS documentation**: https://github.com/facebookresearch/faiss
- **librosa documentation**: https://librosa.org/doc/latest/index.html
