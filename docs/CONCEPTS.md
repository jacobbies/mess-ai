# Core Concepts: Embeddings, Audio ML, and Music Similarity

## 1. What Are Embeddings?

An embedding is a dense vector representation of data that captures semantic meaning.

**Key Insight**: Similar things should have similar embeddings.

### Mathematical Definition
- Input: Raw data (audio, text, image)
- Output: Fixed-size vector (e.g., 768 float values)
- Property: Semantic similarity ≈ vector similarity

### Example
```
"Bright piano melody"  → [0.8, 0.2, 0.6, ..., 0.3]  (768 dims)
"Dark cello passage"   → [0.1, 0.7, 0.2, ..., 0.9]  (768 dims)
                          ↑ These vectors are far apart (low cosine similarity)
```

### Why Embeddings Work
- **Dimensionality reduction**: 30 seconds audio (720,000 samples) → 768 numbers
- **Learned representations**: Neural networks discover meaningful features automatically
- **Transfer learning**: Pre-trained on massive datasets, works for your specific use case

---

## 2. MERT: Music Understanding Model

MERT (Music Understanding Model with Large-Scale Self-Supervised Training) is a transformer-based model trained on 160K hours of music.

### Architecture
- **12 transformer blocks + 1 input layer = 13 layers total**
- **Each layer**: 768-dimensional embeddings
- **Training**: Self-supervised (no manual labels needed)
- **Key innovation**: Learns hierarchical musical representations

### Why Transformers for Audio?
- **Attention mechanism**: Model learns which time segments matter for each musical aspect
- **Hierarchical processing**: Early layers = low-level features, later layers = high-level concepts
- **Context**: Unlike CNNs, transformers see the full temporal context

### MERT vs Alternatives
- **MusicGen**: Generative (creates music), not optimized for similarity
- **Jukebox**: Too large, slow inference
- **Wav2Vec2**: Speech-focused, misses music-specific patterns
- **MERT**: Best balance of performance, speed, and music understanding

### The Wav2Vec2FeatureExtractor Confusion

**Common Question**: "Why use Wav2Vec2FeatureExtractor if Wav2Vec2 is for speech?"

**Answer**: This is a HuggingFace naming quirk that confuses everyone.

`Wav2Vec2FeatureExtractor` is actually a **generic audio preprocessing class**, NOT specific to Wav2Vec2 speech models.

**What it actually does** (see [extractor.py:46-50](mess/extraction/extractor.py)):
```python
self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
    self.model_name,  # ← "m-a-p/MERT-v1-95M" (MERT, not Wav2Vec2!)
    trust_remote_code=True
)
```

**The processor just handles basic audio preprocessing:**
- Load audio file → numpy array
- Normalize amplitude values
- Return in format expected by model (tensors, proper shape)

**Why this naming?**
- HuggingFace reuses preprocessing code across models
- Wav2Vec2FeatureExtractor was written first for Wav2Vec2
- Later audio models (MERT, HuBERT, etc.) reuse the same preprocessor class
- Should have been called `AudioFeatureExtractor` but legacy naming stuck

**Key point**: The actual MERT model is loaded separately (line 52-56). The processor is just plumbing.

---

## 3. Layer Specialization: Our Key Discovery

**Central Question**: What does each MERT layer encode?

### The Problem with Naive Averaging
Most systems do this:
```python
embedding = mert_features.mean(dim=0)  # Average all 13 layers
```

**Problem**: This loses layer-specific information! Layer 0 might encode brightness while Layer 12 encodes tempo. Averaging destroys specialization.

### Our Approach: Linear Probing

**Linear probing** = Train a simple linear model on frozen embeddings to test what information is encoded.

#### Process
1. Extract MERT features (13 layers × 768 dims) for each track
2. Generate ground truth musical descriptors:
   - Spectral centroid (brightness measure from audio)
   - Tempo (from onset detection)
   - Spectral rolloff (timbre measure)
   - Zero-crossing rate (texture)
3. For each layer, train: `Ridge(layer_features) → musical_descriptor`
4. Evaluate with 5-fold cross-validation → R² score

#### Results (from layer_discovery_results.json)
```
Layer 0 → Spectral Centroid:  R² = 0.944  ← EXCELLENT!
Layer 1 → Timbral Texture:    R² = 0.922
Layer 2 → Acoustic Structure: R² = 0.933
Layer 7 → Phrasing:           R² = 0.781  ← Promising
Layer 4 → Temporal Patterns:  R² = 0.673
```

**Interpretation**:
- Layer 0 explains 94.4% of variance in spectral centroid
- This is near-perfect correlation!
- Validates that Layer 0 specializes in timbral brightness

### Why This Matters for Similarity Search

Instead of:
```python
# Bad: Average all layers
avg_embedding = features.mean(axis=0)
similarity = cosine(avg_embedding_A, avg_embedding_B)
```

We do:
```python
# Good: Use validated layer for specific aspect
brightness_layer = features[0]  # R² = 0.944 for brightness
similarity = cosine(brightness_layer_A, brightness_layer_B)
```

**Result**: More accurate similarity for the musical aspect you care about.

---

## 4. Similarity Metrics: Cosine vs Euclidean

### Cosine Similarity
```
cos_sim(A, B) = (A · B) / (||A|| × ||B||)
```

- **Range**: -1 to 1 (for embeddings, typically 0 to 1)
- **Measures**: Angle between vectors (direction, not magnitude)
- **Best for**: Normalized embeddings where scale doesn't matter

**Why it works for embeddings**:
- Embeddings from neural networks are often normalized
- Magnitude is arbitrary (depends on training), direction is meaningful
- Insensitive to length scaling

### Euclidean Distance
```
euclidean(A, B) = sqrt(Σ(A_i - B_i)²)
```

- **Range**: 0 to ∞
- **Measures**: Straight-line distance in vector space
- **Best for**: When magnitude matters (e.g., physical measurements)

**For MERT embeddings**: Cosine similarity is better because:
- MERT outputs are roughly normalized
- We care about "musical direction" not "feature magnitude"
- Empirically validated in our experiments

---

## 5. Multi-Aspect Recommendations: Combining Layers with Weights

### The Problem with Single-Aspect Search

Searching by ONE musical aspect gives narrow results:
- Layer 0 only → finds tracks with similar brightness
- Layer 1 only → finds tracks with similar texture
- Layer 2 only → finds tracks with similar structure

**But what if you want**: "Similar brightness AND texture, but brightness matters more"?

### Solution: Weighted Multi-Aspect Search

**Concept**: Combine multiple MERT layers with custom weights to match your preferences.

#### How It Works (from [layer_based_recommender.py:221-290](mess/search/layer_based_recommender.py))

**Step 1: Define aspect weights**
```python
aspect_weights = {
    'spectral_brightness': 0.6,  # 60% importance
    'timbral_texture': 0.4,      # 40% importance
}
```

**Step 2: Get similarity scores for EACH aspect separately**
```python
# For each aspect, compute similarity using its validated layer
brightness_similarities = {  # Uses Layer 0
    'Track_A': 0.92,
    'Track_B': 0.78,
    'Track_C': 0.65
}

texture_similarities = {  # Uses Layer 1
    'Track_A': 0.85,
    'Track_B': 0.91,
    'Track_C': 0.72
}
```

**Step 3: Combine with weighted average**
```python
# For Track_A:
combined_score = (0.6 × 0.92) + (0.4 × 0.85) = 0.552 + 0.340 = 0.892

# For Track_B:
combined_score = (0.6 × 0.78) + (0.4 × 0.91) = 0.468 + 0.364 = 0.832

# Track_A wins! (0.892 > 0.832)
```

**Step 4: Rank tracks by combined score**

Result: Tracks that match your weighted preference across multiple musical aspects.

### Why This Works Better Than Averaging Layers

**Naive approach** (don't do this):
```python
# Average all layers together
avg_embedding = mert_features.mean(axis=0)  # [768]
```
**Problem**: Destroys layer specialization. You're mixing brightness (Layer 0) with structure (Layer 2) before knowing what you care about.

**Multi-aspect approach** (what we do):
```python
# Keep layers separate, weight them by validated aspect importance
brightness_sim = cosine(query_layer0, candidate_layer0)  # R²=0.944 for brightness
texture_sim = cosine(query_layer1, candidate_layer1)    # R²=0.922 for texture

# Combine AFTER computing similarities
final_score = weight_brightness * brightness_sim + weight_texture * texture_sim
```
**Advantage**: You control which musical aspects matter and by how much.

### Real-World Example

**Query**: "Find tracks like Beethoven Op.27 No.1, prioritizing timbral similarity over brightness"

```python
recommender.multi_aspect_recommendation(
    query_track="Beethoven_Op027No1-01_003_20090916-SMD",
    aspect_weights={
        'spectral_brightness': 0.3,  # Some importance
        'timbral_texture': 0.7,      # High importance
    },
    n_recommendations=5
)
```

**Results breakdown**:
```
1. Beethoven_Op027No2-01 (combined: 0.921)
   spectral_brightness: 0.893
   timbral_texture: 0.935  ← High texture match drives the score!

2. Mozart_KV331-01 (combined: 0.856)
   spectral_brightness: 0.912  ← Higher brightness
   timbral_texture: 0.831      ← Lower texture → lower combined score
```

Even though Mozart has higher brightness similarity, Beethoven Op.27 No.2 wins because texture is weighted higher.

### When to Use Each Approach

| Use Case | Method | Example |
|----------|--------|---------|
| **One clear aspect** | `recommend_by_layer(layer=0)` | "Find bright piano pieces" |
| **One validated aspect** | `recommend_by_aspect('spectral_brightness')` | "Find similar timbre" |
| **Multiple aspects, equal weight** | `multi_aspect_recommendation({...}, all weights = 1.0)` | "Match brightness AND texture equally" |
| **Multiple aspects, custom priority** | `multi_aspect_recommendation({...}, different weights)` | "Mostly texture, some brightness" |

### The Math: Weighted Average

For track T with N aspects:

```
combined_score(T) = Σ(weight_i × similarity_i) / Σ(weight_i)
                    i=1 to N
```

Where:
- `weight_i` = importance of aspect i (e.g., 0.6 for brightness)
- `similarity_i` = cosine similarity for aspect i (0 to 1)
- Division by sum of weights normalizes the result

**Properties**:
- Higher weights → aspect contributes more to final score
- Weights sum to 1.0 (normalized) → combined score stays in [0, 1] range
- If all weights equal → simple average

### Advanced: Layer Weight Calibration

Notice in [layer_based_recommender.py:44-65](mess/search/layer_based_recommender.py):

```python
0: {
    'aspect': 'spectral_brightness',
    'r2_score': 0.944,
    'weight': 1.0  # ← Highest R², full confidence
},
1: {
    'aspect': 'timbral_texture',
    'r2_score': 0.922,
    'weight': 0.95  # ← Slightly lower confidence
},
```

**These weights encode validation confidence**: Layers with higher R² scores get higher default weights.

You can override these with your own aspect weights based on what YOU care about!

---

## 6. Why Not Just Average All Layers?

**Common question**: "Why bother with layer selection and weights? Why not just average all 13 layers?"

### The Averaging Trap

Many systems do this:
```python
# Naive approach
averaged_embedding = mert_features.mean(axis=0)  # [13, 768] → [768]
similarity = cosine(avg_query, avg_candidate)
```

**Problems**:

1. **Loss of Specialization**
   - Layer 0 encodes brightness (R²=0.944)
   - Layer 12 encodes... something else (R²=0.201 for brightness)
   - Averaging them: (0.944 + 0.201) / 2 = worse than just using Layer 0!

2. **Noise Dilution**
   - If you care about brightness, Layer 12 adds noise
   - Signal-to-noise ratio decreases

3. **No User Control**
   - Can't prioritize aspects that matter to you
   - One-size-fits-all approach

### Our Validated Approach

```python
# Use validated layer for specific aspect
if user_wants_brightness:
    layer = 0  # R²=0.944
elif user_wants_texture:
    layer = 1  # R²=0.922
elif user_wants_structure:
    layer = 2  # R²=0.933

similarity = cosine(query[layer], candidate[layer])
```

**Advantages**:
- Higher signal for the aspect you care about
- Backed by empirical validation (R² scores)
- Interpretable results ("similar in brightness")

### When Averaging IS Good: Within-Aspect Combination

Averaging layers **that encode the same aspect** can help:

```python
# Multiple layers encode brightness (hypothetically)
brightness_layers = [0, 3, 5]  # All validated for brightness
brightness_embedding = mert_features[brightness_layers].mean(axis=0)
```

This averages **redundant signals**, reducing noise. But our research shows most layers specialize differently, so this doesn't apply here.

### Empirical Validation

We tested this! See [layer_discovery_results.json](mess/probing/layer_discovery_results.json):

| Method | R² for Brightness |
|--------|-------------------|
| Layer 0 alone | 0.944 |
| Layer 1 alone | 0.701 |
| Layer 12 alone | 0.201 |
| Average all 13 | ~0.65 (worse than Layer 0!) |

**Conclusion**: Smart layer selection beats naive averaging.

---

## 7. FAISS: Fast Similarity Search

**Problem**: Computing cosine similarity for 10K tracks = 10K vector operations ≈ 100ms

**Solution**: FAISS (Facebook AI Similarity Search)

### How FAISS Works
1. **Pre-compute index**: Organize embeddings for fast lookup
2. **Optimized search**: Use SIMD, GPU, and approximate algorithms
3. **Result**: <1ms searches instead of 100ms

### Index Types in MESS-AI

#### IndexFlatIP (Inner Product)
- **What**: Exact brute-force search
- **Speed**: Fast for <100K vectors
- **Accuracy**: 100% (no approximation)
- **Usage**: `IndexFlatIP(768)` for cosine similarity (after L2 normalization)

**Why IP = Cosine?**
```
If vectors are L2-normalized (||A|| = ||B|| = 1):
  cos_sim(A, B) = (A · B) / (||A|| × ||B||)
                = (A · B) / (1 × 1)
                = A · B  ← This is inner product!
```

### When to Use Advanced Indices
- **IVF (Inverted File)**: For >100K tracks, trades accuracy for speed
- **HNSW**: Graph-based, best for very large datasets (millions)
- **Current MESS-AI**: IndexFlatIP is perfect for our ~50-1000 track datasets

---

## 8. Audio Signal Processing Basics

### Waveform → Features Pipeline

#### Step 1: Audio Loading
```python
waveform, sr = torchaudio.load("track.wav")
# waveform: Raw amplitude values [-1, 1]
# sr: Sample rate (e.g., 44100 Hz = 44,100 samples per second)
```

#### Step 2: Preprocessing
```python
# Mono conversion (stereo → single channel)
waveform = waveform.mean(dim=0)

# Resampling (44.1kHz → 24kHz for MERT)
waveform = torchaudio.transforms.Resample(orig_freq=44100, new_freq=24000)(waveform)
```

**Why 24kHz?**: MERT was pre-trained on 24kHz audio. Higher rates waste computation, lower rates lose frequency information (Nyquist limit).

#### Step 3: Segmentation
```python
# Cut into 5-second chunks
segment_length = 24000 * 5  # 120,000 samples
segments = [waveform[i:i+segment_length] for i in range(0, len(waveform), segment_length)]
```

**Why segment?**:
- Transformers have fixed context windows
- 5 seconds captures local musical patterns
- Longer segments → more memory, diminishing returns

#### Step 4: Feature Extraction
```python
# MERT processes segments → outputs [13 layers, time, 768 dims]
features = mert_model(segment)
```

### Key Audio Concepts Used in Probing

**Spectral Centroid**: "Brightness" of sound
- Formula: Weighted mean of frequencies in spectrum
- High centroid → bright, sharp timbre (flute, cymbals)
- Low centroid → dark, warm timbre (cello, bass drum)

**Zero-Crossing Rate**: Texture/noisiness
- Counts how often waveform crosses zero
- High ZCR → noisy, percussive (drums, consonants)
- Low ZCR → tonal, sustained (vowels, strings)

**Spectral Rolloff**: Frequency cutoff containing 85% of energy
- Higher rolloff → more high-frequency content
- Lower rolloff → bass-heavy sound

**Tempo**: Beats per minute
- Detected via onset detection (find note attacks)
- Used to validate temporal layers in MERT

---

## 9. The MESS-AI Pipeline: End-to-End

### Data Flow
```
1. Audio Files (.wav)
   ├─ Load at original sample rate
   ├─ Resample to 24kHz
   └─ Segment into 5s chunks
           ↓
2. MERT Model
   ├─ Process each segment
   ├─ Extract 13 layers × time × 768 dims
   └─ Output: Raw features
           ↓
3. Feature Aggregation
   ├─ Time-average within segments → [13, 768]
   ├─ Save three versions:
   │   - raw: Full temporal [segments, 13, time, 768]
   │   - segments: Time-averaged [segments, 13, 768]
   │   - aggregated: Track-level [13, 768]
   └─ Used for similarity: aggregated
           ↓
4. Layer Discovery (Validation Phase)
   ├─ Generate proxy targets (spectral centroid, tempo, etc.)
   ├─ For each layer: Train Ridge(layer) → proxy_target
   ├─ Compute R² scores → Find specializations
   └─ Save results: layer_discovery_results.json
           ↓
5. Similarity Search
   ├─ Load aggregated features [13, 768]
   ├─ Select layer based on desired aspect
   │   - Brightness → Layer 0
   │   - Texture → Layer 1
   │   - Structure → Layer 2
   ├─ Build FAISS index (IndexFlatIP)
   └─ Query: cosine_similarity(reference, all_tracks)
           ↓
6. Recommendations
   └─ Return top-K most similar tracks
```

### Why This Architecture?

**Separation of Concerns**:
- `extraction/`: Audio → embeddings (ML engineering)
- `probing/`: Validate what layers encode (ML research)
- `search/`: Apply validated knowledge & fast retrieval (application + optimization)
- `datasets/`: Dataset loaders and management

**Reproducibility**:
- Save features once, experiment many times
- Validation results in JSON (auditable)
- Clear separation: data generation vs. experiments

**Flexibility**:
- Want different similarity? Use different layer
- Want multi-aspect? Combine layers with weights
- Want faster search? Swap FAISS index type

---

## 10. Key Research Questions (Ongoing)

### Answered
✅ Which MERT layers encode which musical aspects?
✅ Does layer specialization improve similarity vs averaging?
✅ Is cosine similarity better than Euclidean for MERT embeddings?

### In Progress
🔬 Does fine-tuning MERT on specific genres improve similarity?
🔬 Can we learn aspect weights automatically from user feedback?
🔬 Does mean-centering (subtracting classical mean) improve retrieval?

### Future
📋 Multi-modal: Combine audio + score + metadata embeddings?
📋 User preference learning: Personalized similarity functions?
📋 Domain adaptation: Transfer to non-classical music?

---

## 11. Common Misconceptions

### "Embeddings are just feature engineering"
**No**: Embeddings are *learned* representations. Traditional features (MFCCs, spectral centroid) are hand-crafted. Embeddings discover patterns automatically.

### "Bigger embeddings are always better"
**No**: 768 dims is a sweet spot. More dims = more computation, overfitting risk. Diminishing returns beyond ~1024 for most tasks.

### "All MERT layers are equally good for similarity"
**No**: Our research shows layer specialization. Layer 0 for brightness (R²=0.944) >> Layer 12 for brightness (R²=0.201).

### "FAISS is approximate, so it's inaccurate"
**No**: IndexFlatIP (what we use) is *exact*. Only advanced indices (IVF, PQ) trade accuracy for speed. We chose exact search for research.

### "More training data always helps"
**Context-dependent**: MERT was pre-trained on 160K hours. Fine-tuning helps for domain-specific patterns, but gains diminish. Our 50-track dataset validates, but won't improve MERT's base understanding.

### "Combining layers with weights is the same as averaging"
**No**: There's a critical difference:

**Naive averaging** (bad):
```python
avg = layers.mean(axis=0)  # Mix before comparing
similarity = cosine(avg_query, avg_candidate)
```

**Weighted combination** (good):
```python
# Compare within each layer FIRST
sim_layer0 = cosine(query[0], candidate[0])
sim_layer1 = cosine(query[1], candidate[1])

# THEN combine similarities with weights
final = 0.6 * sim_layer0 + 0.4 * sim_layer1
```

The key: **Compare first, combine later** preserves layer specialization.

---

## 12. Further Reading

### Papers
- **MERT**: https://arxiv.org/abs/2306.00107
- **Linear Probing**: "BERT Rediscovers the Classical NLP Pipeline" (similar methodology)
- **Music Information Retrieval**: Müller, "Fundamentals of Music Processing"

### Tools
- **FAISS**: https://github.com/facebookresearch/faiss
- **librosa**: https://librosa.org/doc/latest/index.html
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers

### Concepts
- **Cosine Similarity**: https://en.wikipedia.org/wiki/Cosine_similarity
- **Transfer Learning**: Sebastian Ruder's blog
- **Self-Supervised Learning**: Yann LeCun's lectures
