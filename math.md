# Mathematical Pipeline: From Raw Audio to Music Recommendations

This document describes the complete mathematical pipeline in the mess-ai system, from raw WAV files through feature extraction, model processing, embedding generation, similarity search, and finally to track recommendations.

## 1. Raw Audio Input and Preprocessing

### 1.1 Audio Loading and Representation
- **Input**: WAV files at 44.1kHz sampling rate
- **Mathematical representation**: Discrete-time signal x[n] where n ∈ {0, 1, ..., N-1}
- **Stereo to Mono conversion**: 
  ```
  x_mono[n] = (x_left[n] + x_right[n]) / 2
  ```

### 1.2 Resampling (44.1kHz → 24kHz for MERT)
- **Sinc interpolation**: Using torchaudio's resampling based on band-limited interpolation
- **Mathematical formulation**:
  ```
  x_resampled[m] = Σ_n x[n] · sinc(π(m·f_old/f_new - n))
  ```
  where f_old = 44100 Hz, f_new = 24000 Hz

### 1.3 Segmentation with Overlap
- **Segment duration**: T_seg = 5.0 seconds
- **Overlap ratio**: r_overlap = 0.5 (50%)
- **Segment length in samples**: L_seg = T_seg × f_s = 5.0 × 24000 = 120,000 samples
- **Hop size**: H = L_seg × (1 - r_overlap) = 60,000 samples
- **Number of segments**: 
  ```
  N_segments = floor((N - L_seg) / H) + 1
  ```

## 2. MERT Feature Extraction

### 2.1 Wav2Vec2 Feature Extraction (Preprocessing)
- **Input normalization**: Zero-mean, unit variance
  ```
  x_norm = (x - μ) / σ
  ```
- **Feature extraction**: 1D CNN layers extract local features
  ```
  F_0 = Conv1D(x_norm, kernel_size=10, stride=5)
  ```

### 2.2 MERT Transformer Architecture
- **Model**: MERT-v1-95M (95 million parameters)
- **Architecture**: 12 transformer layers + 1 embedding layer = 13 total layers
- **Hidden dimension**: d_model = 768
- **Attention mechanism** (for each layer l):
  ```
  Attention(Q, K, V) = softmax(QK^T / √d_k) · V
  ```
  where Q, K, V are query, key, value matrices

### 2.3 Multi-Scale Feature Extraction
For each audio segment s_i:
- **Raw features**: H_raw[i] ∈ ℝ^(13 × T × 768)
  - 13 layers (12 transformer + 1 embedding)
  - T time steps (depends on segment length)
  - 768-dimensional features per timestep

- **Time-averaged segment features**: 
  ```
  H_seg[i] = (1/T) Σ_{t=1}^T H_raw[i, :, t, :] ∈ ℝ^(13 × 768)
  ```

- **Aggregated track features**:
  ```
  H_track = (1/N_segments) Σ_{i=1}^{N_segments} H_seg[i] ∈ ℝ^(13 × 768)
  ```

## 3. Feature Preparation for Similarity Search

### 3.1 Feature Flattening
- **Input**: H_track ∈ ℝ^(13 × 768)
- **Output**: v ∈ ℝ^9984 (13 × 768 = 9984 dimensions)
- **Operation**: Row-major flattening
  ```
  v[k] = H_track[i, j] where k = i × 768 + j
  ```

### 3.2 L2 Normalization
- **Purpose**: Convert to unit vectors for cosine similarity
- **Mathematical operation**:
  ```
  v_norm = v / ||v||_2 = v / √(Σ_{i=1}^{9984} v_i^2)
  ```
- **Special case handling**: If ||v||_2 = 0, then v_norm = v

## 4. FAISS Indexing and Search

### 4.1 Index Construction (IndexFlatIP)
- **Index type**: Inner Product index (equivalent to cosine similarity for normalized vectors)
- **Storage**: Direct storage of all normalized vectors
- **Memory complexity**: O(N × d) where N = number of tracks, d = 9984

### 4.2 Similarity Computation
- **Query vector**: q ∈ ℝ^9984 (L2-normalized)
- **Database vectors**: {v_1, v_2, ..., v_N}, each v_i ∈ ℝ^9984 (L2-normalized)
- **Similarity score**:
  ```
  sim(q, v_i) = q · v_i = Σ_{j=1}^{9984} q_j × v_{i,j}
  ```
- **Note**: For L2-normalized vectors, inner product = cosine similarity:
  ```
  cos(θ) = (q · v_i) / (||q||_2 × ||v_i||_2) = q · v_i  (since ||q||_2 = ||v_i||_2 = 1)
  ```

### 4.3 k-Nearest Neighbor Search
- **Objective**: Find k vectors with highest similarity scores
- **Algorithm**: Exhaustive search (for IndexFlatIP)
  ```
  top_k = argmax_k {sim(q, v_i) | i = 1, ..., N}
  ```
- **Complexity**: O(N × d) per query
- **Optimization**: SIMD instructions for vector operations

## 5. Recommendation Generation

### 5.1 Similarity-Based Ranking
Given reference track r with feature vector v_r:
1. Compute similarities: S = {sim(v_r, v_i) | i ≠ r}
2. Sort in descending order: S_sorted
3. Return top-k tracks with scores

### 5.2 Score Interpretation
- **Range**: [-1, 1] (theoretical), [0, 1] (practical for music)
- **Interpretation**:
  - sim ≈ 1.0: Nearly identical musical characteristics
  - sim ≈ 0.7-0.9: Very similar (same genre, style)
  - sim ≈ 0.5-0.7: Moderately similar
  - sim < 0.5: Less similar

## 6. Mathematical Properties and Guarantees

### 6.1 Metric Properties
- **Cosine similarity** is not a proper metric (violates triangle inequality)
- **Angular distance**: d_angular(u, v) = arccos(u · v) / π is a proper metric
- **Euclidean distance** for normalized vectors:
  ```
  ||u - v||_2^2 = 2(1 - u · v)
  ```

### 6.2 Dimensionality and Information Theory
- **Original audio**: ~2.2M samples (50 seconds at 44.1kHz)
- **MERT embeddings**: 9,984 dimensions
- **Compression ratio**: ~220:1
- **Information preservation**: Captures musical semantics, not signal reconstruction

### 6.3 Statistical Properties
- **Feature distribution**: Approximately Gaussian after layer normalization
- **Similarity distribution**: Typically concentrated around 0.6-0.8 for music
- **Curse of dimensionality**: In 9984-D space, most vectors are nearly orthogonal
  - Expected similarity between random vectors: E[sim] ≈ 0 ± O(1/√d)

## 7. Performance Characteristics

### 7.1 Time Complexity
- **Feature extraction**: O(T × d^2) per track (transformer forward pass)
- **Index building**: O(N × d) 
- **Similarity search**: O(N × d) per query (brute force)
- **With FAISS optimizations**: ~50-100x speedup via SIMD/cache optimization

### 7.2 Space Complexity
- **Raw features per track**: ~94MB (all scales)
- **Aggregated features per track**: ~39KB (9984 × 4 bytes)
- **FAISS index for 50 tracks**: ~2MB
- **Total system**: O(N) space for N tracks

## 8. Future Mathematical Enhancements

### 8.1 Advanced Indexing
- **IVF (Inverted File)**: Approximate search with O(√N × d) complexity
- **HNSW (Hierarchical Navigable Small World)**: O(log N × d) search
- **LSH (Locality Sensitive Hashing)**: Probabilistic O(1) search

### 8.2 Metric Learning
- **Supervised fine-tuning**: Learn projection W ∈ ℝ^(d' × d) where d' < d
- **Contrastive learning**: Optimize embedding space for musical similarity
- **Triplet loss**:
  ```
  L = max(0, sim(a, n) - sim(a, p) + margin)
  ```

### 8.3 Multi-Modal Integration
- **Audio + Metadata fusion**: Weighted combination of similarities
- **Temporal modeling**: RNN/Transformer over segment sequences
- **Cross-attention**: Between different feature scales

## Conclusion

The mess-ai pipeline implements a sophisticated mathematical framework that transforms raw audio signals into high-dimensional semantic representations, enabling efficient similarity search and music recommendation. The system leverages modern deep learning (MERT transformer), efficient indexing (FAISS), and geometric principles (cosine similarity in high-dimensional spaces) to deliver sub-millisecond recommendations while maintaining musical relevance.