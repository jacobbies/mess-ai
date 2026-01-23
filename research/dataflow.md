Data Flow

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