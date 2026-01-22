The Tight, Evidence-Based Plan
Based on GPT's guidance and your actual codebase, here's what to do:
Phase 1: Make Retrieval Controllable & Measurable (1 day)
Task 1.1: Build per-layer indices (2-3 hours)
# For layers: 0, 1, 2, 7, 8, 9, 12
for layer in [0, 1, 2, 7, 8, 9, 12]:
    embeddings = load_all_embeddings(layer)  # [n_tracks, 768]
    embeddings = normalize(embeddings)  # L2-normalize
    index = faiss.IndexFlatIP(768)
    index.add(embeddings)
    faiss.write_index(index, f"layer_{layer}.index")
Evaluation: Self-retrieval sanity check (does each track return itself at rank 1?)
Task 1.2: Add classical mean-centering (1-2 hours)
for layer in [0, 1, 2, 7, 8, 9, 12]:
    embeddings = load_all_embeddings(layer)
    mu = np.mean(embeddings, axis=0)  # [768,]
    centered = embeddings - mu
    centered = normalize(centered)
    # ... index and save
Evaluation: Check similarity distributions (should spread from 0.92-0.99 to 0.6-0.95)
Task 1.3: Distribution analysis script (1 hour)
# For each layer index:
# - Sample 50 random queries
# - Get top-10 similarities
# - Plot histogram
# Compare: raw vs centered, layer 0 vs 7 vs 12
Expected outcome: Centered embeddings show wider similarity spread → better ranking resolution
Phase 2: Same-Work Retrieval Evaluation (1-2 days)
Task 2.1: Build MAESTRO same-work test set (3-4 hours)
Parse MAESTRO CSV metadata
Identify works with multiple performances (e.g., "Beethoven Sonata Op. 27 No. 2")
Create test set: 50-100 query-positive-negative triples
Save as JSON: {query_track, positive_tracks[], negative_tracks[]}
Task 2.2: Implement Recall@k and MAP metrics (2 hours)
def evaluate_retrieval(test_set, index, k=10):
    recalls = []
    aps = []
    for query, positives, negatives in test_set:
        results = index.search(query_embedding, k)
        # Calculate Recall@k
        retrieved_positives = set(results) & set(positives)
        recall = len(retrieved_positives) / len(positives)
        recalls.append(recall)
        # Calculate Average Precision
        # ... (standard IR metric)
    return np.mean(recalls), np.mean(aps)
Task 2.3: Layer comparison (2-3 hours)
Run same-work eval on layers: 0, 1, 2, 7, 8, 9, 12
Compare: raw vs centered
Find the layer(s) that maximize Recall@10
Expected outcome: Discover which layer(s) actually capture work-specific expression vs just style/timbre This is the critical experiment - it tells you if layer-specific indexing helps retrieval or just changes it.
Phase 3: Segment-Level Indexing (2-4 days)
Task 3.1: Build segment-level indices (3-4 hours)
# Use best-performing layer(s) from Phase 2
segment_embeddings = []
segment_metadata = []

for track in tracks:
    segments = load_segments(track)  # [num_segments, 13, 768]
    for seg_idx, seg_emb in enumerate(segments[:, best_layer, :]):
        segment_embeddings.append(seg_emb)
        segment_metadata.append({
            'track_id': track.id,
            'start_time': seg_idx * 2.5,  # 2.5s overlap
            'segment_idx': seg_idx
        })

# Index all segments
index = faiss.IndexFlatIP(768)
index.add(normalize(segment_embeddings))
Task 3.2: Implement segment → track aggregation (2-3 hours)
Retrieval modes: max similarity, mean of top-5, vote counting
Evaluate on same-work test set
Compare: segment-level vs piece-level retrieval
Task 3.3: Phrase-level query demo (2-3 hours)
# Query with 10-30 second audio clip
# Extract multiple segments
# Retrieve per segment, aggregate by track
# Return tracks with most matching segments
Expected outcome: Segment retrieval should improve Recall@k for same-work retrieval (if expression lives in local moments)
Critical Questions to Answer (Research Objectives)
After these 3 phases, you should be able to answer:
Which layer(s) best capture work-specific similarity?
Is it layer 0, 7, 8, or 12?
Does centering help?
Does segment-level indexing improve same-work retrieval?
Quantitative: Recall@k comparison
Qualitative: Can you find "similar rubato" moments?
What's the right aggregation strategy for segment → track?
Max similarity? Mean of top-5? Vote counting?
Are you retrieving expression or just recording quality?
Cross-era test: Can 1950s recording match 2020s recording of same work?

hat NOT to Do (Per GPT's Guidance)
❌ Don't trust R² scores as gospel - they validate proxy targets, not retrieval
❌ Don't assume layer-specific indexing = expression - need same-work eval to prove it
❌ Don't do PCA removal yet - only after you have eval metrics showing it helps
❌ Don't fine-tune - extract more value from current embeddings first
❌ Don't build demos - focus on evaluation first, demos later
Recommended Execution Order
Week 1: Infrastructure + Evaluation
Days 1-2: Phase 1 (per-layer indices, centering, sanity checks)
Days 3-5: Phase 2 (same-work test set, Recall@k evaluation, layer comparison)
Week 2: Segment Retrieval
Days 1-3: Phase 3 (segment indexing, aggregation strategies)
Days 4-5: Analysis and documentation of results
Success criteria:
Quantitative: Layer X achieves Recall@10 > 0.7 on same-work retrieval
Qualitative: Segment retrieval finds "similar interpretive moments" across different performances
This plan is more conservative and evidence-based than my previous recommendations. GPT is right to push for evaluation-driven development rather than speculative architecture changes. Would you like me to proceed with Phase 1 (per-layer indices + centering + sanity checks)? This is the foundation for all subsequent experiments.