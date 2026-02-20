# MESS Core Concepts (Implementation-Accurate)

This document describes the concepts MESS uses today, based on the current code in `mess/`.

Current productionized core is still track-level FAISS retrieval, with aspect-driven layer routing.

## 2) Representation Contracts

MERT feature views in MESS:
- `raw`: `[num_segments, 13, time_steps, 768]`
- `segments`: `[num_segments, 13, 768]`
- `aggregated`: `[13, 768]`

Current extraction writes these in `data/embeddings/<dataset>-emb/{raw,segments,aggregated}/`.

## 3) Why Layers Matter

MERT has 13 hidden-state layers and each layer encodes different information.
MESS validates layer usefulness with linear probing in `mess/probing/discovery.py`.

Key idea:
- do not blindly average layers before deciding intent,
- choose layer(s) based on validated targets/aspects.

## 4) Probing and Aspect Resolution

`LayerDiscoverySystem`:
- probes each layer against scalar proxy targets,
- reports `r2_score`, correlation, RMSE,
- saves results to JSON.

`ASPECT_REGISTRY` + `resolve_aspects()`:
- map user-facing aspects (for example `brightness`, `phrasing`) to validated layers,
- filter by minimum R2 (`min_r2`),
- attach confidence labels.

This is the bridge from research metrics to searchable controls.

## 5) Search Stack Today

Search is implemented in `mess/search/search.py`.

### Baseline search

`load_features(features_dir, layer=None)`:
- loads `.npy` files,
- converts each file to exactly one indexed vector,
- enforces deterministic `1 row == 1 track`.

Supported input shapes:
- `[13, 768]`
- `[segments, 13, 768]` (pooled)
- `[segments, 13, time, 768]` (pooled)
- `[feature_dim]` (already vectorized)

`build_index(features)`:
- L2-normalizes vectors,
- builds `faiss.IndexFlatIP` (exact cosine via inner product on normalized vectors).

`find_similar(query_track, features, track_names, k, exclude_self)`:
- nearest-neighbor retrieval by cosine score.

### Single-aspect search

`search_by_aspect(query_track, aspect, features_dir, k)`:
- resolves one aspect to one validated layer,
- loads layer vectors,
- runs cosine retrieval.

### Multi-aspect weighted search

`search_by_aspects(query_track, aspect_weights, features_dir, k, min_r2, scale_by_r2)`:
- resolves multiple aspects to layers,
- builds a combined layer-weight vector,
- fuses per-track layer embeddings into one vector,
- retrieves with FAISS.

Notes:
- aspect weights are query-time controls,
- optional `scale_by_r2=True` lets validated aspects contribute proportionally to confidence.

## 6) Current Capability vs North Star

Implemented:
- track-level similarity retrieval,
- probing-validated aspect routing,
- weighted multi-aspect track retrieval.

Not implemented yet:
- clip-level index and recommendation,
- timestamp-aware retrieval outputs,
- natural language query parser,
- learned reranker and personalization.

## 7) Practical Query Patterns (Current)

Single intent:
- "similar brightness" -> `search_by_aspect(..., aspect="brightness")`

Multi intent:
- "mostly brightness, some phrasing" ->
  `search_by_aspects(..., {"brightness": 0.7, "phrasing": 0.3})`

CLI:
```bash
python scripts/demo_recommendations.py --track "<TRACK_ID>"
python scripts/demo_recommendations.py --track "<TRACK_ID>" --aspect brightness --k 10
python scripts/demo_recommendations.py --track "<TRACK_ID>" --aspects "brightness=0.7,phrasing=0.3" --k 10
```

## 8) Common Pitfalls

1. Treating probing R2 as equivalent to retrieval quality.
- High R2 is a good signal, not a ranking guarantee.

2. Mixing all layers indiscriminately.
- This can blur aspect-specific signal.

3. Using outdated module paths from old docs.
- Active search module is `mess/search/search.py`.
- Aspect registry/resolution lives in `mess/probing/discovery.py`.

## 9) What To Build Next

If the goal is expressive 5-second recommendations, the next milestone is clip indexing:
- index unit: clip (not full track),
- schema: `track_id`, `clip_idx`, `start_sec`, `end_sec`, vector,
- retrieval output includes timestamp spans.

That unlocks the core product primitive:
"I like this 5-second moment, find more like it."
