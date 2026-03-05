# Expressive Retrieval Plan (Offline-First)

## Goal
Build expressive similarity search for:
- clip -> clip retrieval
- clip -> track recommendation

using pretrained MERT segment embeddings plus discovery/proxy-target supervision.

This plan is intentionally offline-first. Serving latency concerns are handled in the EC2 consumer repo.

## Why This Plan
Cosine similarity on frozen embeddings is a baseline, not a validated definition of expressive similarity.
We need a learned scoring function grounded in:
- discovery layer/aspect evidence
- proxy targets
- evaluation gates that test expressive behavior directly

## Core Approach
Phase the work so we can validate value early:

1. Retrieval-Augmented Projection Retriever (PR #1 baseline)
- Frozen MERT clip vectors as input.
- Train a projection head with multi-positive InfoNCE.
- Positives/negatives are mined from a FAISS neighborhood graph.
- Use a momentum/EMA target encoder and periodic index refresh for stability.

2. Optional Oracle Scorer (later phase)
- Add only if projection retrieval improves recall but ranking quality still needs richer pairwise/listwise supervision.

## Inputs and Unit of Work
- Primary unit: `segments` embeddings (`[num_segments, 13, 768]`)
- Avoid `raw` for production retrieval training due to cost.
- Use discovery outputs to route/select layers by aspect where applicable.

## Required Artifacts

### 1) Clip Index + Splits
`clip_index/`
- clip metadata (`clip_id`, `recording_id`, `track_id`, `start_sec`, `end_sec`, split)
- leak-safe train/val/test split definitions (recording-level)
- dataset/version metadata

### 2) Projection Retriever Artifact
`projection_retriever/`
- projection head weights
- projection config (input layout, output dim, normalization)
- training metadata (seed, loss params, index refresh cadence)
- projected vectors or deterministic generation script
- ANN index artifact + manifest/checksums

### 3) Evaluation Reports
`evaluation/`
- baseline vs model comparison
- clip->clip and clip->track metrics
- aspect-faithfulness and counterfactual behavior checks

## Pipeline Steps

1. Build clip index and split artifacts
- Generate clip rows from segment embeddings.
- Assign splits by recording (never by clip).
- Freeze and version split/index files.

2. Establish baseline retrieval
- Evaluate cosine on frozen MERT clip vectors.
- Record Recall@K/MRR (clip->clip, clip->track).

3. Train retrieval-augmented projection head
- Frozen backbone input vectors.
- Multi-positive InfoNCE objective.
- Retrieval mining guardrails:
  - exclude self
  - exclude near-time same-track windows
  - optional cross-recording-positive requirement

4. Refresh retrieval index periodically
- Re-embed clips with EMA target encoder.
- Rebuild FAISS index every N steps.

5. Evaluate end-to-end
- Compare against cosine/discovery-weighted baselines.
- Report clip->clip and clip->track outcomes.

6. Export artifacts
- Save checkpoints, config, metrics, and FAISS artifacts with manifests.

## Evaluation Gates (Must Pass)

1. Ranking quality
- Recall@K, MRR, nDCG for clip->clip and clip->track.

2. Aspect faithfulness
- Scores must align with intended aspects (e.g., phrasing vs tempo constraints).

3. Counterfactual behavior
- Changing aspect weights should shift ranking in expected direction.

4. Human audit subset
- Small blind-judgment set to validate perceived expressive similarity.

## Baselines and Ablations
- Cosine on frozen embeddings
- Discovery-guided weighted cosine
- Projection retriever (no retrieval mining; random positives only)
- Projection retriever (retrieval mining + guardrails)
- Projection retriever with/without EMA target encoder
- Optional later: projection retriever distilled from oracle scorer

## Non-Goals (for this phase)
- No real-time serving optimization in this repo.
- No raw feature indexing for production.
- No mandatory reranking stage in final serving path (optional in consumer repo).

## Implementation Staging (when execution starts)
1. Add training package (`mess/training`) with config, mining, loss, index, trainer.
2. Add `scripts/train_retrieval_ssl.py` for reproducible runs.
3. Add tests for mining/loss/index/trainer smoke.
4. Add artifact packaging + publish hooks after baseline quality gate passes.

## Success Definition
We can demonstrate that learned retrieval/scoring improves expressive relevance over cosine baselines,
and we can export reproducible, versioned artifacts ready for downstream serving integration.
