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
Train two models:

1. Oracle Expressive Scorer (slow, high-quality, offline)
- Scores a query clip and candidate clip/track for expressive similarity.
- Used as quality reference and distillation teacher.

2. Projection Retriever (fast, scalable)
- Learns a retrieval embedding space for segment vectors.
- Trained using proxy/discovery constraints and distillation from the oracle scorer.

## Inputs and Unit of Work
- Primary unit: `segments` embeddings (`[num_segments, 13, 768]`)
- Avoid `raw` for production retrieval training due to cost.
- Use discovery outputs to route/select layers by aspect where applicable.

## Required Artifacts

### 1) Labels and Splits
`labels_and_splits/`
- pair/triple label dataset (query, candidate, relevance/targets)
- leak-safe train/val/test split definitions
- schema and dataset version metadata

### 2) Oracle Scorer Artifact
`oracle_scorer/`
- model weights
- model config (features, aspect controls, layer usage)
- training metadata (seed, loss weights, dataset version)
- eval report JSON (overall ranking + aspect-faithfulness)

### 3) Projection Retriever Artifact
`projection_retriever/`
- projection head weights
- projection config (input layout, output dim, normalization)
- projected vectors (or deterministic generation script)
- ANN index artifact + manifest/checksums

### 4) Evaluation Reports
`evaluation/`
- baseline vs model comparison
- clip->clip and clip->track metrics
- aspect-faithfulness and counterfactual behavior checks

## Pipeline Steps

1. Build training data
- Generate segment-level pairs/triples.
- Include positives/negatives and aspect-aware signals from proxy targets.
- Freeze and version split files.

2. Train oracle expressive scorer
- Train offline pairwise/listwise scorer.
- Include aspect-aware and proxy-aware objectives.

3. Validate oracle scorer
- Run ranking metrics and faithfulness checks.
- Reject model if it scores well overall but fails aspect behavior tests.

4. Train projection retriever
- Train retrieval embedding head with:
  - ranking/contrastive objective
  - proxy/discovery regularization
  - distillation target from oracle scorer

5. Build retrieval index
- Generate projected vectors for candidate corpus.
- Build and version index artifact.

6. Evaluate end-to-end
- Compare against cosine and discovery-guided baselines.
- Report clip->clip and clip->track outcomes.

7. Export artifacts
- Publish all versioned artifacts and manifests for EC2 consumer integration.

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
- Oracle scorer only
- Projection retriever only
- Projection retriever distilled from oracle

## Non-Goals (for this phase)
- No real-time serving optimization in this repo.
- No raw feature indexing for production.
- No mandatory reranking stage in final serving path (optional in consumer repo).

## Implementation Staging (when execution starts)
1. Add dataset builder + split tooling.
2. Add oracle scorer training/eval.
3. Add projection retriever training/eval.
4. Add artifact packaging and publishing hooks.

## Success Definition
We can demonstrate that learned retrieval/scoring improves expressive relevance over cosine baselines,
and we can export reproducible, versioned artifacts ready for downstream serving integration.
