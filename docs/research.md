# Research Context

This file is non-binding research context for experiment planning and prioritization.
Execution and contract policy live in `AGENTS.md`.

## Research Focus

MESS-AI is research-first infrastructure for expressive, content-based similarity over MERT-derived clip embeddings.

Primary representation baseline:
- 13 transformer layers
- 768 dims per layer
- track matrix view `[13, 768]`

Core workflow:
`Audio -> Features -> Targets -> Probing -> Aspect Mapping -> Retrieval -> Training -> Artifact`

## Ordered Roadmap

1. Clip index contract + leakage-safe splits
   - Enforce clip metadata contract (`clip_id`, `recording_id`, timestamps, split).
   - Preserve recording-level split boundaries.
2. MIDI-derived expression targets
   - Rubato, velocity dynamics, articulation for expression probing.
   - Treat expression as optional coverage where MIDI is missing.
3. Baseline quality gates
   - Retrieval Recall@K/MRR plus sanity checks before geometry changes.
4. Retrieval-augmented projection training
   - Frozen MERT vectors + projection head + EMA target encoder.
   - Retrieval-mined positives/negatives + periodic index refresh.
5. Progressive geometry learning
   - Diagonal weighting -> linear projection -> MLP projection.
   - Consider selective upper-layer unfreezing only after head-only saturation.
6. Two-stage reranking (conditional)
   - Only if recall is strong and ranking within top-K remains weak.

## Decision Rules

- If Recall@K is low, improve embedding geometry before reranking.
- If proxy targets disagree with human judgment, fix proxy quality first.
- If retrieved positives are noisy, tighten mining/filtering before model scaling.
- If linear projection saturates, try MLP before backbone unfreezing.
- Use two-stage retrieval only when first-stage recall ceiling is already high.

## Evaluation Emphasis

- Keep deterministic evaluation reports where possible.
- Track both retrieval quality and expressive behavior:
  - ranking quality (Recall@K, MRR, nDCG)
  - sanity checks
  - aspect faithfulness
  - counterfactual behavior
- Maintain gate-based pass/warn/fail decisions to prevent regression drift.

## Strategic Preference

- Prioritize embedding-first expressive retrieval over metadata-first approaches.
- Prioritize audio-to-audio expressive retrieval first.
- Treat text guidance as a future interface layer, not core dependency.
- Use MIDI as an anchor for disentangling content from expression.

