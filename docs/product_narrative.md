# Product Narrative

This file contains product-facing narrative context.
Execution policy and engineering contract live in `AGENTS.md`.

## North-Star Goal

Build an audio-first system for expressive, content-based similarity search over MERT-derived clip embeddings.

Capability target:
- A user selects a short gesture (around 5 seconds).
- The system retrieves matching passages by expressive character.
- Optional aspect controls adjust intent (for example rubato, dynamics, articulation, harmony).

Natural-language querying is a later interface layer, not a prerequisite for core retrieval quality.

## Query Archetypes

1. Segment-level expressive retrieval
- "Find performances that feel like this 5-second cadenza."
- "Find a similar build-up but more restrained."

2. Cross-performance comparison
- "How does performer A phrase this passage vs performer B?"
- "Who stretches this cadence more?"

3. Aspect-constrained retrieval
- "Similar phrasing but not tempo."
- "Similar rubato profile, ignore absolute tempo."
- "Same dynamic contour, different harmonic density."

4. Structure-aware retrieval
- "Find climaxes similar to this one."
- "Find cadential resolutions with similar behavior."

5. Future text-guided retrieval
- "Melancholic but hopeful."
- "Heroic but restrained."

## Product Direction Guardrails

- Prioritize expressive user-intent retrieval quality.
- Keep the core system audio-first and embedding-first.
- Preserve reproducibility and modularity in `mess/`.
- Avoid over-coupling to metadata-only assumptions.

