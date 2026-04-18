# Context Index (Pointer Router)

This file is the routing table for optional markdown context.

Precedence:
1. `AGENTS.md`
2. Code in `mess/` and tests in `tests/`
3. Pointer docs listed below

Load policy:
- Load only the minimum docs needed for the task.
- Do not bulk-read `docs/`.
- If a pointer doc conflicts with code/tests, code/tests win.

## Pointer Table

| File | Use When | Skip When | Priority |
| --- | --- | --- | --- |
| `docs/product_narrative.md` | You are making product-facing tradeoffs, retrieval UX decisions, or evaluating north-star alignment. | You are implementing backend mechanics without product decision impact. | Medium |
| `docs/research.md` | You want historical context on the archived learned-projection direction. | You are doing current-roadmap work. | Low (archived) |

## Recommended Load Sequences

### Sequence A: Feature/Extraction/Search bugfix
1. `AGENTS.md`
2. Relevant code/tests

### Sequence B: Product-direction decision
1. `AGENTS.md`
2. `docs/product_narrative.md`
3. Relevant code/tests
