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
| `docs/system_data_flow.md` | You need end-to-end architecture/data movement context. | You are fixing a narrow local bug in one module. | High for system-level tasks |
| `docs/research.md` | You are designing experiments, changing probing/training/evaluation criteria, or making roadmap choices. | You are doing pure implementation/mechanical refactors. | High for ML workflow tasks |
| `docs/product_narrative.md` | You are making product-facing tradeoffs, retrieval UX decisions, or evaluating north-star alignment. | You are implementing backend mechanics without product decision impact. | Medium |
| `docs/EXPRESSIVE_RETRIEVAL_PLAN.md` | You are working specifically on expressive retrieval evaluation/planning details. | Work is unrelated to expressive retrieval. | Medium |
| `docs/pr_backlog.md` | You are executing a non-trivial PR/debug task and need explicit handoff state. | Small local edits with no handoff requirement. | High when applicable |

## Recommended Load Sequences

### Sequence A: Feature/Extraction/Search bugfix
1. `AGENTS.md`
2. Relevant code/tests
3. `docs/system_data_flow.md` only if cross-module behavior is unclear

### Sequence B: Training/Probing experiment change
1. `AGENTS.md`
2. Relevant code/tests
3. `docs/research.md`
4. `docs/system_data_flow.md` if artifact/serving coupling matters

### Sequence C: Product-direction decision
1. `AGENTS.md`
2. `docs/product_narrative.md`
3. `docs/research.md`
4. Relevant code/tests
