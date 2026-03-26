# PR/TASK HANDOFF
- Task/PR: Collapse packaging extras to `search` and `ml`, refresh lockfile, and cut a local tag
- Branch: `main`
- Status: `completed`
- Depends on: current worktree state

## Objective
- Keep the base install lightweight with only always-safe core dependencies.
- Preserve a narrow `search` extra for FAISS and S3 support.
- Move the remaining runtime ML stack under a single `ml` extra and remove stale install paths.

## Scope
- In:
  - `pyproject.toml` optional dependencies and `dev` dependency group
  - `uv.lock`
  - Install guidance in docs/CI that names supported extras
  - Packaging contract tests and runtime install hints
  - Local tag creation once validation passes
- Out:
  - Runtime behavior changes in extraction, probing, search, or training
  - Dependency version upgrades unrelated to the extra reshuffle
  - Publishing tags upstream

## Files To Inspect
- `pyproject.toml`
- `uv.lock`
- `README.md`
- `.github/workflows/ci.yml`
- `mess/probing/discovery.py`
- `tests/test_packaging_contracts.py`
- `tests/tests.md`
- `docs/pr_backlog.md`

## Contracts To Preserve
- Base install remains lightweight and usable for safe metadata/config usage.
- `search` contains only repo-required search dependencies (`faiss-cpu`, `boto3`).
- All probing, extraction, and training runtime dependencies install through `mess-ai[ml]`.
- CI and contributor install commands must reference only supported extras.

## Plan
1. Update this task record before editing.
2. Rewrite package extras and dev tooling to the `base` / `search` / `ml` model.
3. Update CI, docs, and runtime error hints to the supported install paths.
4. Refresh `uv.lock` and run focused packaging validation.
5. Create a local git tag for the resolved packaging state.

## Validation
- `uv lock`
- `uv run pytest -q tests/test_packaging_contracts.py`
- `uv run ruff check tests/test_packaging_contracts.py`

## Risks / Open Questions
- Local tag creation is in scope here. Publishing the tag upstream still requires `git push origin <tag>`.
- `uv.lock` already had local edits at task start; this change preserves the `nnaudio` removal while reshaping extras to `search` and `ml`.
