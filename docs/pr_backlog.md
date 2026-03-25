# PR/TASK HANDOFF
- Task/PR: Cleanup clip-search README example and cut a stable downstream pin
- Branch: `pr16-clip-search-cleanup`
- Status: `in_progress`
- Depends on: `2f3c577` (`Merge pull request #19 from jacobbies/pr15-clip-search-artifact`)

## Objective
- Replace the stale pre-artifact `search_by_clip(...)` README example with the current artifact-backed call pattern.
- Close the open compatibility decision for downstream consumers such as `classical-recsys`.
- Cut a stable commit/tag once the docs and focused tests pass.

## Scope
- In:
  - `README.md` clip-search example
  - Compatibility-wrapper decision documentation
  - Focused regression coverage for the README example
  - Stable commit/tag creation for downstream pinning
- Out:
  - Search behavior changes
  - Public API renames or removals
  - Artifact schema or storage changes

## Files To Inspect
- `README.md`
- `docs/pr_backlog.md`
- `mess/search/search.py`
- `mess/search/__init__.py`
- `mess/__init__.py`
- `scripts/demo_recommendations.py`
- `tests/search/test_search.py`
- `tests/search/test_search_init_exports.py`
- `tests/test_public_api.py`

## Contracts To Preserve
- `search_by_clip` stays the stable public clip-search entry point.
- Clip search continues to operate on prebuilt clip artifacts rather than raw segment directories.
- Cleanup must not change any caller-visible behavior beyond fixing the stale README example.

## Plan
1. Update the task record before editing.
2. Replace the stale README snippet with the current artifact-backed example.
3. Add a focused regression test for the README example.
4. Run focused validation for the touched surface.
5. Create a stable commit and tag for downstream pinning.

## Validation
- `uv run pytest -q tests/test_readme_examples.py tests/search/test_search.py tests/search/test_search_init_exports.py tests/test_public_api.py`
- `uv run ruff check tests/test_readme_examples.py`

## Risks / Open Questions
- Decision: keep `search_by_clip` as the stable compatibility wrapper/public entry point for now. It is exported from both `mess` and `mess.search`, used by repo callers, and removing or renaming it would create downstream churn without solving a cleanup problem.
- Local tag creation is in scope here. Publishing that tag upstream still requires a later `git push origin <tag>`.
