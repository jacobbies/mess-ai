# Repository Guidelines

## Project Structure & Module Organization
- `mess/`: core Python package.
  - `mess/extraction/`: MERT feature extraction pipeline.
  - `mess/probing/`: layer discovery and proxy-target probing.
  - `mess/search/`: FAISS-backed similarity and aspect-based recommendation.
  - `mess/datasets/`: dataset loaders (`smd`, `maestro`) and factory logic.
- `scripts/`: CLI entry points for extraction, probing, evaluation, and demos.
- `notebooks/`: exploratory analysis notebooks.
- `data/`: local datasets, embeddings, indices, and model artifacts.
- `docs/`: project documentation and evaluation plans.

## Build, Test, and Development Commands
- `uv sync`: install core dependencies.
- `uv sync --group dev`: install full ML/research stack (`mess-ai[ml]`).
- `python scripts/extract_features.py --dataset smd`: generate embeddings.
- `python scripts/run_probing.py --samples 50 --alpha 1.0 --folds 5`: run layer discovery.
- `python scripts/demo_recommendations.py --track "<TRACK_ID>"`: smoke-test recommendation flow.
- `mlflow ui`: inspect experiment runs at `http://localhost:5000`.

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indentation, PEP 8 naming (`snake_case` functions/modules, `PascalCase` classes).
- Prefer type hints and concise docstrings for public functions.
- Keep modules focused by domain (`extraction`, `probing`, `search`, `datasets`) rather than utility dumping.
- Script names in `scripts/` should be verb-first and descriptive (for example, `extract_features.py`).

## Testing Guidelines
- There is currently no dedicated `tests/` suite or coverage gate in this repository.
- Validate changes with targeted script-level smoke tests relevant to your edits.
- For search/probing changes, run at minimum:
  - `python scripts/run_probing.py --samples 10`
  - `python scripts/demo_recommendations.py --track "<TRACK_ID>"`

## Documentation Placement
- Keep canonical, project-wide documentation in `docs/` (architecture, onboarding, standards).
- Keep experiment notes and analysis writeups in `research/`.
- Add short module-local references (for example, `mess/probing/DISCOVERY_REFERENCE.md`) only when the content is tightly coupled to one code path and should be updated with that module.
- When adding a module-local doc, add a link from a central doc (`README.md` or `docs/`) so it remains discoverable.

## Commit & Pull Request Guidelines
- Follow the existing history style: short, imperative commit subjects (for example, `Refactor search module`).
- Keep one logical change per commit; avoid mixing refactors and behavior changes.
- PRs should include:
  - clear problem/solution summary,
  - commands run for validation,
  - linked issue(s) when applicable,
  - sample output/screenshots for result-facing changes (MLflow views, recommendation output).
- GitHub Actions run Claude-based review on PR updates; address review feedback before merge.
