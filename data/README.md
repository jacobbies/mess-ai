# Data Directory

This repository expects runtime data under `data/`, but large/generated artifacts are intentionally git-ignored.

## Quick Demo Bootstrap

Create a tiny synthetic dataset for first-run commands:

```bash
uv run python scripts/setup_demo_data.py --data-root data
```

This generates:

- `data/audio/smd/wav-44/*.wav` (3 synthetic mono WAV tracks)
- `data/metadata/smd_metadata.csv` (basic metadata rows for hybrid search demos)

## Data Source And Licensing

### Synthetic demo data

- Produced locally by `scripts/setup_demo_data.py`
- Not copied from external recordings
- Intended license: `CC0-1.0`

### Real datasets (bring your own data)

For research-quality runs, place real audio into:

- `data/audio/smd/wav-44/`
- `data/audio/maestro/`

You are responsible for obtaining data from official dataset sources and complying with the original dataset licenses/terms.

## Generated Outputs

Workflow scripts write outputs under:

- `data/embeddings/`
- `data/proxy_targets/`
- `data/indices/`
- `data/metadata/`

These outputs are environment-local and should not be committed.
