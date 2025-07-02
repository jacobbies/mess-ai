# Data Processing Scripts

Utilities for dataset processing and feature extraction.

## Available Scripts

### Dataset Processing
- `process_maestro.py` - Process MAESTRO dataset metadata
- `extract_maestro_features.py` - Extract MERT features from MAESTRO
- `generate_waveforms.py` - Generate waveform visualizations

### Database & Migration
- `migrate_features_to_database.py` - Migrate features to database
- `test_database.py` - Test database connectivity
- `test_database_integration.py` - Integration tests

### Testing & Validation
- `test_multi_dataset_library.py` - Test multi-dataset support
- `test_multi_dataset_search.py` - Test cross-dataset search
- `maestro_extractor.py` - Maestro-specific extraction utilities

## Usage

Run scripts from project root:

```bash
# Process Maestro metadata
python scripts/process_maestro.py

# Extract features for Maestro dataset
python scripts/extract_maestro_features.py

# Generate waveform visualizations
python scripts/generate_waveforms.py
```

## Note

For production deployment, use the backend microservice in `src/` instead of these development scripts.