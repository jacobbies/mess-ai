# ML Preprocessing Pipeline File System Organization

## Plan A: Hierarchical Processing Structure

### Directory Structure
```
data/
├── smd/                    # Raw dataset (existing)
├── processed/              # All processed/intermediate data
│   ├── features/           # Extracted features
│   │   ├── raw/           # Raw MERT embeddings
│   │   ├── segments/      # Segmented features  
│   │   └── aggregated/    # Track-level aggregated features
│   ├── splits/            # Train/val/test splits
│   └── cache/             # Temporary processing cache
└── models/                # Trained models and checkpoints
    ├── checkpoints/       # Training checkpoints
    └── final/             # Production models
```

### Rationale for Plan A

**1. Clear Separation of Concerns**
- Raw data (`smd/`) remains untouched and preserved
- Processed data is isolated in dedicated directories
- Models are separated from data for clean organization

**2. Scalability for AWS S3 Integration**
- Structure maps well to S3 bucket organization
- Easy to sync specific directories to cloud storage
- Supports versioning and backup strategies

**3. Pipeline Stage Organization**
- `features/raw/` - Direct MERT model outputs (.pt, .npy files)
- `features/segments/` - Time-segmented feature representations
- `features/aggregated/` - Track-level summary features
- `splits/` - Training/validation/test data splits (.json metadata)
- `cache/` - Temporary processing files with auto-cleanup

**4. File Type Strategy**
- **Features**: `.npy`, `.pt` (PyTorch tensors), `.h5` (HDF5 for large arrays)
- **Metadata**: `.json`, `.csv` (track info, segment boundaries, splits)
- **Models**: `.pt`, `.pth` (PyTorch models), `.pkl` (scikit-learn)
- **Cache**: `.cache`, `.tmp` (temporary processing files)
- **Logs**: `.log` (processing logs and metrics)

**5. Memory and Performance Benefits**
- HDF5 format for large embeddings enables efficient partial loading
- Cache directory allows resumable processing
- Organized structure prevents data duplication

**6. Development Workflow Support**
- Checkpoint system supports iterative model development
- Clear separation enables easy cleanup and regeneration
- Structured layout aids debugging and data inspection

### Usage Guidelines
- Use `cache/` for intermediate computations that can be regenerated
- Store final production models in `models/final/`
- Include timestamps/hashes in filenames for version control
- Clean cache directory periodically to manage disk space