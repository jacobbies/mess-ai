# Scripts and Automation Instructions

## Development Automation

### Main Development Script
```bash
# Start full development environment
./scripts/dev.sh

# What it does:
# 1. Checks Docker is running
# 2. Cleans up existing containers
# 3. Builds and starts services
# 4. Verifies health checks
# 5. Displays access URLs and useful commands
```

### Script Structure
```bash
scripts/
├── dev.sh                    # Main development startup
├── extract_maestro_features.py  # MAESTRO dataset processing
├── maestro_extractor.py      # Feature extraction utilities
├── migrate_features_to_database.py  # Database migration
├── process_maestro.py        # MAESTRO data processing
├── test_database.py          # Database testing
├── test_database_integration.py  # Integration testing
├── test_multi_dataset_library.py  # Multi-dataset testing
└── test_multi_dataset_search.py   # Search functionality testing
```

## Data Processing Scripts

### MAESTRO Dataset Processing
```bash
# Extract features from MAESTRO dataset
python scripts/extract_maestro_features.py

# Process MAESTRO metadata
python scripts/process_maestro.py

# What these scripts do:
# 1. Download MAESTRO dataset if needed
# 2. Extract MERT embeddings from audio files
# 3. Process metadata into standardized format
# 4. Cache features for fast similarity search
```

### Feature Extraction Pipeline
```python
# Reference: backend/CLAUDE.local.md for MERT implementation details
# Reference: global CLAUDE.md for data flow architecture

# Extract features from audio files
from pipeline.mess_ai.features.extractor import extract_features

# Process full dataset (~2.6 minutes on M3 Pro)
extract_features()

# Custom feature extraction
def extract_custom_features(audio_path, output_dir):
    """Extract MERT features from audio file."""
    # Load audio at 44kHz (MERT requirement)
    # Process through MERT-v1-95M model
    # Save as .npy files (raw, segments, aggregated)
    pass
```

### Database Migration Scripts
```bash
# Migrate features to database
python scripts/migrate_features_to_database.py

# Test database connectivity
python scripts/test_database.py

# Run integration tests
python scripts/test_database_integration.py
```

## Testing Automation

### Multi-Dataset Testing
```bash
# Test multi-dataset library functionality
python scripts/test_multi_dataset_library.py

# Test search across multiple datasets
python scripts/test_multi_dataset_search.py

# What these test:
# 1. Dataset loading and metadata processing
# 2. Cross-dataset similarity search
# 3. FAISS index consistency
# 4. Performance benchmarks
```

### Test Data Management
```python
# Create test fixtures
def create_test_data():
    """Generate test audio files and metadata."""
    # Generate synthetic audio samples
    # Create mock metadata entries
    # Set up test FAISS indices
    pass

# Clean test data
def cleanup_test_data():
    """Remove test files and reset state."""
    # Delete test audio files
    # Clear test database entries
    # Reset FAISS indices
    pass
```

## Jupyter Notebook Automation

### Feature Extraction Testing
```bash
# Start Jupyter server
jupyter notebook notebooks/

# Key notebooks:
# - notebooks/test_feature_extraction.ipynb
# - notebooks/similarity_analysis.ipynb
# - notebooks/dataset_exploration.ipynb
```

### Notebook Automation
```python
# Run notebooks programmatically
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_path, output_path=None):
    """Execute Jupyter notebook programmatically."""
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': './'}})
    
    if output_path:
        with open(output_path, 'w') as f:
            nbformat.write(nb, f)
```

## Performance Monitoring Scripts

### Benchmarking
```python
# Benchmark feature extraction performance
def benchmark_feature_extraction():
    """Measure MERT feature extraction performance."""
    import time
    
    start_time = time.time()
    # Run feature extraction
    extract_features()
    end_time = time.time()
    
    print(f"Feature extraction took: {end_time - start_time:.2f} seconds")

# Benchmark FAISS search performance
def benchmark_search_performance():
    """Measure FAISS similarity search performance."""
    # Test query performance
    # Measure index build time
    # Compare different FAISS index types
    pass
```

### System Monitoring
```python
# Monitor system resources during processing
import psutil
import GPUtil

def monitor_system_resources():
    """Monitor CPU, RAM, and GPU usage."""
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    try:
        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load * 100 if gpus else 0
    except:
        gpu_usage = 0
    
    print(f"CPU: {cpu_percent}%, RAM: {memory_percent}%, GPU: {gpu_usage}%")
```

## Data Validation Scripts

### Dataset Integrity Checks
```python
# Validate dataset integrity
def validate_dataset_integrity():
    """Check dataset files and metadata consistency."""
    # Verify all audio files exist
    # Check metadata completeness
    # Validate feature file consistency
    # Report any issues found
    pass

# Validate FAISS indices
def validate_faiss_indices():
    """Check FAISS index integrity."""
    # Verify index dimensions
    # Check for corrupted indices
    # Validate search results
    # Report index statistics
    pass
```

### Data Quality Metrics
```python
# Calculate data quality metrics
def calculate_data_quality():
    """Calculate dataset quality metrics."""
    # Audio quality metrics (SNR, dynamic range)
    # Metadata completeness percentage
    # Feature extraction success rate
    # Similarity search accuracy
    pass
```

## Deployment Automation

### Environment Setup
```bash
# Set up development environment
./scripts/setup_dev_environment.sh

# What it does:
# 1. Check system requirements
# 2. Install Python dependencies
# 3. Set up virtual environment
# 4. Download required models
# 5. Initialize database
```

### Data Synchronization
```bash
# Sync data with remote storage
./scripts/sync_data.sh

# Options:
# --upload: Upload local data to S3
# --download: Download data from S3
# --backup: Create backup of current data
```

### Maintenance Scripts
```python
# Clean up old files
def cleanup_old_files():
    """Remove old temporary files and logs."""
    # Delete old log files
    # Remove temporary processing files
    # Clean up old model checkpoints
    pass

# Update dependencies
def update_dependencies():
    """Update Python and Node.js dependencies."""
    # Update pip packages
    # Update npm packages
    # Check for security vulnerabilities
    pass
```

## Utility Functions

### File Management
```python
# File utilities
def ensure_directory_exists(path):
    """Create directory if it doesn't exist."""
    import os
    os.makedirs(path, exist_ok=True)

def get_file_size(path):
    """Get file size in human-readable format."""
    import os
    size = os.path.getsize(path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"
```

### Configuration Management
```python
# Configuration utilities
def load_config(config_path):
    """Load configuration from YAML or JSON file."""
    import yaml
    import json
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)

def validate_config(config):
    """Validate configuration parameters."""
    # Check required parameters
    # Validate data types
    # Check file paths exist
    pass
```

## Error Handling and Logging

### Script Logging
```python
import logging
import sys

def setup_logging(log_level='INFO'):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('scripts.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def handle_script_errors(func):
    """Decorator for error handling in scripts."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Script failed: {str(e)}")
            sys.exit(1)
    return wrapper
```

## Related Documentation
- **System Architecture**: See global CLAUDE.md for data flow and processing pipeline
- **Backend Implementation**: See backend/CLAUDE.local.md for ML pipeline and API details
- **Frontend Integration**: See frontend/CLAUDE.local.md for UI testing and build processes
- **Deployment Procedures**: See deploy/CLAUDE.local.md for automation and CI/CD integration