# ML Pipeline Instructions

## Pipeline Architecture Overview

The ML pipeline handles the core machine learning workflows for audio processing, feature extraction, and similarity analysis. This is separate from the backend API layer and focuses on data processing and model operations.

### Directory Structure
```
pipeline/
└── mess_ai/
    ├── features/           # MERT feature extraction
    │   └── extractor.py   # Core feature extraction logic
    └── analysis/          # Analysis and evaluation tools
        ├── embedding_analyzer.py    # Embedding analysis utilities
        └── embedding_diversity.py   # Diversity metrics and analysis
```

## Feature Extraction Pipeline

### MERT Feature Extraction
```python
# Reference: Global CLAUDE.md for overall data flow
# Reference: backend/CLAUDE.local.md for API integration

from pipeline.mess_ai.features.extractor import extract_features

# Extract features from full dataset (~2.6 minutes on M3 Pro)
extract_features()

# Custom feature extraction
def extract_custom_features(audio_paths, output_dir):
    """Extract MERT features from audio files."""
    # Process audio at 44kHz (MERT requirement)
    # Use MERT-v1-95M with trust_remote_code=True
    # Generate multi-scale outputs: raw, segments, aggregated
    # Cache results as .npy files for fast loading
    pass
```

### Feature Processing Workflow
1. **Audio Loading**: Load WAV files at 44kHz sample rate
2. **MERT Processing**: Extract embeddings using MERT-v1-95M transformer
3. **Multi-Scale Output**: Generate raw, segmented, and aggregated features
4. **Caching**: Save as .npy files for fast similarity search
5. **Validation**: Verify feature integrity and dimensions

### Performance Optimization
```python
# Apple Silicon MPS acceleration
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Batch processing for efficiency
def process_audio_batch(audio_files, batch_size=8):
    """Process multiple audio files in batches."""
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i+batch_size]
        # Process batch with MERT model
        # Save features for each file
        pass

# Memory management
def cleanup_memory():
    """Clean up GPU memory after processing."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

## Feature Analysis Tools

### Embedding Analysis
```python
# Reference: scripts/CLAUDE.local.md for analysis automation

from pipeline.mess_ai.analysis.embedding_analyzer import EmbeddingAnalyzer

analyzer = EmbeddingAnalyzer()

# Analyze embedding quality
quality_metrics = analyzer.analyze_embedding_quality(features)

# Dimensionality analysis
dim_analysis = analyzer.analyze_dimensions(features)

# Clustering analysis
clusters = analyzer.perform_clustering(features, n_clusters=10)
```

### Diversity Metrics
```python
from pipeline.mess_ai.analysis.embedding_diversity import DiversityAnalyzer

diversity = DiversityAnalyzer()

# Calculate embedding diversity
diversity_score = diversity.calculate_diversity(embeddings)

# Analyze coverage across musical features
coverage_analysis = diversity.analyze_musical_coverage(embeddings, metadata)

# Compare different embedding methods
comparison = diversity.compare_embedding_methods(embeddings_dict)
```

## Data Processing Patterns

### Audio File Management
```python
# Audio file validation
def validate_audio_files(audio_paths):
    """Validate audio files for processing."""
    valid_files = []
    for path in audio_paths:
        try:
            # Check file exists and is readable
            # Validate audio format and sample rate
            # Check file size and duration
            valid_files.append(path)
        except Exception as e:
            logger.warning(f"Invalid audio file {path}: {e}")
    return valid_files

# Audio preprocessing
def preprocess_audio(audio_path, target_sr=44100):
    """Preprocess audio for MERT processing."""
    import librosa
    
    # Load audio at target sample rate
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Normalize audio
    audio = librosa.util.normalize(audio)
    
    # Handle mono/stereo conversion
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    
    return audio, sr
```

### Feature Storage Management
```python
# Feature file organization
def organize_features(base_dir, dataset_name):
    """Organize feature files by dataset and type."""
    feature_dirs = {
        'raw': base_dir / dataset_name / 'raw',
        'segments': base_dir / dataset_name / 'segments', 
        'aggregated': base_dir / dataset_name / 'aggregated'
    }
    
    for dir_path in feature_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return feature_dirs

# Feature validation
def validate_features(feature_path):
    """Validate extracted features."""
    import numpy as np
    
    try:
        features = np.load(feature_path)
        
        # Check dimensions
        if len(features.shape) != 3:  # [segments, layers, features]
            raise ValueError(f"Invalid feature shape: {features.shape}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            raise ValueError("Features contain NaN or infinite values")
        
        return True
    except Exception as e:
        logger.error(f"Feature validation failed for {feature_path}: {e}")
        return False
```

## Model Integration

### MERT Model Management
```python
# Model loading and configuration
def load_mert_model(model_name="m-a-p/MERT-v1-95M"):
    """Load MERT model with proper configuration."""
    from transformers import Wav2Vec2FeatureExtractor, AutoModel
    
    # Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    return model, feature_extractor

# Model inference
def extract_mert_features(audio, model, feature_extractor, device):
    """Extract MERT features from audio."""
    import torch
    
    # Prepare input
    inputs = feature_extractor(
        audio,
        sampling_rate=44100,
        return_tensors="pt"
    ).to(device)
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state
    
    return features.cpu().numpy()
```

### Feature Aggregation Strategies
```python
# Different aggregation methods
def aggregate_features(features, method='mean'):
    """Aggregate temporal features."""
    import numpy as np
    
    if method == 'mean':
        return np.mean(features, axis=1)  # Average over time
    elif method == 'max':
        return np.max(features, axis=1)   # Max pooling
    elif method == 'std':
        return np.std(features, axis=1)   # Standard deviation
    elif method == 'concat':
        return np.concatenate([
            np.mean(features, axis=1),
            np.std(features, axis=1)
        ], axis=-1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

# Multi-scale feature generation
def generate_multiscale_features(audio, model, feature_extractor, device):
    """Generate features at multiple scales."""
    # Extract raw features
    raw_features = extract_mert_features(audio, model, feature_extractor, device)
    
    # Generate segment-level features
    segment_features = aggregate_features(raw_features, method='mean')
    
    # Generate track-level features
    track_features = np.mean(segment_features, axis=0)
    
    return {
        'raw': raw_features,
        'segments': segment_features,
        'aggregated': track_features
    }
```

## Performance Monitoring

### Processing Metrics
```python
# Track processing performance
def track_processing_metrics(func):
    """Decorator to track processing metrics."""
    import time
    import psutil
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"{func.__name__} took {end_time - start_time:.2f}s")
        logger.info(f"Memory usage: {end_memory - start_memory:.2f}MB")
        
        return result
    return wrapper

# GPU monitoring
def monitor_gpu_usage():
    """Monitor GPU usage during processing."""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            logger.info(f"GPU Usage: {gpu.load*100:.1f}%")
            logger.info(f"GPU Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
    except ImportError:
        logger.warning("GPUtil not available for GPU monitoring")
```

## Error Handling and Validation

### Robust Processing
```python
# Error handling for audio processing
def safe_process_audio(audio_path, max_retries=3):
    """Process audio with error handling and retries."""
    for attempt in range(max_retries):
        try:
            # Process audio file
            features = extract_mert_features(audio_path)
            
            # Validate features
            if validate_features(features):
                return features
            else:
                raise ValueError("Feature validation failed")
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {audio_path}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"All attempts failed for {audio_path}")
                raise
            time.sleep(1)  # Wait before retry

# Data integrity checks
def check_data_integrity(feature_dir):
    """Check integrity of extracted features."""
    import numpy as np
    from pathlib import Path
    
    issues = []
    
    for feature_file in Path(feature_dir).glob("*.npy"):
        try:
            features = np.load(feature_file)
            
            # Check for expected dimensions
            if features.shape[-1] != 768:  # MERT feature dimension
                issues.append(f"{feature_file}: unexpected feature dimension")
            
            # Check for missing values
            if np.any(np.isnan(features)):
                issues.append(f"{feature_file}: contains NaN values")
                
        except Exception as e:
            issues.append(f"{feature_file}: {e}")
    
    return issues
```

## Integration with Other Services

### API Integration Points
```python
# Reference: backend/CLAUDE.local.md for API implementation
# Reference: deploy/CLAUDE.local.md for containerization

# Async processing for API integration
async def process_audio_async(audio_path, callback=None):
    """Process audio asynchronously for API integration."""
    import asyncio
    
    loop = asyncio.get_event_loop()
    
    # Run processing in thread pool
    features = await loop.run_in_executor(
        None,
        extract_mert_features,
        audio_path
    )
    
    if callback:
        await callback(features)
    
    return features

# Batch processing for large datasets
def process_dataset_batch(audio_files, output_dir, batch_size=8):
    """Process dataset in batches for memory efficiency."""
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i:i+batch_size]
            future = executor.submit(process_audio_batch, batch, output_dir)
            futures.append(future)
        
        # Wait for all batches to complete
        for future in futures:
            future.result()
```

## Testing and Validation

### Unit Testing
```python
# Test feature extraction
def test_feature_extraction():
    """Test MERT feature extraction."""
    import numpy as np
    
    # Create test audio
    test_audio = np.random.randn(44100)  # 1 second of random audio
    
    # Extract features
    features = extract_mert_features(test_audio)
    
    # Validate output
    assert features.shape[-1] == 768  # MERT feature dimension
    assert not np.any(np.isnan(features))
    assert not np.any(np.isinf(features))

# Performance testing
def test_processing_performance():
    """Test processing performance on sample data."""
    import time
    
    start_time = time.time()
    
    # Process test dataset
    test_files = ["test1.wav", "test2.wav", "test3.wav"]
    for file in test_files:
        features = extract_mert_features(file)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Verify performance requirements
    assert processing_time < 60  # Should process 3 files in under 1 minute
    logger.info(f"Processing time: {processing_time:.2f}s")
```

## Related Documentation
- **System Architecture**: See global CLAUDE.md for data flow and cross-service integration
- **API Integration**: See backend/CLAUDE.local.md for backend service integration
- **Frontend Usage**: See frontend/CLAUDE.local.md for UI integration patterns
- **Deployment**: See deploy/CLAUDE.local.md for containerization and production deployment
- **Automation**: See scripts/CLAUDE.local.md for pipeline automation and batch processing