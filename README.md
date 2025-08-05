# MESS-AI: Music Similarity Search Engine

A production-ready music similarity search system using MERT embeddings and FAISS for fast, accurate music discovery across classical music datasets.

## Overview

MESS-AI finds musically similar tracks using transformer-based AI embeddings. Select from a curated library of classical music, and get instant recommendations based on deep musical characteristics like melody, harmony, rhythm, and form.

**Key Features:**
- ⚡ **Sub-millisecond** similarity search across 1,300+ tracks
- 🎵 **Real-time audio playback** with waveform visualization  
- 🎼 **Multi-dataset support** (SMD, MAESTRO) with extensible architecture
- 🚀 **Production-ready** Docker deployment with monitoring
- 🧠 **Advanced ML pipeline** using MERT-v1-95M transformer
- 🔍 **FAISS-powered search** with IndexFlatIP for exact similarity

## Quick Start

### Development Environment (Recommended)
```bash
# Backend development with hot reload
cd backend && python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Access points:
# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

### Manual Setup
```bash
# Install dependencies
python -m pip install -r requirements.txt

# Start backend server
cd backend && python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Architecture

```
Audio Files → MERT Feature Extraction → FAISS Similarity Search → FastAPI Backend → JSON API
```

### Core Components

- **Feature Extraction**: MERT-v1-95M transformer processes 44kHz audio into multi-scale embeddings
- **Similarity Search**: FAISS IndexFlatIP provides exact cosine similarity with <1ms query time
- **Backend API**: FastAPI with async request handling and comprehensive error management
- **REST API**: FastAPI with OpenAPI documentation and CORS support
- **Data Processing**: Smart caching, feature persistence, and scalable dataset handling

### Tech Stack

- **Machine Learning**: PyTorch 2.6+, transformers 4.38+, FAISS, MERT embeddings
- **Backend**: Python 3.11+, FastAPI, Pydantic, async/await
- **API Framework**: FastAPI, Pydantic validation, OpenAPI/Swagger docs
- **Audio Processing**: soundfile, librosa, torchaudio with Apple Silicon acceleration
- **Infrastructure**: Docker multi-stage builds, health checks, CORS configuration

## Performance Metrics

| Component | Performance |
|-----------|------------|
| **Similarity Search** | <1ms per query |  
| **Feature Extraction** | 2.6 minutes for full SMD dataset (M3 Pro + MPS) |
| **Dataset Scale** | 1,300+ tracks (MAESTRO), 50 tracks (SMD) |
| **Memory Usage** | <2GB RAM for full system |
| **Storage** | 94GB processed features (all scales) |
| **API Response** | <100ms for metadata, <500ms for recommendations |

## Project Structure

```
mess-ai/
├── backend/                    # FastAPI application & ML integration
│   ├── api/                   # REST API endpoints and routers
│   ├── core/                  # Configuration, dependencies, services
│   ├── mess_ai/               # ML models, search, datasets, audio
│   │   ├── models/           # Recommendation engines & metadata
│   │   ├── search/           # FAISS similarity search & caching
│   │   ├── pipeline/         # MERT feature extraction pipeline
│   │   ├── datasets/         # SMD, MAESTRO dataset handlers
│   │   └── audio/            # Audio playback & waveform generation
│   └── cli.py                # Command-line interface for pipeline ops
├── data/                     # Audio files and processed features
│   ├── smd/                  # Saarland Music Dataset (50 tracks)
│   ├── maestro/              # MAESTRO Dataset (1,276 tracks)
│   ├── processed/features/   # MERT embeddings (94GB total)
│   │   ├── raw/             # Full temporal features
│   │   ├── segments/        # Time-averaged features  
│   │   └── aggregated/      # Track-level features (used for search)
│   └── models/              # Future model checkpoints
└── requirements.txt          # Production dependencies (22 packages)
```

## Dataset Information

### Saarland Music Dataset (SMD)
- **50 tracks** of classical music at 44kHz
- Performance annotations and MIDI representations
- Used for development and testing
- Fast feature extraction (~2.6 minutes)

### MAESTRO Dataset  
- **1,276 piano pieces** from competition performances
- High-quality recordings with rich metadata
- Production-scale similarity search
- Composer, era, and form classifications

### Custom Dataset Support
The system supports extensible dataset integration with:
- Automatic metadata parsing
- Flexible audio format support  
- Configurable feature extraction
- Dataset-specific preprocessing

## API Reference

### Core Endpoints

```bash
# Get all tracks with optional filtering
GET /tracks?composer=bach&era=baroque&search=fugue

# Get track recommendations  
GET /recommend/{track_id}?n_recommendations=10&strategy=similarity

# Stream audio files
GET /audio/{track_id}

# Generate waveform visualization
GET /waveform/{track_id}  

# System health and metrics
GET /health

# Track metadata
GET /metadata/{track_id}
```

### Response Formats

```typescript
// Recommendation Response
interface RecommendationResponse {
  reference_track: string;
  recommendations: Recommendation[];
  total_tracks: number;
  reference_metadata?: TrackMetadata;
}

// Track Metadata
interface TrackMetadata {
  track_id: string;
  title: string;
  composer: string;
  composer_full: string;
  era?: string;
  form?: string;
  opus?: string;
  movement?: string;
  filename: string;
  tags: string[];
  recording_date?: string;
}
```

## Development Workflow

### Feature Extraction Pipeline

```bash
# CLI for pipeline operations
python backend/cli.py extract --show-config          # Show current configuration
python backend/cli.py extract --audio-file track.wav # Process single file
python backend/cli.py extract --audio-dir /path/to/audio # Process directory
python backend/cli.py validate --verbose             # Validate setup
```

### Configuration Management

The system uses a hybrid configuration approach:
- **Auto-detection**: Automatically finds project root and sets paths
- **Environment overrides**: Use `MERT_DEVICE`, `MERT_OUTPUT_DIR` for customization  
- **Device optimization**: Prefers MPS → CUDA → CPU for best performance
- **Smart defaults**: Works out-of-the-box for development

### Development Commands

```bash
# Backend development with hot reload
cd backend && python -m uvicorn api.main:app --reload

# Feature extraction
python backend/cli.py extract

# Validate configuration
python backend/cli.py validate --verbose

# API testing
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

### Testing & Quality Assurance

```bash
# Import validation (all modules)
python -c "from mess_ai.models import AsyncUnifiedMusicRecommender"

# Configuration testing
python backend/cli.py validate

# API health check
curl http://localhost:8000/health
```

## Production Deployment

### System Requirements

- **CPU**: 4+ cores (8+ recommended for concurrent requests)
- **Memory**: 16GB+ RAM (features loaded in memory)  
- **Storage**: 100GB+ (audio files + processed features)
- **Network**: High bandwidth for audio streaming

### Docker Deployment

```bash
# Local development (if Docker configs exist)
# docker-compose up

# Manual deployment
cd backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Environment configuration
export MERT_DEVICE=cuda  # or mps, cpu
export MERT_OUTPUT_DIR=/path/to/features
```

### Monitoring & Health Checks

The system includes comprehensive health monitoring:
- **Service Health**: Memory usage, disk space, service status
- **ML Pipeline**: Feature extraction status, model availability
- **API Performance**: Response times, error rates
- **Database**: Metadata integrity, search index status

## Advanced Features

### Recommendation Strategies

- **Similarity**: Pure cosine similarity using MERT embeddings
- **Diverse**: Maximal Marginal Relevance for varied recommendations
- **Popular**: Boost recommendations by composer popularity  
- **Hybrid**: Balanced approach combining multiple strategies
- **Random**: Discovery mode for exploration

### Caching System

- **Multi-level caching**: Memory + disk persistence
- **FAISS index caching**: Instant startup with pre-built indices
- **Smart invalidation**: Automatic cache management
- **Performance optimization**: Sub-millisecond repeat queries

### Apple Silicon Optimization

- **MPS acceleration**: Native Metal Performance Shaders support
- **CPU fallback**: Automatic fallback for compatibility
- **Memory efficiency**: Optimized tensor operations
- **Fast feature extraction**: ~2.6 minutes for full dataset

## Troubleshooting

### Common Issues

**Import Errors**: Ensure you're running from the `backend/` directory
```bash
cd backend && python -c "from mess_ai.models import AsyncUnifiedMusicRecommender"
```

**Feature Extraction Fails**: Check device availability
```bash  
python backend/cli.py validate --verbose
```

**API Connection Issues**: Verify CORS configuration in settings
```bash
curl -v http://localhost:8000/health
```

**Audio Playback Issues**: Ensure audio files are in correct WAV format (44kHz)

### Performance Optimization

- Use MPS device on Apple Silicon for 3x faster feature extraction
- Pre-extract features for large datasets to avoid runtime delays
- Enable FAISS index caching for instant startup
- Use appropriate worker counts for your CPU core count

## Contributing

1. **Fork** the repository on GitHub
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes with appropriate tests
4. **Update** documentation and CLAUDE.md files as needed
5. **Test** all imports and functionality
6. **Submit** a pull request with detailed description

### Code Standards

- **Python**: Black formatting, type hints, async/await patterns
- **API Design**: RESTful patterns, proper HTTP status codes  
- **Documentation**: Update README and relevant CLAUDE.md files
- **Testing**: Include tests for new functionality
- **Docker**: Multi-stage builds, non-root users, health checks

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with ❤️ for music discovery and AI-powered recommendation systems.**