# CLAUDE.md - Global Project Instructions

This file provides system-wide guidance for the MESS-AI project. For service-specific instructions, see the corresponding local CLAUDE.md files.

## Cross-Service Documentation
- **Backend**: See `backend/CLAUDE.local.md` for FastAPI, database, and API implementation
- **Frontend**: See `frontend/CLAUDE.local.md` for React, TypeScript, and UI patterns
- **Pipeline**: See `pipeline/CLAUDE.local.md` for ML processing, MERT features, and analysis
- **Deployment**: See `deploy/CLAUDE.local.md` for Docker, CI/CD, and production deployment
- **Scripts**: See `scripts/CLAUDE.local.md` for automation, testing, and data processing

## Development Workflow

### Quick Start
```bash
# Start full development environment
./scripts/dev.sh

# Or start services individually:
# Backend: docker-compose -f deploy/docker-compose.yml up
# Frontend: cd frontend && npm start
```

### Project Structure
```
mess-ai/
├── backend/           # FastAPI server, API endpoints, database
├── frontend/          # React application, UI components
├── pipeline/          # ML processing, MERT features, analysis
├── deploy/            # Docker configuration, CI/CD
├── scripts/           # Automation, testing, data processing
├── data/              # Audio files, features, metadata
└── docs/              # Documentation and examples
```

## System Architecture

**MESS-AI** is a production-ready music similarity search system that uses MERT embeddings to find musically similar classical pieces across multiple datasets.

### Core Data Flow
```
Audio Files → MERT Feature Extraction → FAISS Similarity Search → Web API → React UI
```

1. **Audio Processing**: WAV files at 44kHz processed through MERT-v1-95M transformer
2. **Feature Storage**: Multi-scale embeddings cached as .npy files (94GB total)
3. **Similarity Search**: FAISS IndexFlatIP provides sub-millisecond queries
4. **API Layer**: FastAPI serves audio, metadata, and recommendations
5. **User Interface**: React frontend with real-time similarity search

### Service Communication
- **Frontend ↔ Backend**: REST API with JSON responses
- **Backend ↔ Data**: File-based storage with FAISS indices
- **Development**: Docker networking between services
- **Production**: Load balancer with multiple backend instances

## Cross-Service Data Models

### TrackMetadata Interface
```typescript
interface TrackMetadata {
  track_id: string;
  title: string;
  composer: string;
  composer_full: string;
  era?: string;
  form?: string;
  key_signature?: string;
  opus?: string;
  movement?: string;
  filename: string;
  tags: string[];
  recording_date?: string;
}
```

### API Response Formats
```typescript
interface RecommendationResponse {
  reference_track: string;
  recommendations: Recommendation[];
  total_tracks: number;
  reference_metadata?: TrackMetadata;
}

interface TracksResponse {
  tracks: TrackMetadata[];
  count: number;
  filters?: FilterOptions;
}
```

## Shared Development Standards

### Code Quality
- **Python**: Black formatting, type hints, pytest testing
- **TypeScript**: Strict mode, ESLint, React Testing Library
- **Docker**: Multi-stage builds, non-root users, health checks
- **Git**: Conventional commits, feature branches, PR reviews

### Environment Management
- **Development**: Hot reload, debug logging, test data
- **Staging**: Production-like, integration testing
- **Production**: Optimized builds, monitoring, scaling

### Testing Strategy
- **Unit Tests**: Component/function level testing
- **Integration Tests**: Cross-service communication
- **End-to-End Tests**: Full user workflows
- **Performance Tests**: ML pipeline benchmarks

## Technical Stack

- **Backend:** Python 3.11+, FastAPI, PyTorch 2.6+, transformers 4.38+, FAISS
- **Frontend:** React 19, TypeScript, TailwindCSS, Framer Motion
- **Audio:** soundfile, librosa, torchaudio with Apple Silicon acceleration
- **ML:** MERT-v1-95M transformer, Wav2Vec2FeatureExtractor, FAISS IndexFlatIP
- **Storage:** NumPy .npy files, FAISS indices with disk caching
- **Deployment:** Docker, Docker Compose, multi-stage builds

## Development Status

**✅ Completed (Production Ready):**
- **MERT Feature Extraction Pipeline** - Complete with MPS acceleration, 2.6min processing time
- **FAISS Similarity Search System** - High-performance IndexFlatIP with sub-millisecond queries, 50-100x speedup
- **React Frontend** - Interactive player with AI recommendations and waveform visualization
- **API Endpoints** - `/recommend/{track}`, `/tracks`, `/audio/{track}`, `/waveform/{track}` powered by FAISS
- **Docker Development Environment** - Full-stack containerization with hot reload
- **Apple Silicon Optimization** - MPS acceleration with CPU fallback, optimized FAISS performance
- **Smart Caching** - FAISS index persistence, instant startup, user-controlled recommendations

**🚧 In Progress:**
- Model fine-tuning on SMD dataset for domain-specific similarity
- Advanced FAISS indices (IVF, HNSW) for even larger datasets
- CI/CD pipeline integration with GitHub Actions

**📋 Planned:**
- AWS S3 integration for cloud storage
- Expanded dataset support beyond classical music
- User preference learning and personalization
- Comprehensive testing suite with automated deployment

## Dataset Structure

The Saarland Music Dataset is organized with processed features:

```
data/
├── smd/                    # Original SMD dataset
│   ├── wav-44/            # 50 audio files at 44kHz (MERT compatible)
│   ├── csv/               # Performance annotations
│   └── midi/              # Symbolic representations
├── processed/features/     # MERT embeddings (94GB total)
│   ├── raw/               # Full temporal features [segments, 13, time, 768] 
│   ├── segments/          # Time-averaged [segments, 13, 768]
│   └── aggregated/        # Track-level [13, 768] - used for similarity search
└── models/                # Future training checkpoints
```

## Cross-Service Integration Points

### API Contract Standards
- **Error Responses**: Consistent HTTP status codes and error message formats
- **Data Validation**: Pydantic models for request/response validation
- **Rate Limiting**: Protect ML inference endpoints from overload
- **Health Checks**: All services must implement `/health` endpoints

### Performance Requirements
- **FAISS Queries**: Sub-millisecond response times for similarity search
- **Feature Extraction**: ~2.6 minutes for full dataset processing on M3 Pro
- **API Response Times**: <100ms for metadata queries, <500ms for recommendations
- **Frontend Loading**: <2s initial load, <1s for track switching

### Security Standards
- **CORS Configuration**: Properly configured origins for frontend access
- **Input Validation**: All user inputs sanitized and validated
- **File Access**: Secure audio file serving with proper headers
- **Error Handling**: No sensitive information in error messages

## Development Best Practices

### Git Workflow
- **Feature Branches**: Create branches for new features
- **Conventional Commits**: Use semantic commit messages
- **Pull Requests**: All changes go through PR review process
- **Testing**: All PRs must pass automated tests

### Documentation Standards
- **API Documentation**: OpenAPI/Swagger specs for all endpoints
- **Code Comments**: Document complex algorithms and business logic
- **README Updates**: Keep service documentation current
- **CLAUDE.md Files**: Update local instructions as patterns evolve