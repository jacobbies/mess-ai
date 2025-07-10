# MESS-AI: Music Similarity Search Engine

A production-ready music similarity search system using MERT embeddings and FAISS for fast, accurate music discovery.

## What It Does

MESS-AI finds musically similar tracks using AI embeddings. Upload a song or select from the library, and get instant recommendations based on musical characteristics like melody, harmony, and rhythm.

**Key Features:**
- Sub-millisecond similarity search across 1,300+ tracks
- Real-time audio playback with waveform visualization  
- Multi-dataset support (classical, piano, extensible)
- Production Docker deployment with monitoring

## Quick Start

### Development Environment
```bash
# Start full stack with Docker
./scripts/dev.sh

# Access points:
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Simple Demo
```bash
# Backend only
docker-compose -f deploy/docker-compose.yml up

# Open demo frontend
open frontend/simple/index.html
```

## Architecture

```
Audio Files → MERT Embeddings → FAISS Search → FastAPI → React UI
```

**Tech Stack:**
- **ML**: MERT transformer, FAISS similarity search, PyTorch
- **Backend**: FastAPI, async Python, Pydantic validation
- **Frontend**: React 19, TypeScript, TailwindCSS
- **Infrastructure**: Docker, multi-stage builds, health checks

## Performance

| Metric | Value |
|--------|-------|
| Search Speed | <1ms |
| Feature Extraction | 2.6 min/dataset (M3 Pro) |
| Dataset Scale | 1,300+ tracks |
| Memory Usage | <2GB RAM |

## Project Structure

```
mess-ai/
├── backend/           # FastAPI server & ML integration
├── frontend/          # React application
├── pipeline/          # ML processing & MERT features  
├── deploy/            # Docker configs & CI/CD
├── scripts/           # Automation & testing
└── data/              # Audio files & features
```

## Development

### Local Setup
```bash
# Backend development
docker-compose -f deploy/docker-compose.yml up
cd backend && python -m uvicorn api.main:app --reload

# Frontend development  
cd frontend && npm install && npm start

# Full stack development
./scripts/dev.sh
```

### Testing
```bash
# Backend tests
cd backend && pytest

# Frontend tests
cd frontend && npm test

# Integration tests
python scripts/test_database_integration.py
```

## Deployment

### Single Instance (EC2)
```bash
# Automated deployment
./deploy/scripts/deploy-ec2.sh

# Manual deployment
cd deploy
cp .env.production.example .env.production
docker-compose -f docker-compose.prod.yml up -d
```

### Requirements
- **Development**: Docker, Node.js 18+, Python 3.11+
- **Production**: 4+ vCPUs, 16GB+ RAM, 100GB+ storage

## Datasets

- **SMD**: 50 classical tracks for development
- **MAESTRO**: 1,276 piano pieces for production scale
- **Custom**: Extensible architecture for any audio dataset

## Key Features

### Fast Similarity Search
Uses FAISS IndexFlatIP for exact cosine similarity with sub-millisecond response times.

### Multi-Scale Features
MERT embeddings processed at multiple scales: raw temporal features, segment averages, and track-level aggregates.

### Production Ready
- Health checks and monitoring
- Non-root containers for security
- Async request handling
- Comprehensive error handling
- Environment-based configuration

### Developer Experience
- Hot reload in development
- Comprehensive documentation
- Type safety with TypeScript/Pydantic
- Local CLAUDE.md files for context-specific guidance

## API Endpoints

```bash
# Get all tracks
GET /tracks

# Get recommendations
GET /recommend/{track_name}?top_k=5

# Stream audio
GET /audio/{filename}

# Generate waveform
GET /waveform/{filename}

# Health check
GET /health
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.