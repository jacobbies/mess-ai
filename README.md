# mess-ai: AI-Powered Music Similarity Search

A production-ready system for discovering musical similarity using deep learning.

## 🏗️ Project Structure

```
mess-ai/
├── src/                    # 🎯 Backend Microservice (Self-contained)
│   ├── api/               # FastAPI application
│   ├── core/              # Configuration & dependencies
│   ├── mess_ai/           # ML & audio processing
│   ├── Dockerfile         # Container definition
│   ├── requirements.txt   # Dependencies
│   ├── docker-compose.yml # Development setup
│   └── deploy.sh          # Deployment script
│
├── frontend/               # React web application (separate deployment)
├── data/                   # Datasets & processed features (not in containers)
├── notebooks/              # Research & experimentation
├── scripts/                # Utilities & data processing
└── docs/                   # Documentation
```

## 🎯 Components

### Backend Microservice (`src/`)
**Self-contained FastAPI service** with everything needed for production deployment:
- MERT-based music similarity search
- FAISS high-performance indexing  
- Multi-dataset support (SMD, Maestro)
- AWS integration (S3, RDS, CloudFront)
- Production-ready containerization

[**→ See Backend README**](src/README.md)

### Frontend Application (`frontend/`)
React-based web interface for music discovery:
- Interactive music player
- Real-time similarity search
- Waveform visualization
- Responsive design

### Data Processing (`scripts/`, `notebooks/`)
Tools for dataset processing and feature extraction:
- MERT feature extraction pipeline
- Multi-dataset metadata processing  
- FAISS index optimization
- Research notebooks

## 🚀 Quick Start

### Backend Development
```bash
cd src
docker-compose up -d
# API at http://localhost:8000
```

### Backend Production  
```bash
cd src
./deploy.sh production
```

### Frontend Development
```bash
cd frontend
npm install && npm start
# App at http://localhost:3000
```

## 🎵 Architecture

**mess-ai** uses a microservice architecture optimized for music AI:

1. **Feature Extraction**: MERT transformer processes audio → 768-dim embeddings
2. **Similarity Search**: FAISS IndexFlatIP provides sub-millisecond queries  
3. **Multi-Dataset**: Unified interface for SMD (50 tracks) + Maestro (1,276 tracks)
4. **Hybrid Storage**: Local features for speed + S3/CDN for audio streaming
5. **Production Ready**: Containerized, scalable, AWS-optimized

## 📊 Performance

- **Feature Extraction**: 2.6 minutes for full dataset (M3 Pro + MPS)
- **Search Latency**: <1ms similarity queries via FAISS
- **Throughput**: 4 Gunicorn workers handle concurrent requests
- **Storage**: 94GB features + 241GB audio (hybrid local/cloud strategy)

## 🔧 Technology Stack

- **Backend**: Python 3.11, FastAPI, PyTorch, FAISS, PostgreSQL
- **Frontend**: React, TypeScript, Tailwind CSS
- **ML**: MERT-v1-95M, Wav2Vec2, transformers
- **Infrastructure**: Docker, AWS (EC2, RDS, S3, CloudFront)
- **Audio**: librosa, soundfile, torchaudio

## 🎯 Deployment Strategy

### Phase 1: MVP (0-50 users)
- Backend microservice on single EC2 instance  
- Frontend on S3 + CloudFront
- PostgreSQL RDS
- Local EBS storage for features

### Phase 2: Growth (50-200 users)
- Auto Scaling Groups
- Load balancer
- S3 hybrid storage
- Caching layer

### Phase 3: Scale (200-500 users)  
- Multi-AZ deployment
- ElastiCache
- Advanced monitoring
- Performance optimization

## 🎵 Supported Datasets

- **SMD (Saarland Music Dataset)**: 50 classical recordings, curated for research
- **MAESTRO v3.0**: 1,276 piano performances, competition-grade recordings
- **Extensible**: Architecture supports adding new datasets

## 🔒 Production Features

- **Security**: Non-root containers, environment-based config, CORS protection
- **Monitoring**: Structured logging, health checks, metrics ready  
- **Scalability**: Horizontal scaling, resource limits, performance tuning
- **Reliability**: Health checks, restart policies, graceful shutdown

---

**Built for production music AI at scale** 🎵 ✨