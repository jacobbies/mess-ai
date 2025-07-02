# mess-ai Backend Microservice

Production-ready FastAPI backend for AI-powered music similarity search.

## 🏗️ Architecture

This is a self-contained microservice that provides:
- **FastAPI REST API** for music similarity search
- **MERT embeddings** for deep music understanding  
- **FAISS similarity search** with sub-millisecond queries
- **Multi-dataset support** (SMD, Maestro)
- **AWS integration** (S3, RDS, CloudFront)

## 🚀 Quick Start

### Development
```bash
# Start with PostgreSQL
docker-compose up -d

# API available at http://localhost:8000
# Health check: curl http://localhost:8000/health/ready
```

### Production
```bash
# Deploy to production
./deploy.sh production

# Or build image only
docker build -t mess-ai-backend --target production .
```

## 📁 Structure

```
src/
├── api/                    # FastAPI application
├── core/                   # Configuration & dependencies  
├── mess_ai/               # Core ML & audio processing
├── Dockerfile             # Container definition
├── requirements*.txt      # Dependencies
├── docker-compose*.yml    # Orchestration
└── deploy.sh             # Deployment script
```

## 🔧 Configuration

Environment variables in `.env.production`:

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql://user:pass@host:5432/db` |
| `AWS_S3_BUCKET` | Audio storage bucket | `my-audio-bucket` |
| `ALLOWED_ORIGINS` | CORS origins | `https://mydomain.com` |
| `DATA_ROOT_DIR` | Data mount point | `/data` |

## 🎯 Endpoints

- `GET /health/ready` - Health check
- `GET /tracks` - List available tracks
- `GET /recommend/{track_id}` - Get similar tracks
- `GET /audio/{track_id}` - Stream audio file
- `GET /waveform/{track_id}` - Get waveform visualization

## 🐳 Container Details

### Production Image
- **Base**: `python:3.11-slim`
- **Size**: ~500MB (optimized)
- **User**: Non-root (`appuser`)
- **Process**: Gunicorn + Uvicorn workers
- **Health**: Built-in health checks

### Data Volumes
- `/data` - Audio files, features, metadata (EBS mount)
- `/app/logs` - Application logs

## 🔒 Security

- Non-root container user
- Minimal attack surface (only required packages)
- Environment-based configuration
- Health check endpoints
- Structured logging

## 📊 Performance

- **Startup**: ~30 seconds (feature loading)
- **Memory**: 4-8GB (configurable)
- **Search**: Sub-millisecond FAISS queries
- **Concurrency**: 4 Gunicorn workers (configurable)

## 🎵 Supported Datasets

- **SMD (Saarland Music Dataset)**: 50 classical recordings
- **Maestro**: 1,276 piano performances (optional)
- **Extensible**: Easy to add new datasets

## 🚀 Deployment

This microservice is designed for:
- **Single EC2 instance** (Phase 1)
- **Auto Scaling Groups** (Phase 2+)  
- **Container orchestration** (ECS, EKS)
- **Local development** with Docker Compose

## 📝 Monitoring

- Structured JSON logging
- Prometheus metrics (optional)
- Health check endpoints
- CloudWatch integration ready

---

**Self-contained microservice** - Everything needed to run the backend is in this directory.