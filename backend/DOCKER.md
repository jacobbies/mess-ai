# Docker Deployment Guide

## Overview

Multi-stage Docker build that bundles the `mess` library with the backend API.

**Build Strategy:** Copy `mess/` directory into Docker image during build (Option 1 from workspace discussion).

## Quick Start

### Build and Run Locally

```bash
# From backend/ directory
cd backend

# Build the image (builds from parent directory context)
make build

# Run the container
make run

# View logs
make logs

# Stop container
make stop
```

### Manual Build (without Make)

```bash
# From project root directory
cd /home/jacobb423/Projects/mess-ai

# Build image
docker build -f backend/Dockerfile -t mess-ai-backend:latest .

# Run container
docker run -d \
  --name mess-backend \
  -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_REGION=us-east-1 \
  mess-ai-backend:latest

# Check logs
docker logs -f mess-backend

# Test health endpoint
curl http://localhost:8000/health
curl http://localhost:8000/docs  # API documentation
```

## How It Works

### Build Context

```
docker build -f backend/Dockerfile .
              ↑                    ↑
         Dockerfile in         Build from
         backend/ dir          parent dir
```

**Why build from parent?** The Dockerfile needs access to both:
- `mess/` - The library
- `backend/` - The API application

### Multi-Stage Build

```dockerfile
# Stage 1: Builder
- Installs uv (fast package installer)
- Copies mess/ library
- Copies backend/ app
- Installs both into .venv
- Result: /app/.venv with all dependencies

# Stage 2: Runtime
- Clean python:3.11-slim base
- Copies only .venv and app code
- Runs as non-root user
- Final image: ~500-600MB
```

### What Gets Copied

**Included in image:**
- ✅ `mess/` - Python package only (no data/)
- ✅ `backend/app/` - FastAPI application
- ✅ All Python dependencies (numpy, faiss, etc.)

**Excluded (via .dockerignore):**
- ❌ `data/` - Audio files and features (loaded from S3)
- ❌ `research/` - Notebooks and scripts
- ❌ `.venv/` - Local dev environment
- ❌ `__pycache__/` - Python cache files

### Dependency Resolution

```python
# In container:
from mess.search import LayerBasedRecommender  # ✅ Works!
from mess.datasets import DatasetFactory       # ✅ Works!
import numpy as np                             # ✅ Transitive from mess-ai
import faiss                                   # ✅ Transitive from mess-ai
```

mess-ai is installed as a regular package (not editable), backend imports from it normally.

## Environment Variables

Required for S3 access:
```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
```

Set via:
- Docker run: `-e VAR=value`
- Docker Compose: `environment:` section
- AWS ECS: Task definition environment
- `.env` file (for local dev with docker-compose)

## Deployment

### Push to ECR (AWS)

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and tag
docker build -f backend/Dockerfile -t mess-ai-backend .
docker tag mess-ai-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/mess-ai-backend:latest

# Push
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/mess-ai-backend:latest
```

### Using Makefile

```bash
# Set registry
export REGISTRY=<account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push
make build
make push
```

### Deploy to EC2/ECS

**EC2:**
```bash
ssh your-ec2-instance
docker pull <registry>/mess-ai-backend:latest
docker run -d -p 80:8000 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  <registry>/mess-ai-backend:latest
```

**ECS Task Definition:**
```json
{
  "family": "mess-ai-backend",
  "containerDefinitions": [{
    "name": "backend",
    "image": "<registry>/mess-ai-backend:latest",
    "portMappings": [{"containerPort": 8000, "hostPort": 8000}],
    "environment": [
      {"name": "AWS_REGION", "value": "us-east-1"}
    ],
    "secrets": [
      {"name": "AWS_ACCESS_KEY_ID", "valueFrom": "arn:aws:secretsmanager:..."},
      {"name": "AWS_SECRET_ACCESS_KEY", "valueFrom": "arn:aws:secretsmanager:..."}
    ]
  }]
}
```

## Development vs Production

**Local Development (workspace):**
```bash
cd backend
uv run uvicorn app.main:app --reload
# Changes to mess/ immediately visible ✅
```

**Docker (production-like):**
```bash
cd backend
make build  # Rebuilds image with current mess/ code
make run    # Runs in container
# Changes to mess/ require rebuild ⚠️
```

**Docker Dev Mode (hybrid):**
```bash
make run-dev  # Mounts app/ for hot reload
# Changes to app/ immediately visible ✅
# Changes to mess/ require rebuild ⚠️
```

## Troubleshooting

### Build fails with "COPY failed"
- Make sure you're building from parent directory
- Check that mess/ directory exists and contains __init__.py

### Import errors in container
- Verify mess-ai is installed: `docker exec mess-backend pip list | grep mess`
- Check Python can import: `docker exec mess-backend python -c "import mess; print('OK')"`

### Health check failing
- Check if container is running: `docker ps`
- View logs: `docker logs mess-backend`
- Manual test: `docker exec mess-backend curl http://localhost:8000/health`

### Large image size
- Current size should be ~500-600MB
- If larger, check .dockerignore is excluding data/
- Use `docker image inspect mess-ai-backend` to analyze layers

## Makefile Commands Reference

```bash
make help       # Show all available commands
make build      # Build Docker image
make run        # Run container in background
make run-dev    # Run with hot reload (mount app/)
make stop       # Stop and remove container
make logs       # Follow container logs
make shell      # Open shell in running container
make test       # Test health endpoint
make clean      # Remove container and image
make push       # Push to registry (set REGISTRY env var)
```

## Files Created

```
backend/
├── Dockerfile         # Multi-stage build configuration
├── .dockerignore      # Files to exclude from image
├── Makefile           # Docker operation helpers
└── DOCKER.md          # This file

(parent directory)
└── .dockerignore      # Build context exclusions
```

## Next Steps

1. **Test locally:**
   ```bash
   cd backend
   make build
   make run
   curl http://localhost:8000/health
   ```

2. **Update backend code to use mess library:**
   ```python
   # backend/app/services/faiss_service.py
   from mess.search.layer_based_recommender import LayerBasedRecommender
   ```

3. **Deploy to production:**
   - Push to ECR
   - Update ECS task definition
   - Deploy new version
