# Deployment Instructions

## Docker Development Workflow

### Multi-Environment Setup
```bash
# Full development stack (backend + frontend)
docker-compose -f deploy/docker-compose.dev.yml up --build

# Backend-only development (for local React)
docker-compose -f deploy/docker-compose.yml up --build

# Production deployment
docker-compose -f deploy/docker-compose.prod.yml up --build -d
```

### Build Context Strategy
- **Backend**: Builds from `../backend` directory with `backend.Dockerfile`
- **Frontend**: Builds from `../frontend` directory with `frontend.dev.Dockerfile`
- **Deploy**: All configuration isolated in `deploy/` directory

## Environment Management

### Development Environment
```bash
# Backend environment variables
ENVIRONMENT=development
DEBUG=true
ALLOWED_ORIGINS=http://localhost:3000,http://frontend:3000
DATA_ROOT_DIR=/data

# Frontend environment variables
REACT_APP_API_URL=http://localhost:8000
CHOKIDAR_USEPOLLING=true
```

### Production Environment
```bash
# Create production environment file
cp deploy/.env.production.example deploy/.env.production

# Edit production values
ENVIRONMENT=production
DEBUG=false
ALLOWED_ORIGINS=https://yourdomain.com
DATABASE_URL=postgresql://user:pass@host:5432/db
```

### Environment Variable Precedence
1. Docker compose environment section
2. `.env` files in deploy directory
3. System environment variables
4. Application defaults

## Docker Multi-Stage Builds

### Backend Dockerfile Stages
```dockerfile
# Base stage - common dependencies
FROM python:3.11-slim AS base
# Install system dependencies and Python packages

# Development stage - dev tools + hot reload
FROM base AS development
# Add dev dependencies, enable hot reload

# Production stage - optimized + security
FROM base AS production
# Remove dev tools, create non-root user, optimize
```

### Image Build Commands
```bash
# Build development image
docker build -f deploy/backend.Dockerfile --target development -t mess-ai-backend:dev backend/

# Build production image
docker build -f deploy/backend.Dockerfile --target production -t mess-ai-backend:prod backend/

# Build frontend development image
docker build -f deploy/frontend.dev.Dockerfile -t mess-ai-frontend:dev frontend/
```

## Volume Mount Strategy

### Development Mounts
```yaml
volumes:
  - ../backend:/app/backend     # Hot reload for backend
  - ../frontend/src:/app/src    # Hot reload for frontend
  - ../data:/data               # Persistent ML data
  - /app/node_modules          # Prevent overwriting node_modules
```

### Production Mounts
```yaml
volumes:
  - ../data:/data:rw           # Read-write access to ML data
  - ../logs:/app/logs:rw       # Application logs
  - ./nginx.conf:/etc/nginx/nginx.conf:ro  # Nginx configuration
```

## CI/CD Pipeline Integration

### GitHub Actions Workflow
```yaml
# Reference: Global CLAUDE.md for cross-service communication
# Reference: backend/CLAUDE.local.md for API testing
# Reference: frontend/CLAUDE.local.md for UI testing

name: Deploy MESS-AI
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run backend tests
        run: |
          docker-compose -f deploy/docker-compose.dev.yml up backend -d
          docker-compose -f deploy/docker-compose.dev.yml exec backend pytest
      
      - name: Run frontend tests
        run: |
          docker-compose -f deploy/docker-compose.dev.yml up frontend -d
          docker-compose -f deploy/docker-compose.dev.yml exec frontend npm test -- --coverage
      
      - name: Cleanup
        run: docker-compose -f deploy/docker-compose.dev.yml down

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # SSH to production server
          ssh user@server 'cd mess-ai && git pull && docker-compose -f deploy/docker-compose.prod.yml up -d'
```

### Image Registry Strategy
```bash
# Tag images with commit SHA
docker tag mess-ai-backend:latest mess-ai-backend:$GITHUB_SHA

# Push to registry
docker push your-registry/mess-ai-backend:$GITHUB_SHA
docker push your-registry/mess-ai-backend:latest
```

## Production Deployment

### Server Setup
```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Deployment Commands
```bash
# Clone repository
git clone https://github.com/yourusername/mess-ai.git
cd mess-ai

# Configure environment
cp deploy/.env.production.example deploy/.env.production
# Edit .env.production with production values

# Deploy services
docker-compose -f deploy/docker-compose.prod.yml up -d

# Verify deployment
docker-compose -f deploy/docker-compose.prod.yml ps
curl http://localhost:8000/health
```

## Monitoring & Health Checks

### Health Check Configuration
```yaml
services:
  backend:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
  
  frontend:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Logging Configuration
```yaml
services:
  backend:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service=backend"
```

### Monitoring Commands
```bash
# View service status
docker-compose -f deploy/docker-compose.prod.yml ps

# View logs
docker-compose -f deploy/docker-compose.prod.yml logs -f backend
docker-compose -f deploy/docker-compose.prod.yml logs -f frontend

# View resource usage
docker stats

# Health check status
curl http://localhost:8000/health
curl http://localhost:3000
```

## Security Configuration

### Non-Root User Setup
```dockerfile
# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set permissions
RUN chown -R appuser:appuser /app /data
USER appuser
```

### Network Security
```yaml
networks:
  mess-ai-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Secret Management
```yaml
secrets:
  db_password:
    file: ./secrets/db_password.txt
  api_key:
    external: true
    name: mess_ai_api_key
```

## Troubleshooting

### Common Issues
- **Port conflicts**: Use `lsof -i :8000` to check port usage
- **Volume permissions**: Ensure proper ownership of mounted volumes
- **Memory issues**: Adjust Docker memory limits in compose files
- **Network connectivity**: Check Docker network configuration

### Debug Commands
```bash
# Enter running container
docker-compose -f deploy/docker-compose.dev.yml exec backend bash
docker-compose -f deploy/docker-compose.dev.yml exec frontend sh

# Check container logs
docker-compose -f deploy/docker-compose.dev.yml logs backend
docker-compose -f deploy/docker-compose.dev.yml logs frontend

# Inspect container configuration
docker inspect <container_id>

# Clean rebuild
docker-compose -f deploy/docker-compose.dev.yml down
docker system prune -f
docker-compose -f deploy/docker-compose.dev.yml up --build
```

### Performance Optimization
```bash
# Build with BuildKit for faster builds
DOCKER_BUILDKIT=1 docker build -f deploy/backend.Dockerfile backend/

# Use multi-stage builds to reduce image size
# Cache dependencies separately from application code
# Use .dockerignore to exclude unnecessary files
```

## Related Documentation
- **System Architecture**: See global CLAUDE.md for service communication and data flow
- **Backend Implementation**: See backend/CLAUDE.local.md for API patterns and ML pipeline
- **Frontend Implementation**: See frontend/CLAUDE.local.md for React patterns and build process
- **Automation Scripts**: See scripts/CLAUDE.local.md for deployment automation and utilities