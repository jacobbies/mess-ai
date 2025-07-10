# Deployment Configuration

This directory contains all deployment logic for the MESS-AI project, following Docker best practices with clean separation of concerns.

## Structure

```
deploy/
├── docker-compose.dev.yml     # Full development stack (backend + frontend)
├── docker-compose.yml         # Backend-only development
├── docker-compose.prod.yml    # Production deployment
├── backend.Dockerfile         # Backend multi-stage image
├── frontend.dev.Dockerfile    # Frontend development image
└── requirements/              # Python dependencies
    ├── requirements.txt       # Base dependencies
    ├── requirements-dev.txt   # Development tools
    └── requirements-production.txt # Production dependencies
```

## Docker Best Practices

### Clean Build Contexts
- **Backend**: Builds from `../backend` directory
- **Frontend**: Builds from `../frontend` directory
- **Deploy**: Configuration isolated in `deploy/` directory

### Multi-Stage Dockerfiles
- **Development**: Hot reload, dev tools, debugging
- **Production**: Optimized, security-hardened, minimal

### Service Communication
- **Internal**: Services communicate via container names
- **External**: Host ports 3000 (frontend) and 8000 (backend)

## Usage

### Full Development Stack
```bash
# Start both backend and frontend
docker-compose -f deploy/docker-compose.dev.yml up --build

# Or use the convenience script
./scripts/dev.sh
```

### Backend Only
```bash
# Start backend only for React development
docker-compose -f deploy/docker-compose.yml up --build
```

### Production Deployment
```bash
# Deploy to production
docker-compose -f deploy/docker-compose.prod.yml up --build -d
```

## Environment Variables

### Development
```bash
# Backend
ENVIRONMENT=development
DEBUG=true
ALLOWED_ORIGINS=http://localhost:3000,http://frontend:3000

# Frontend
REACT_APP_API_URL=http://localhost:8000
CHOKIDAR_USEPOLLING=true
```

### Production
```bash
# Create .env.production file
cp .env.example .env.production
# Edit production values
```

## Volume Mounts

### Development
- **Backend**: `../backend:/app/backend` (hot reload)
- **Frontend**: `../frontend/src:/app/src` (hot reload)
- **Data**: `../data:/data` (persistent ML features)

### Production
- **Data**: `../data:/data:rw` (persistent ML features)
- **Logs**: `../logs:/app/logs:rw` (application logs)

## Health Checks

All services include health checks for monitoring:
- **Backend**: `http://localhost:8000/health`
- **Interval**: 30s
- **Timeout**: 10s
- **Retries**: 3

## Troubleshooting

### Port Conflicts
```bash
# Check what's using ports
lsof -i :3000 :8000

# Kill processes
pkill -f "node.*3000"
pkill -f "python.*8000"
```

### Clean Rebuild
```bash
# Stop and remove containers
docker-compose -f deploy/docker-compose.dev.yml down

# Remove unused images/containers
docker system prune -f

# Rebuild from scratch
docker-compose -f deploy/docker-compose.dev.yml up --build
```

### View Logs
```bash
# All services
docker-compose -f deploy/docker-compose.dev.yml logs -f

# Specific service
docker-compose -f deploy/docker-compose.dev.yml logs -f backend
docker-compose -f deploy/docker-compose.dev.yml logs -f frontend
```