#!/bin/bash

# Backend microservice deployment script
set -e

echo "🚀 Deploying mess-ai backend microservice..."

# Configuration
ENVIRONMENT=${1:-production}
IMAGE_NAME="mess-ai-backend"
CONTAINER_NAME="mess-ai-backend"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Build and deploy locally
deploy_local() {
    print_status "Building and running backend microservice locally..."
    
    # Build and run
    docker-compose up -d --build
    
    print_status "Backend microservice started ✓"
    print_status "API: http://localhost:8000"
    print_status "Health: curl http://localhost:8000/health/ready"
}

# Deploy to production
deploy_production() {
    print_status "Deploying backend microservice to production..."
    
    # Stop existing container
    docker-compose -f docker-compose.prod.yml down || true
    
    # Build and start production
    docker-compose -f docker-compose.prod.yml up -d --build
    
    # Wait for health check
    print_status "Waiting for service to be healthy..."
    for i in {1..30}; do
        if curl -f http://localhost:8000/health/ready 2>/dev/null; then
            print_status "Service is healthy ✓"
            break
        fi
        echo "Waiting... ($i/30)"
        sleep 5
    done
    
    # Show status
    docker-compose -f docker-compose.prod.yml ps
    
    print_status "Production deployment completed ✓"
}

# Main
case $ENVIRONMENT in
    "local"|"dev"|"development")
        deploy_local
        ;;
    "prod"|"production")
        deploy_production
        ;;
    *)
        echo "Usage: $0 [local|production]"
        exit 1
        ;;
esac

print_status "🎉 Deployment completed!"