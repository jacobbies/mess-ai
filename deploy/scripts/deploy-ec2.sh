#!/bin/bash
# EC2 deployment script for MESS-AI backend

set -e

echo "🚀 Deploying MESS-AI backend to EC2..."

# Configuration
DEPLOY_DIR="/home/ubuntu/mess-ai"
DATA_DIR="/data"
LOGS_DIR="/home/ubuntu/mess-ai/logs"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Check if running on EC2
if [ ! -f /sys/hypervisor/uuid ] || [ $(head -c 3 /sys/hypervisor/uuid) != "ec2" ]; then
    print_warning "This script is designed for EC2 instances. Proceed with caution."
fi

# Create necessary directories
print_status "Creating directories..."
sudo mkdir -p $DATA_DIR
sudo mkdir -p $LOGS_DIR
sudo chown -R ubuntu:ubuntu $DEPLOY_DIR $LOGS_DIR
sudo chown -R ubuntu:ubuntu $DATA_DIR

# Update system packages
print_status "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    print_status "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    rm get-docker.sh
    print_warning "Docker installed. Please log out and back in for group changes to take effect."
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    print_status "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Pull latest code
print_status "Pulling latest code..."
cd $DEPLOY_DIR
git pull origin main

# Build Docker image
print_status "Building Docker image..."
cd $DEPLOY_DIR/deploy/docker
docker-compose -f docker-compose.prod.yml build

# Stop existing containers
print_status "Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down || true

# Start new containers
print_status "Starting new containers..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for health check
print_status "Waiting for health check..."
sleep 10
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -f http://localhost:8000/health/ready &> /dev/null; then
        print_status "Health check passed! API is ready."
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    print_warning "Health check attempt $ATTEMPT/$MAX_ATTEMPTS failed, retrying..."
    sleep 5
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    print_error "Health check failed after $MAX_ATTEMPTS attempts"
    print_error "Checking logs..."
    docker-compose -f docker-compose.prod.yml logs --tail 50
    exit 1
fi

# Clean up old images
print_status "Cleaning up old Docker images..."
docker image prune -f

# Show status
print_status "Deployment complete! Checking status..."
docker-compose -f docker-compose.prod.yml ps

# Show API endpoint
INSTANCE_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "localhost")
print_status "API available at: http://$INSTANCE_IP:8000"
print_status "API docs available at: http://$INSTANCE_IP:8000/docs"

# Set up log rotation
print_status "Setting up log rotation..."
sudo tee /etc/logrotate.d/mess-ai > /dev/null <<EOF
$LOGS_DIR/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ubuntu ubuntu
}
EOF

print_status "✅ Deployment complete!"