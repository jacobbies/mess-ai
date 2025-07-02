# MESS-AI Backend Deployment

This directory contains all deployment configurations and scripts for the MESS-AI backend.

## Directory Structure

```
deploy/
├── docker/                 # Docker configurations
│   ├── Dockerfile         # Multi-stage Dockerfile
│   ├── docker-compose.yml # Development compose file
│   ├── docker-compose.prod.yml # Production compose file
│   ├── .dockerignore      # Docker ignore patterns
│   └── .env.production.example # Production env template
├── requirements/          # Python dependencies
│   ├── requirements.txt   # Base requirements
│   ├── requirements-dev.txt # Development requirements
│   └── requirements-production.txt # Production requirements
└── scripts/              # Deployment scripts
    └── deploy-ec2.sh     # EC2 deployment script
```

## Local Development

1. Build and run with Docker Compose:
```bash
cd deploy/docker
docker-compose up --build
```

2. API will be available at http://localhost:8000

## Production Deployment on EC2

### Prerequisites

1. EC2 instance (recommended: t3.xlarge or larger with 4+ vCPUs, 16GB+ RAM)
2. Ubuntu 20.04 or 22.04 LTS
3. Security group with port 8000 open
4. Attached EBS volume for data storage (recommended: 100GB+)

### Initial Setup

1. SSH into your EC2 instance:
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/mess-ai.git
cd mess-ai
```

3. Copy and configure production environment:
```bash
cp deploy/docker/.env.production.example deploy/docker/.env.production
nano deploy/docker/.env.production  # Edit with your settings
```

4. Upload your data files to `/data` directory:
```bash
# Option 1: Use SCP
scp -i your-key.pem -r local-data-dir/* ubuntu@your-ec2-ip:/data/

# Option 2: Use AWS S3
aws s3 sync s3://your-bucket/data /data/
```

### Deployment

Run the deployment script:
```bash
cd ~/mess-ai
./deploy/scripts/deploy-ec2.sh
```

The script will:
- Install Docker and Docker Compose if needed
- Build the production Docker image
- Start the backend service
- Set up log rotation
- Run health checks

### Manual Docker Commands

Build production image:
```bash
cd deploy/docker
docker-compose -f docker-compose.prod.yml build
```

Start services:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

View logs:
```bash
docker-compose -f docker-compose.prod.yml logs -f
```

Stop services:
```bash
docker-compose -f docker-compose.prod.yml down
```

### Monitoring

Check service status:
```bash
docker-compose -f docker-compose.prod.yml ps
```

Check API health:
```bash
curl http://localhost:8000/health/ready
```

View metrics:
```bash
curl http://localhost:8000/metrics
```

### Data Structure

Ensure your data is organized as follows on the EC2 instance:

```
/data/
├── smd/
│   └── wav-44/           # Audio files
├── processed/
│   ├── features/
│   │   └── aggregated/   # MERT features
│   └── cache/            # FAISS cache
└── metadata/             # Metadata CSV files
```

### Performance Tuning

1. **Memory**: Adjust Docker memory limits in `docker-compose.prod.yml`
2. **CPU**: Modify CPU limits based on instance type
3. **Cache**: Configure cache TTL in `.env.production`
4. **Workers**: Adjust Gunicorn workers based on CPU cores

### SSL/TLS Setup (Optional)

For HTTPS, use a reverse proxy like Nginx:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Troubleshooting

1. **Container won't start**: Check logs with `docker-compose logs`
2. **Out of memory**: Increase instance size or adjust Docker limits
3. **Slow performance**: Check FAISS index is cached, increase workers
4. **Data not found**: Verify data paths in `.env.production`

### Backup

Regular backups recommended:
```bash
# Backup data
aws s3 sync /data s3://your-backup-bucket/data/

# Backup logs
aws s3 sync /home/ubuntu/mess-ai/logs s3://your-backup-bucket/logs/
```

## CI/CD Integration

For automated deployments, consider:
1. GitHub Actions workflow
2. AWS CodeDeploy
3. Docker Hub for image registry
4. Terraform for infrastructure as code

## Security Best Practices

1. Use IAM roles instead of AWS keys
2. Enable CloudWatch logging
3. Set up ALB with WAF
4. Use Parameter Store for secrets
5. Enable VPC endpoints for S3
6. Regular security updates