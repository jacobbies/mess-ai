# Backend microservice Dockerfile
FROM python:3.11-slim AS base

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY ../requirements.txt ./
COPY requirements/requirements-production.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt -r requirements-production.txt \
    && rm -rf ~/.cache/pip

# Production stage
FROM base AS production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application code
COPY . ./backend/

# Create directories and set permissions
RUN mkdir -p /app/logs /data \
    && chown -R appuser:appuser /app /data

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

# Production command with Gunicorn
CMD ["gunicorn", "backend.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# Development stage
FROM base AS development

# Copy dev requirements and install
COPY requirements/requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy all source code
COPY . ./backend/

# Development command with hot reload
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]