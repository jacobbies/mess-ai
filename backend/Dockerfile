# syntax=docker/dockerfile:1

# ============================================================================
# Stage 1: Builder - Install dependencies and build wheels
# ============================================================================
FROM python:3.11-slim AS builder

# Install uv - fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory for build
WORKDIR /build

# Copy mess-ai library (the dependency) from parent directory
# This gives us the library that backend depends on
# NOTE: Build from parent directory: docker build -f backend/Dockerfile .
COPY mess ./mess-ai/mess
COPY pyproject.toml ./mess-ai/pyproject.toml
COPY README.md ./mess-ai/README.md

# Copy backend application
COPY backend/pyproject.toml ./backend/pyproject.toml
COPY backend/app ./backend/app

# Install both packages into a virtual environment
# This creates a clean, relocatable .venv with all dependencies
WORKDIR /build/backend
RUN uv venv /app/.venv && \
    uv pip install --no-cache /build/mess-ai && \
    uv pip install --no-cache .

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /build/backend/app /app/app

# Set working directory and user
WORKDIR /app
USER appuser

# Expose port
EXPOSE 8000

# Use virtual environment's uvicorn
ENV PATH="/app/.venv/bin:$PATH"

# Health check using built-in urllib
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://localhost:8000/health', timeout=2).read()" || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
