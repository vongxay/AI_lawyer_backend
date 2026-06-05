# =============================================================================
# AI Lawyer Backend - Production Dockerfile
# Multi-stage build for minimal image size
# Python 3.11 slim Debian base
# =============================================================================

FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build dependencies only in builder stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements early for better layer caching
COPY requirements.txt .

# Build wheels in isolated directory
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# =============================================================================
# Development stage (with dev dependencies for testing/debugging)
# =============================================================================
FROM python:3.11-slim as dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=development \
    PYTHONPATH=/app

WORKDIR /app

# Install runtime + build dependencies for dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy wheels from builder
COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install all dependencies (including dev)
RUN pip install --no-cache /wheels/* && rm -rf /wheels && \
    pip install --no-cache -r requirements.txt

# Copy application code with proper ownership
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 8000

# Development uses uvicorn with auto-reload via volume mounts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# =============================================================================
# Production stage (minimal footprint, non-root user)
# =============================================================================
FROM python:3.11-slim as production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=production \
    PYTHONPATH=/app

WORKDIR /app

# Install only runtime dependencies (postgres client for migrations)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy only wheels from builder stage (much smaller footprint)
COPY --from=builder /wheels /wheels

# Install production dependencies only
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Copy application code with proper ownership
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 8000

# Health check for orchestrators (K8s, Docker Swarm, etc.)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production runs with gunicorn + uvicorn workers
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]
