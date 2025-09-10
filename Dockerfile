# FRA AI Fusion System - Multi-stage Docker Build
# Optimized for GPU support and production deployment

# Stage 1: Base image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Kolkata

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    gdal-bin \
    libspatialindex-dev \
    libsqlite3-dev \
    redis-server \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Stage 2: Development image
FROM base as development

# Copy requirements first for better caching
COPY Full\ prototype/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY Full\ prototype/ /app/
COPY scripts/ /app/scripts/

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/logs /app/outputs /app/models

# Set permissions
RUN chmod +x /app/run.py /app/scripts/*.py

# Expose ports
EXPOSE 8000 6379

# Stage 3: Production image
FROM base as production

# Create app user for security
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /app && \
    chown app:app /app

USER app
WORKDIR /app

# Copy requirements and install dependencies
COPY --chown=app:app Full\ prototype/requirements.txt /app/requirements.txt
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# Add user's pip bin to PATH
ENV PATH="/home/app/.local/bin:${PATH}"

# Copy application code
COPY --chown=app:app Full\ prototype/ /app/
COPY --chown=app:app scripts/ /app/scripts/

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/logs /app/outputs /app/models

# Copy configuration files
COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY docker/entrypoint.sh /app/entrypoint.sh

USER root
RUN chmod +x /app/entrypoint.sh /app/run.py /app/scripts/*.py
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["--serve"]
