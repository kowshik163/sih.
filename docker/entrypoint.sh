#!/bin/bash
"""
Docker Entrypoint for FRA AI Fusion System
Handles initialization and service startup
"""

set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

log "Starting FRA AI Fusion System..."

# Set up environment variables
export PYTHONPATH="/app:${PYTHONPATH}"

# Create required directories
mkdir -p /app/logs /app/data/raw /app/data/processed /app/outputs /app/models

# Check if HF_TOKEN is provided for model downloads
if [ -n "${HF_TOKEN}" ]; then
    log "HF_TOKEN detected - will be able to download private models"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

# Initialize the system based on arguments
if [ "$1" = "--setup" ]; then
    log "Setting up environment..."
    python /app/run.py --setup
    exit 0
    
elif [ "$1" = "--download-models" ]; then
    log "Downloading models..."
    python /app/run.py --download-models
    exit 0
    
elif [ "$1" = "--download-data" ]; then
    log "Downloading datasets..."
    python /app/run.py --download-data
    exit 0
    
elif [ "$1" = "--complete" ]; then
    log "Running complete pipeline..."
    python /app/run.py --complete
    
elif [ "$1" = "--serve" ]; then
    log "Starting API server..."
    # Check if model exists
    if [ ! -f "/app/2_model_fusion/checkpoints/final_model.pth" ]; then
        log "No trained model found. Starting with demo/mock mode."
    fi
    
    # Start the API server
    python /app/run.py --serve --host 0.0.0.0 --port 8000
    
elif [ "$1" = "--train" ]; then
    log "Starting model training..."
    python /app/run.py --train
    
elif [ "$1" = "--data-pipeline" ]; then
    log "Running data processing pipeline..."
    python /app/run.py --data-pipeline
    
else
    # Default: show status and available commands
    log "FRA AI Fusion System Docker Container"
    log "Available commands:"
    log "  --setup          : Setup environment"
    log "  --download-models: Download required models"
    log "  --download-data  : Download required datasets"
    log "  --data-pipeline  : Run data processing"
    log "  --train          : Train the fusion model"
    log "  --serve          : Start API server (default)"
    log "  --complete       : Run complete pipeline"
    log ""
    log "Environment variables:"
    log "  HF_TOKEN         : Hugging Face token for private models"
    log "  CUDA_VISIBLE_DEVICES: GPU device selection"
    log ""
    python /app/run.py --status
    log "Starting API server by default..."
    python /app/run.py --serve --host 0.0.0.0 --port 8000
fi
