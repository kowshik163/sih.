# FRA AI Fusion System - Deployment Guide

This guide covers different deployment options for the FRA AI Fusion System, from development setup to production deployment.

## Quick Start

### Prerequisites

1. **System Requirements:**
   - Python 3.8+ 
   - CUDA-compatible GPU (recommended for training)
   - 16GB+ RAM
   - 100GB+ free disk space
   - Docker and Docker Compose (for containerized deployment)

2. **API Keys:**
   - Hugging Face token (`HF_TOKEN`) for model downloads
   - Optional: Weights & Biases API key for experiment tracking

### 1. Local Development Setup

#### Option A: Direct Python Installation

```bash
# Clone repository
git clone <repository-url>
cd sih_-main

# Setup Python environment
python -m venv fra_env
source fra_env/bin/activate  # On Windows: fra_env\Scripts\activate

# Install dependencies
pip install -r "Full prototype/requirements.txt"

# Setup environment variables
export HF_TOKEN="your_huggingface_token_here"

# Initialize system
cd "Full prototype"
python run.py --setup

# Download models and data
python run.py --download-models
python run.py --download-data

# Run complete pipeline
python run.py --complete
```

#### Option B: Using Scripts

```bash
# Download models only
python run.py --download-models --models layoutlm trocr bert_base

# Download specific datasets
python run.py --download-data --datasets sample_fra_documents village_boundaries

# Process data
python run.py --data-pipeline

# Train model
python run.py --train

# Start API server
python run.py --serve
```

### 2. Docker Development Setup

```bash
# Build and start development environment
docker-compose up fra-dev

# Or with specific GPU
CUDA_VISIBLE_DEVICES=0 docker-compose up fra-dev

# Run specific commands
docker-compose run fra-dev --download-models
docker-compose run fra-dev --train
docker-compose run fra-dev --complete
```

### 3. Production Deployment

#### Using Docker Compose

```bash
# Set environment variables
export HF_TOKEN="your_token"

# Start production services
docker-compose --profile production up -d

# Check status
docker-compose ps
docker-compose logs fra-prod
```

#### Manual Production Setup

```bash
# Build production image
docker build --target production -t fra-fusion:prod .

# Run with GPU support
docker run --gpus all \
  -p 80:8000 \
  -e HF_TOKEN="your_token" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v fra_models:/app/models \
  -v fra_data:/app/data \
  fra-fusion:prod --serve
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | Hugging Face API token | Required for private models |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | `0` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `DATABASE_URL` | PostgreSQL connection | SQLite by default |
| `NODE_ENV` | Environment mode | `development` |

### Model Configuration

Edit `Full prototype/configs/config.json`:

```json
{
  "model_sources": {
    "layoutlm": "microsoft/layoutlmv3-base",
    "custom_model": "your_org/your_model"
  },
  "data_sources": {
    "your_dataset": {
      "type": "huggingface",
      "url": "your_org/your_dataset",
      "description": "Your dataset description"
    }
  }
}
```

## API Endpoints

Once deployed, the API will be available at `http://localhost:8000` with the following endpoints:

### Core Endpoints
- `GET /` - API documentation
- `GET /health` - Health check
- `GET /status` - System status

### Data Processing
- `POST /digitize` - Digitize FRA documents
- `POST /ocr` - Extract text from images
- `POST /ner` - Named entity recognition

### Model Services
- `POST /predict` - General model predictions
- `POST /fusion` - Multimodal fusion predictions
- `POST /dss` - Decision support queries

### WebGIS
- `GET /villages` - Village boundaries
- `GET /satellite` - Satellite imagery
- `POST /analysis` - Spatial analysis

## Training Pipeline

### Staged Training Process

The system uses a multi-stage training pipeline:

1. **Stage 0 - Multimodal Pretraining**
   ```bash
   python run.py --train --stage pretraining
   ```

2. **Stage 1 - Foundation Training**
   ```bash
   python run.py --train --stage foundation
   ```

3. **Stage 2 - Task Alignment**
   ```bash
   python run.py --train --stage alignment
   ```

4. **Stage 3 - Tool Skills**
   ```bash
   python run.py --train --stage tools
   ```

5. **Stage 4 - DSS Fine-tuning**
   ```bash
   python run.py --train --stage dss
   ```

### Knowledge Distillation

Create smaller, deployable models:

```bash
# After training the main model
python 2_model_fusion/distillation.py \
  --teacher-model checkpoints/final_model.pth \
  --student-config configs/student_config.json \
  --output-dir checkpoints/distilled/
```

## Monitoring and Logging

### Development Monitoring

- Logs: `Full prototype/logs/`
- Outputs: `Full prototype/outputs/`
- Checkpoints: `Full prototype/2_model_fusion/checkpoints/`

### Production Monitoring

```bash
# View logs
docker-compose logs -f fra-prod

# Monitor GPU usage
nvidia-smi

# Check system resources
docker stats
```

### Experiment Tracking

Configure Weights & Biases in `config.json`:

```json
{
  "logging": {
    "wandb": {
      "enabled": true,
      "project": "fra-fusion",
      "entity": "your_team"
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```bash
   # Reduce batch size in config.json
   "training": {"batch_size": 2}
   
   # Use gradient checkpointing
   "model": {"gradient_checkpointing": true}
   ```

2. **Model Download Failures**
   ```bash
   # Check HF_TOKEN
   echo $HF_TOKEN
   
   # Manual download
   python scripts/download_models.py --models layoutlm
   ```

3. **CUDA Issues**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Set specific GPU
   export CUDA_VISIBLE_DEVICES=0
   ```

4. **Permission Issues (Docker)**
   ```bash
   # Fix volume permissions
   sudo chown -R $USER:$USER data models outputs
   ```

### Performance Optimization

1. **For Training:**
   - Use mixed precision training
   - Enable gradient checkpointing for large models
   - Use distributed training for multiple GPUs

2. **For Inference:**
   - Use model quantization
   - Enable ONNX export
   - Use TensorRT optimization

## Scaling

### Multi-GPU Training

```bash
# Using accelerate
accelerate config
accelerate launch Full\ prototype/2_model_fusion/train_fusion.py

# Using torchrun
torchrun --nproc_per_node=2 Full\ prototype/2_model_fusion/train_fusion.py
```

### Cloud Deployment

#### AWS ECS/Fargate
```bash
# Build and push to ECR
aws ecr create-repository --repository-name fra-fusion
docker tag fra-fusion:prod <account>.dkr.ecr.<region>.amazonaws.com/fra-fusion:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/fra-fusion:latest
```

#### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/fra-fusion
gcloud run deploy --image gcr.io/PROJECT_ID/fra-fusion --platform managed
```

#### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fra-fusion
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fra-fusion
  template:
    metadata:
      labels:
        app: fra-fusion
    spec:
      containers:
      - name: fra-fusion
        image: fra-fusion:prod
        ports:
        - containerPort: 8000
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: fra-secrets
              key: hf-token
```

## Security

### Production Security Checklist

- [ ] Use environment variables for secrets
- [ ] Enable HTTPS with proper certificates
- [ ] Set up API rate limiting
- [ ] Configure proper CORS policies
- [ ] Use non-root containers
- [ ] Regular security updates
- [ ] Input validation and sanitization
- [ ] Authentication and authorization

### API Security

```json
{
  "security": {
    "jwt_secret": "your-secret-key",
    "jwt_expire_hours": 24,
    "allowed_ips": ["127.0.0.1"],
    "rate_limit": {
      "requests_per_minute": 60
    }
  }
}
```

## Support

For issues and questions:

1. Check logs first: `docker-compose logs fra-prod`
2. Review configuration: `Full prototype/configs/config.json`
3. Validate data paths and model availability
4. Check GPU memory and system resources
5. Verify API endpoints with `/health` and `/status`

## License and Compliance

Ensure compliance with:
- Hugging Face model licenses
- Data usage agreements
- Export control regulations
- Privacy and data protection laws

Remember to review and accept the licenses of all downloaded models and datasets before deployment.
