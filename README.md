# 🌲 FRA AI Fusion System - Complete Automation Suite

**Forest Rights Act (FRA) 2006 - AI-Powered Digital Transformation Platform**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

The FRA AI Fusion System is a comprehensive, automated platform that digitizes Forest Rights Act documents, integrates satellite data, and provides AI-powered decision support for tribal welfare departments. This system now features **complete automation** from model downloads to deployment.

### ✨ Key Features

- **🚀 Fully Automated Pipeline**: One-command setup from downloads to deployment
- **🤖 Advanced AI Models**: Multimodal fusion with OCR, NER, and Computer Vision
- **🗺️ WebGIS Integration**: Interactive mapping with satellite imagery
- **📊 Decision Support System**: AI-driven scheme recommendations
- **🐳 Docker Ready**: Complete containerization for easy deployment
- **📱 REST API**: Comprehensive API for all functionalities
- **🔄 Knowledge Distillation**: Model compression for edge deployment

## 🏗️ System Architecture

```
FRA AI Fusion System
├── 📥 Data Ingestion (Automated)
│   ├── Document OCR & NER
│   ├── Satellite Image Processing
│   └── Census Data Integration
├── 🧠 AI/ML Pipeline (Automated)
│   ├── Multimodal Pretraining
│   ├── Foundation Model Training
│   ├── Knowledge Distillation
│   └── Model Deployment
├── 🗺️ WebGIS Backend
│   ├── Spatial Data Management
│   ├── Interactive Mapping
│   └── Asset Visualization
└── 💡 Decision Support System
    ├── Scheme Eligibility Analysis
    ├── Priority Recommendations
    └── Impact Assessment
```

## 🚀 Quick Start (Fully Automated)

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Docker & Docker Compose (for containerized deployment)
- Hugging Face token (for model downloads)

### Option 1: Complete Automated Setup

```bash
# Clone repository
git clone <repository-url>
cd sih_-main

# Set your Hugging Face token
export HF_TOKEN="your_huggingface_token_here"

# Run complete automated pipeline
cd "Full prototype"
python run.py --complete

# That's it! 🎉 
# The system will:
# 1. Setup environment
# 2. Download all required models
# 3. Download and process datasets
# 4. Train the fusion model
# 5. Run evaluations
# 6. Start the API server
```

### Option 2: Docker Deployment (Recommended)

```bash
# Set environment variables
export HF_TOKEN="your_huggingface_token_here"

# Start complete system with Docker
docker-compose up fra-dev

# For production deployment
docker-compose --profile production up -d
```

### Option 3: Step-by-Step Manual Control

```bash
# Setup environment
python run.py --setup

# Download models only
python run.py --download-models

# Download datasets only  
python run.py --download-data

# Process data
python run.py --data-pipeline

# Train model
python run.py --train

# Start API server
python run.py --serve
```

## 📋 Available Commands

| Command | Description |
|---------|-------------|
| `--complete` | **🚀 Run complete automated pipeline** |
| `--setup` | Initialize environment and dependencies |
| `--download-models` | Download all required AI models |
| `--download-data` | Download and prepare datasets |
| `--data-pipeline` | Process raw data for training |
| `--train` | Train the multimodal fusion model |
| `--serve` | Start the API server |
| `--eval` | Evaluate model performance |
| `--status` | Show current system status |

### Advanced Options

```bash
# Download specific models only
python run.py --download-models --models layoutlm trocr bert_base

# Download specific datasets
python run.py --download-data --datasets village_boundaries census_data

# Skip downloads in complete pipeline (if already done)
python run.py --complete --skip-downloads

# Resume training from checkpoint
python run.py --train --resume-from checkpoints/stage_2.pth

# Run with custom host/port
python run.py --serve --host 0.0.0.0 --port 8080
```

## 🔧 Configuration

### Model Sources (Auto-Downloaded)

The system automatically downloads these models:

```json
{
  "model_sources": {
    "layoutlm": "microsoft/layoutlmv3-base",
    "trocr": "microsoft/trocr-base-stage1", 
    "distilgpt2": "distilgpt2",
    "bert_base": "bert-base-uncased",
    "roberta_base": "roberta-base",
    "detr": "facebook/detr-resnet-50",
    "clip": "openai/clip-vit-base-patch32"
  }
}
```

### Data Sources (Auto-Downloaded)

Configure your data sources in `configs/config.json`:

```json
{
  "data_sources": {
    "fra_documents": {
      "type": "http",
      "url": "https://your-domain.com/fra_docs.zip",
      "description": "FRA document samples"
    },
    "village_boundaries": {
      "type": "huggingface", 
      "url": "your_org/village-boundaries",
      "description": "Village boundary shapefiles"
    }
  }
}
```

## 🐳 Docker Deployment

### Development Environment

```bash
# Start all services
docker-compose up

# Individual services
docker-compose up fra-dev        # Main application
docker-compose up redis          # Caching layer
docker-compose up postgres       # Database
```

### Production Deployment

```bash
# Production with load balancing
docker-compose --profile production up -d

# Scale API instances
docker-compose up --scale fra-prod=3
```

### Jupyter Development Environment

```bash
# Start Jupyter for development
docker-compose --profile jupyter up
# Access at http://localhost:8888
```

## 🌐 API Endpoints

Once deployed, access the interactive API documentation at `http://localhost:8000`

### Core Endpoints

- **GET /** - API Documentation (Swagger UI)
- **GET /health** - Health check
- **GET /status** - System status

### Document Processing
- **POST /digitize** - Upload and digitize FRA documents
- **POST /ocr** - Extract text from document images
- **POST /ner** - Named entity recognition on text

### AI Model Services
- **POST /predict** - General model predictions
- **POST /fusion** - Multimodal fusion inference
- **POST /dss** - Decision support queries

### WebGIS Services
- **GET /villages/{id}** - Village boundary data
- **GET /satellite** - Satellite imagery tiles
- **POST /analysis** - Spatial analysis requests

## 🧠 Training Pipeline

### Multi-Stage Training (Automated)

The system uses a sophisticated 5-stage training process:

1. **Stage 0 - Multimodal Pretraining** (15 epochs)
   - Cross-modal alignment
   - Contrastive learning
   - Masked language modeling

2. **Stage 1 - Foundation Training** (10 epochs) 
   - Task-specific fine-tuning
   - Multi-task learning
   - Knowledge graph integration

3. **Stage 2 - Alignment Training** (8 epochs)
   - Human preference alignment
   - RLHF integration
   - Safety fine-tuning

4. **Stage 3 - Tool Skills** (5 epochs)
   - API calling capabilities
   - SQL generation
   - WebGIS integration

5. **Stage 4 - DSS Specialization** (5 epochs)
   - Decision support optimization
   - Scheme recommendation
   - Policy analysis

### Knowledge Distillation

Create smaller, deployable models:

```bash
# After training, create a compressed model
python 2_model_fusion/distillation.py \
  --teacher-model checkpoints/final_model.pth \
  --compression-ratio 4x \
  --output-dir checkpoints/distilled/
```

## 📊 Monitoring & Evaluation

### Built-in Metrics

- **OCR Accuracy**: Character and word-level accuracy
- **NER F1 Score**: Named entity recognition performance  
- **Segmentation mIoU**: Satellite image segmentation quality
- **SQL Accuracy**: Generated query correctness
- **DSS Precision**: Decision support recommendation accuracy

### Logging & Tracking

```bash
# View real-time logs
tail -f logs/fra_fusion.log

# Monitor training progress
tensorboard --logdir logs/tensorboard

# For Docker deployments
docker-compose logs -f fra-prod
```

## 🔐 Security & Compliance

### Environment Variables

```bash
export HF_TOKEN="your_huggingface_token"          # Required for model downloads
export CUDA_VISIBLE_DEVICES="0,1"                # GPU selection
export WANDB_API_KEY="your_wandb_key"            # Optional: experiment tracking
export DATABASE_URL="postgresql://..."            # Optional: production database
```

### Production Security

- JWT-based authentication
- CORS policy configuration
- Rate limiting
- Input validation and sanitization
- HTTPS support with certificates

## 🚀 Advanced Features

### Distributed Training

```bash
# Multi-GPU training with accelerate
accelerate config
accelerate launch Full\ prototype/2_model_fusion/train_fusion.py

# Or with torchrun
torchrun --nproc_per_node=2 Full\ prototype/2_model_fusion/train_fusion.py
```

### Model Optimization

```bash
# Quantization for inference speed
python run.py --train --quantize

# ONNX export for deployment
python run.py --export-onnx --model-path checkpoints/final_model.pth
```

### Cloud Deployment

Supports deployment on:
- ☁️ **AWS ECS/Fargate**
- ☁️ **Google Cloud Run** 
- ☁️ **Azure Container Instances**
- ☁️ **Kubernetes clusters**

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed cloud deployment guides.

## 📁 Project Structure

```
sih_-main/
├── 📄 Full prototype/              # Main application
│   ├── 1_data_processing/          # Data ingestion & preprocessing
│   ├── 2_model_fusion/            # AI model training & inference
│   │   └── distillation.py        # 🆕 Knowledge distillation
│   ├── 3_webgis_backend/          # WebGIS API server
│   ├── configs/                   # Configuration files
│   ├── main_fusion_model.py       # Core model architecture
│   └── run.py                     # 🆕 Main automated runner
├── 📄 scripts/                    # 🆕 Automation scripts
│   ├── download_models.py         # Model download automation
│   └── download_data.py           # Dataset download automation
├── 📄 docker/                     # 🆕 Docker configuration
├── 🐳 Dockerfile                  # Container definition
├── 🐳 docker-compose.yml          # Multi-service orchestration
├── 📋 DEPLOYMENT.md               # 🆕 Deployment guide
└── 🧪 test_integration.py         # 🆕 Integration tests
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
python test_integration.py

# Code formatting
black Full\ prototype/ scripts/
flake8 Full\ prototype/ scripts/
```

## 📈 Roadmap

### Completed ✅
- ✅ Automated model downloads
- ✅ Automated dataset ingestion  
- ✅ Complete pipeline orchestration
- ✅ Knowledge distillation
- ✅ Docker containerization
- ✅ Production deployment guides

### Coming Soon 🚧
- 🚧 Real-time satellite data integration
- 🚧 Mobile app for field agents
- 🚧 Blockchain-based document verification
- 🚧 Advanced visualization dashboards
- 🚧 Multi-language support
- 🚧 Edge device deployment

## 🏆 Recognition

This system addresses the **Smart India Hackathon (SIH) 2024** problem statement for FRA digitization and was built to provide a complete, production-ready solution for tribal welfare departments across India.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ministry of Tribal Affairs** for problem definition
- **Hugging Face** for model hosting and APIs  
- **PyTorch** community for ML framework
- **FastAPI** for web framework
- **Docker** for containerization support

## 📞 Support

For technical support, deployment assistance, or feature requests:

- 📧 **Email**: [support@fra-fusion.ai](mailto:support@fra-fusion.ai)
- 💬 **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 **Documentation**: [Full Documentation](https://fra-fusion.readthedocs.io)
- 🚀 **Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)

---

**🌲 Empowering Forest Rights with AI - Built for India's Tribal Communities 🇮🇳**
