# FRA AI System - Quick Start Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- 50GB+ free disk space for models
- 16GB+ RAM recommended
- NVIDIA GPU (optional, for training)

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/kowshik163/sih.git
cd sih

# Create virtual environment
python3 -m venv fra_env
source fra_env/bin/activate  # On Windows: fra_env\Scripts\activate

# Install basic dependencies
pip install -r "Full prototype/requirements.txt"

# Setup environment variables
cd "Full prototype"
cp .env.example .env
# Edit .env file with your configuration
```

### 2. Quick Test

```bash
# Run basic system test
cd "Full prototype"
python basic_test.py

# Run smoke test (creates dummy data)
python smoke_test.py
```

### 3. Download Models (Optional)

```bash
# Download essential models only
python ../scripts/download_models.py --priority essential

# Or download specific models
python ../scripts/download_models.py --models primary_llm layoutlm deeplabv3
```

### 4. Start API Server

```bash
# Development mode
python run.py --serve

# Or direct FastAPI
uvicorn 3_webgis_backend.api:app --reload --host 0.0.0.0 --port 8000
```

### 5. Access the System

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Base**: http://localhost:8000/api/v1/

## 📖 Detailed Instructions

### Environment Setup

1. **Configure Secrets**:
   ```bash
   python configs/secrets.py setup
   ```

2. **Setup Database (Optional)**:
   ```bash
   # Using Docker
   docker run -d --name fra-postgres \
     -e POSTGRES_DB=fra_gis \
     -e POSTGRES_USER=fra_user \
     -e POSTGRES_PASSWORD=fra_password \
     -e POSTGRES_EXTENSIONS=postgis \
     -p 5432:5432 \
     postgis/postgis:13-3.1
   ```

### Data Pipeline

1. **Download Real Datasets**:
   ```bash
   python ../scripts/download_data.py --priority essential
   ```

2. **Process Data**:
   ```bash
   python run.py --data-pipeline
   ```

### Training

1. **Train Model**:
   ```bash
   # Quick training with dummy data
   python run.py --train --data-dir data/sample_small

   # Full training
   python run.py --train
   ```

2. **With Accelerate (GPU)**:
   ```bash
   accelerate launch --config_file configs/accelerate/single_gpu.yaml \
     2_model_fusion/train_fusion.py
   ```

### Complete Pipeline

```bash
# Run everything
python run.py --complete

# Or step by step
python run.py --download-data
python run.py --download-models  
python run.py --data-pipeline
python run.py --train
python run.py --serve
```

## 🐳 Docker Setup

### Quick Docker Deploy

```bash
# Build and run with docker-compose
docker-compose up --build

# Or individual containers
docker build -t fra-ai-system .
docker run -p 8000:8000 fra-ai-system
```

### Docker with GPU

```bash
# Build GPU version
docker build -f Dockerfile.gpu -t fra-ai-system-gpu .

# Run with GPU support
docker run --gpus all -p 8000:8000 fra-ai-system-gpu
```

## 🧪 Testing

### Run All Tests

```bash
# Basic functionality
python basic_test.py

# Complete system test
python smoke_test.py

# Integration tests
python test_integration.py
```

### API Testing

```bash
# Test OCR endpoint
curl -X POST "http://localhost:8000/api/v1/ocr" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/document.pdf"

# Test health endpoint
curl http://localhost:8000/health
```

## 📁 Directory Structure

```
Full prototype/
├── configs/           # Configuration files
├── 1_data_processing/ # Data pipeline
├── 2_model_fusion/    # Model training/inference  
├── 3_webgis_backend/  # FastAPI backend
├── data/             # Data storage
├── models/           # Downloaded models
├── logs/             # Application logs
├── basic_test.py     # Basic system test
├── smoke_test.py     # Complete smoke test
└── run.py           # Main orchestration script
```

## 🔧 Configuration

### Key Configuration Files

- **`configs/config.json`**: Main system configuration
- **`configs/model_config.py`**: Model definitions and mappings
- **`configs/secrets.py`**: Environment/secrets management
- **`.env`**: Environment variables (copy from `.env.example`)

### Important Environment Variables

```bash
# Required
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret

# Optional
HF_TOKEN=your-hugging-face-token
DB_HOST=localhost
API_PORT=8000
DEBUG=true
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Install missing dependencies
   ```bash
   pip install torch transformers fastapi pillow
   ```

2. **Model Download Fails**: Set HF_TOKEN environment variable
   ```bash
   export HF_TOKEN=your_token_here
   ```

3. **CUDA Errors**: Install PyTorch with CUDA support
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Port Already in Use**: Change API_PORT in .env file

### Performance Optimization

- Use GPU for inference: Set `device: "cuda"` in config.json
- Enable model quantization: Set `load_in_8bit: true` in llm_config
- Use smaller models: Download with `--priority essential`

## 📊 System Status

Check current implementation status:

```bash
python run.py --status
```

This will show:
- ✓ Components implemented
- ⚠️ Components with limitations  
- ✗ Missing components

## 🔄 Development Workflow

1. **Make Changes**: Edit code in relevant directories
2. **Test**: Run `python basic_test.py` 
3. **Integration Test**: Run `python smoke_test.py`
4. **Start Server**: `python run.py --serve`
5. **Test API**: Access http://localhost:8000/docs

## 📞 Support

For issues and questions:
- Check logs in `logs/fra_fusion.log`
- Run diagnostic: `python run.py --status`
- Review configuration: `python configs/secrets.py`

## 🎯 Next Steps

1. **Production Deployment**: Use Docker with proper secrets management
2. **Scale Training**: Use multi-GPU with accelerate
3. **Add Real Data**: Configure your own datasets in `data_sources`
4. **Custom Models**: Add your models to `model_sources` in config.json
