# FRA AI Complete System - Single File Solution

## 🚀 Overview

This is a **single executable Python file** (`fra_ai_complete_system.py`) that provides a comprehensive AI backend system for Forest Rights Act (FRA) monitoring and analysis. It handles everything from model download to production deployment in one unified solution.

## ✨ Features

### 🤖 **Automatic AI Model Management**
- **Auto-downloads** all required LLMs: Llama 3.1 8B, Mistral 7B, Falcon 7B
- **Computer Vision Models**: LayoutLMv3, TrOCR, CLIP, SWIN Transformer
- **Specialized Models**: OCR (Indian languages), NER, Geospatial models
- **Model Distillation**: Automatically distills knowledge from larger to smaller models
- **Fine-tuning**: Fine-tunes models on FRA-specific data

### 📊 **Dataset Integration**
- **Automatic Dataset Download**: FRA legal corpus, Indian NLP datasets, satellite imagery
- **Multi-source Support**: Hugging Face Hub, Git repositories, HTTP downloads
- **Preprocessing Pipeline**: Automated data cleaning and preparation

### 🗄️ **Database Management**  
- **SQLite/PostgreSQL**: Automatic database setup with proper schemas
- **Full CRUD Operations**: Upload, read, update, delete FRA data
- **Spatial Data Support**: PostGIS integration for geospatial queries

### 🌐 **Production API Endpoints**
- **RESTful API**: FastAPI-based with automatic documentation
- **Weight Endpoints**: Model weight information for frontend integration
- **Real-time Analysis**: Satellite data processing, OCR, NER, DSS
- **Maps Integration**: GeoJSON endpoints, tile servers, live data feeds

### 🗺️ **Geospatial & Mapping**
- **Satellite Analysis**: Land cover classification, NDVI/NDWI calculation
- **Asset Detection**: Automatic detection of water bodies, forests, agriculture
- **Live Data Integration**: Real-time satellite feeds and map updates
- **Decision Support System**: AI-powered recommendations for CSS schemes

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- At least 20GB free disk space (for models)
- Internet connection (for downloads)

### Option 1: Complete Setup (Recommended)
```bash
# Navigate to the directory
cd "/path/to/sih_-main/Full prototype"

# Run complete pipeline (setup + train + serve)
python fra_ai_complete_system.py --action all

# This will:
# 1. Auto-install all dependencies
# 2. Download all AI models (Llama 3.1 8B, Mistral, Falcon, etc.)
# 3. Download and process datasets
# 4. Fine-tune models on FRA data
# 5. Perform knowledge distillation  
# 6. Setup database with sample data
# 7. Start production API server on http://localhost:8000
```

### Option 2: Step-by-Step
```bash
# 1. Setup only (install deps, download models, setup DB)
python fra_ai_complete_system.py --action setup

# 2. Train models (fine-tuning + distillation)
python fra_ai_complete_system.py --action train

# 3. Start API server
python fra_ai_complete_system.py --action serve --host 0.0.0.0 --port 8000
```

## 🔧 System Architecture

### File Structure Created
```
Full prototype/
├── fra_ai_complete_system.py    # Single executable file
├── models/                      # Downloaded AI models
│   ├── primary_llm/            # Llama 3.1 8B
│   ├── secondary_llm/          # Mistral 7B  
│   ├── backup_llm/             # Falcon 7B
│   ├── ocr_model/              # TrOCR Large
│   └── distilled_fra_model/    # Distilled model
├── data/                       # Datasets
│   ├── fra_legal/             # FRA legal corpus
│   ├── indic_nlp/             # Indian NLP data
│   └── satellite_data/        # Satellite imagery
├── database/                   # Database files
│   └── fra_system.db          # SQLite database
├── logs/                      # System logs
└── .cache/                    # Model cache
```

## 📡 API Endpoints

Once the system is running, you can access these endpoints:

### Core Endpoints
- `GET /` - System status and available endpoints
- `GET /health` - Health check and system status
- `GET /models/status` - Model loading status
- `GET /weights` - Model weight information for frontend

### FRA Claims Management  
- `POST /claims` - Create new FRA claim
- `GET /claims` - Get all FRA claims
- `PUT /claims/{id}` - Update claim status

### AI Analysis Endpoints
- `POST /satellite/analyze` - Satellite imagery analysis
- `POST /dss/recommend` - Decision Support System recommendations
- `POST /ocr/process` - OCR document processing
- `POST /ner/extract` - Named Entity Recognition

### Maps & Geospatial
- `POST /maps/integration` - Maps integration endpoints
- `GET /geojson/fra-boundaries` - FRA boundaries as GeoJSON
- `GET /live/satellite-updates` - Live satellite data feed

### API Documentation
- `GET /docs` - Interactive API documentation (Swagger)
- `GET /redoc` - Alternative API documentation

## 🔍 Example Usage

### 1. Create FRA Claim
```bash
curl -X POST "http://localhost:8000/claims" \
-H "Content-Type: application/json" \
-d '{
  "village_name": "Rampur Village",
  "patta_holder": "Raja Singh", 
  "claim_type": "Individual Forest Rights",
  "area_hectares": 2.5,
  "coordinates": "28.1234,77.5678",
  "district": "Tehri",
  "state": "Uttarakhand"
}'
```

### 2. Analyze Satellite Data
```bash
curl -X POST "http://localhost:8000/satellite/analyze" \
-H "Content-Type: application/json" \
-d '{
  "coordinates": [28.1234, 77.5678],
  "analysis_type": "comprehensive"
}'
```

### 3. Get DSS Recommendations
```bash
curl -X POST "http://localhost:8000/dss/recommend" \
-H "Content-Type: application/json" \
-d '{
  "village_id": "VIL_001",
  "schemes": ["Jal Jeevan Mission", "MGNREGA"],
  "priority_factors": ["water", "employment"]
}'
```

## 🎯 Frontend Integration

### Connecting Frontend
The system provides special endpoints for frontend integration:

1. **Model Weights**: `GET /weights` - Get model information
2. **Real-time Data**: WebSocket support for live updates
3. **GeoJSON Data**: `GET /geojson/fra-boundaries` for map visualization
4. **Tile Servers**: Configured tile servers for satellite imagery

### Maps Integration
```javascript
// Example frontend integration
fetch('http://localhost:8000/geojson/fra-boundaries')
  .then(response => response.json())
  .then(data => {
    // Add GeoJSON data to your map
    map.addSource('fra-boundaries', { type: 'geojson', data: data });
  });
```

## 🔧 Configuration

### Model Configuration
The system uses models defined in `REQUIRED_MODELS`:
- **Primary LLM**: Llama 3.1 8B Instruct
- **Secondary LLM**: Mistral 7B Instruct
- **Backup LLM**: Falcon 7B Instruct
- **OCR Model**: TrOCR Large
- **Vision Model**: CLIP ViT Large
- **Satellite Model**: SWIN Transformer Large

### Dataset Sources  
Automatically downloads from:
- Hugging Face Hub (legal NER, translations)
- GitHub repositories (Indian NLP corpus, satellite data)
- HTTP sources (OCR datasets, government data)

## 📊 System Monitoring

### Logs
All system activities are logged to `logs/fra_system_YYYYMMDD.log`:
- Model download progress
- Training progress
- API requests
- Error tracking

### Health Monitoring
- `GET /health` - System health status
- Database connectivity check
- Model loading status
- Memory and disk usage

## ⚙️ Advanced Usage

### Custom Configuration
Modify the `SYSTEM_CONFIG` dictionary in the script:
```python
SYSTEM_CONFIG = {
    "models_dir": Path("./custom_models"),
    "data_dir": Path("./custom_data"),
    # ... other settings
}
```

### Adding New Models
Add to `REQUIRED_MODELS`:
```python
REQUIRED_MODELS = {
    "custom_model": "your_org/your-model-name",
    # ... existing models
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Disk Space**
   - Models require ~50GB total
   - Clean up cache: `rm -rf .cache/`

2. **Memory Issues**
   - Reduce batch sizes in training
   - Use CPU instead of GPU: `export CUDA_VISIBLE_DEVICES=""`

3. **Network Issues**
   - Check internet connection
   - Use proxy if required: `export HTTP_PROXY=your_proxy`

4. **Permission Issues**
   - Make script executable: `chmod +x fra_ai_complete_system.py`
   - Check write permissions for data directories

### Debug Mode
Run with verbose logging:
```bash
python fra_ai_complete_system.py --action all --verbose
```

## 🔮 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim
COPY fra_ai_complete_system.py /app/
WORKDIR /app
RUN python fra_ai_complete_system.py --action setup
EXPOSE 8000
CMD ["python", "fra_ai_complete_system.py", "--action", "serve"]
```

### Cloud Deployment
The system is ready for cloud deployment on:
- AWS EC2 (recommended: p3.2xlarge for GPU acceleration)
- Google Cloud Compute Engine
- Azure Virtual Machines
- Any cloud with Python 3.8+ support

## 📈 Performance Optimization

### Hardware Recommendations
- **Minimum**: 16GB RAM, 50GB storage
- **Recommended**: 32GB RAM, 100GB SSD, GPU (RTX 4090/V100)
- **Production**: 64GB RAM, 500GB NVMe SSD, Multi-GPU setup

### Model Optimization
The system automatically:
- Uses 8-bit quantization for large models
- Implements LoRA fine-tuning for efficiency
- Performs knowledge distillation for smaller deployment models

## 📄 License & Attribution

This system integrates multiple open-source models and datasets:
- **Llama 3.1**: Meta AI (Custom License)
- **Mistral 7B**: Apache 2.0
- **Falcon 7B**: Apache 2.0
- **Other models**: Check individual model licenses

## 🤝 Support

For issues and questions:
1. Check the logs in `logs/` directory
2. Review the API documentation at `/docs`
3. Monitor system health at `/health`

---

**🎉 You now have a complete, production-ready FRA AI system running in a single Python file!**

The system handles everything from model download to API deployment, making it perfect for rapid deployment and testing of AI-powered FRA monitoring solutions.