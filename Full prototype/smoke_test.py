#!/usr/bin/env python3
"""
Smoke Test for FRA AI System
Creates dummy data and tests the complete pipeline
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_data():
    """Create minimal dummy data for testing"""
    logger.info("Creating dummy data for smoke test...")
    
    # Create data directories
    data_dir = project_root / "data" / "sample_small"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create dummy FRA documents (images)
    docs_dir = data_dir / "documents"
    docs_dir.mkdir(exist_ok=True)
    
    for i in range(3):
        # Create dummy document image
        img = Image.new('RGB', (800, 1000), color=(255, 255, 255))
        img_path = docs_dir / f"fra_document_{i+1}.jpg"
        img.save(img_path)
        
        # Create corresponding metadata
        metadata = {
            "document_id": f"FRA_DOC_{i+1:03d}",
            "village_name": f"Test Village {i+1}",
            "patta_holder": f"Test Holder {i+1}",
            "claim_type": "Individual Forest Rights" if i % 2 == 0 else "Community Forest Rights",
            "status": "Approved" if i < 2 else "Pending",
            "coordinates": [77.1025 + i*0.01, 28.7041 + i*0.01],
            "area": 2.5 + i*0.5,
            "date": "2023-01-15",
            "district": "Test District",
            "state": "Test State"
        }
        
        metadata_path = docs_dir / f"fra_document_{i+1}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # 2. Create dummy satellite imagery
    satellite_dir = data_dir / "satellite"
    satellite_dir.mkdir(exist_ok=True)
    
    for i in range(3):
        # Create dummy satellite tile (RGB + NIR simulation)
        tile_data = np.random.randint(0, 255, (512, 512, 4), dtype=np.uint8)
        tile_path = satellite_dir / f"tile_{i+1}.tif"
        
        # Save as simple numpy file for now (in real case would be GeoTIFF)
        np.save(tile_path.with_suffix('.npy'), tile_data)
        
        # Tile metadata
        tile_metadata = {
            "tile_id": f"TILE_{i+1:03d}",
            "coordinates": [77.0 + i*0.1, 28.0 + i*0.1, 77.1 + i*0.1, 28.1 + i*0.1],
            "date": "2023-06-15",
            "satellite": "Sentinel-2",
            "cloud_cover": 5.2 + i*2.1,
            "bands": ["B2", "B3", "B4", "B8"]
        }
        
        metadata_path = satellite_dir / f"tile_{i+1}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(tile_metadata, f, indent=2)
    
    # 3. Create dummy training data
    training_data = []
    for i in range(10):
        sample = {
            "document_path": str(docs_dir / f"fra_document_{(i%3)+1}.jpg"),
            "satellite_path": str(satellite_dir / f"tile_{(i%3)+1}.npy"),
            "text": f"Forest Rights Act claim for Village {i+1}. Patta holder: John Doe {i+1}. Area: {2.5 + i*0.3} hectares.",
            "entities": {
                "village_name": f"Village {i+1}",
                "patta_holder": f"John Doe {i+1}",
                "area": f"{2.5 + i*0.3} hectares",
                "claim_type": "IFR" if i % 2 == 0 else "CFR"
            },
            "coordinates": [77.1 + i*0.01, 28.7 + i*0.01],
            "land_use": ["forest", "agriculture", "water"][i % 3]
        }
        training_data.append(sample)
    
    # Save training data
    training_path = data_dir / "training_data.json"
    with open(training_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    logger.info(f"Created dummy data in: {data_dir}")
    return data_dir

def create_dummy_model():
    """Create a minimal model checkpoint for testing"""
    logger.info("Creating dummy model checkpoint...")
    
    checkpoint_dir = project_root / "2_model_fusion" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal model state dict
    dummy_state = {
        'model_state_dict': {
            'encoder.embeddings.weight': torch.randn(1000, 512),
            'encoder.layer.0.attention.self.query.weight': torch.randn(512, 512),
            'encoder.layer.0.attention.self.key.weight': torch.randn(512, 512),
            'encoder.layer.0.attention.self.value.weight': torch.randn(512, 512),
        },
        'epoch': 1,
        'loss': 0.5,
        'config': {
            'hidden_size': 512,
            'num_attention_heads': 8,
            'num_hidden_layers': 6
        }
    }
    
    checkpoint_path = checkpoint_dir / "final_model.pth"
    torch.save(dummy_state, checkpoint_path)
    
    logger.info(f"Created dummy checkpoint at: {checkpoint_path}")
    return checkpoint_path

def test_data_pipeline():
    """Test the data processing pipeline"""
    logger.info("Testing data pipeline...")
    
    try:
        sys.path.append(str(project_root / "1_data_processing"))
        from data_pipeline import EnhancedFRADataProcessor
        
        # Initialize processor
        processor = EnhancedFRADataProcessor("data/sample_small")
        
        # Test OCR processing
        docs_dir = project_root / "data" / "sample_small" / "documents"
        if docs_dir.exists():
            doc_files = list(docs_dir.glob("*.jpg"))
            if doc_files:
                result = processor.process_documents([str(doc_files[0])])
                logger.info(f"OCR test result: {len(result)} documents processed")
        
        logger.info("✓ Data pipeline test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data pipeline test failed: {e}")
        return False

def test_model_loading():
    """Test model loading and inference"""
    logger.info("Testing model loading...")
    
    try:
        sys.path.append(str(project_root))
        from main_fusion_model import EnhancedFRAUnifiedEncoder
        from configs.config import config
        
        # Initialize model
        model = EnhancedFRAUnifiedEncoder(config.config)
        
        # Test forward pass with dummy data
        batch_size = 1
        seq_len = 128
        image_size = 224
        
        dummy_input = {
            'text_tokens': torch.randint(0, 1000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'image_features': torch.randn(batch_size, 3, image_size, image_size),
            'coordinates': torch.randn(batch_size, 2),
            'temporal_features': torch.randn(batch_size, 10, 64)
        }
        
        with torch.no_grad():
            output = model(dummy_input)
            logger.info(f"Model output shape: {output.shape}")
        
        logger.info("✓ Model loading test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    logger.info("Testing API endpoints...")
    
    try:
        from fastapi.testclient import TestClient
        sys.path.append(str(project_root / "3_webgis_backend"))
        from api import app
        
        client = TestClient(app)
        
        # Test health check
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test OCR endpoint with dummy data
        with open(project_root / "data" / "sample_small" / "documents" / "fra_document_1.jpg", "rb") as f:
            files = {"file": ("test.jpg", f, "image/jpeg")}
            response = client.post("/api/v1/ocr", files=files)
            assert response.status_code == 200
        
        logger.info("✓ API endpoints test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ API endpoints test failed: {e}")
        return False

def run_smoke_test():
    """Run complete smoke test"""
    logger.info("="*60)
    logger.info("FRA AI SYSTEM SMOKE TEST")
    logger.info("="*60)
    
    results = {}
    
    # 1. Create dummy data
    try:
        data_dir = create_dummy_data()
        results['dummy_data'] = True
    except Exception as e:
        logger.error(f"Failed to create dummy data: {e}")
        results['dummy_data'] = False
    
    # 2. Create dummy model
    try:
        checkpoint_path = create_dummy_model()
        results['dummy_model'] = True
    except Exception as e:
        logger.error(f"Failed to create dummy model: {e}")
        results['dummy_model'] = False
    
    # 3. Test data pipeline
    results['data_pipeline'] = test_data_pipeline()
    
    # 4. Test model loading
    results['model_loading'] = test_model_loading()
    
    # 5. Test API endpoints
    results['api_endpoints'] = test_api_endpoints()
    
    # Summary
    logger.info("="*60)
    logger.info("SMOKE TEST RESULTS")
    logger.info("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:20}: {status}")
        if result:
            passed += 1
    
    logger.info("="*60)
    logger.info(f"OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All smoke tests passed! System is ready for development.")
        return True
    else:
        logger.warning("⚠️  Some tests failed. Check logs above for details.")
        return False

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
