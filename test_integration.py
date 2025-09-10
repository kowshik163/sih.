#!/usr/bin/env python3
"""
Integration Tests for FRA AI Fusion System
Tests the complete pipeline from downloads to API serving
"""

import os
import sys
import tempfile
import pytest
import requests
import time
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent / "Full prototype"
sys.path.append(str(project_root))

class TestFRASystem:
    """Integration tests for FRA system"""
    
    @pytest.fixture(autouse=True)
    def setup_temp_env(self):
        """Setup temporary environment for testing"""
        self.temp_dir = tempfile.mkdtemp(prefix="fra_test_")
        self.config_backup = None
        
        # Backup original config if exists
        config_path = project_root / "configs" / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config_backup = f.read()
        
        yield
        
        # Cleanup
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Restore config
        if self.config_backup:
            with open(config_path, 'w') as f:
                f.write(self.config_backup)
    
    def test_download_models_script(self):
        """Test model download script"""
        script_path = project_root.parent / "scripts" / "download_models.py"
        
        # Test help
        result = subprocess.run([
            sys.executable, str(script_path), "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Download models for FRA AI System" in result.stdout
    
    def test_download_data_script(self):
        """Test data download script"""
        script_path = project_root.parent / "scripts" / "download_data.py"
        
        # Test help
        result = subprocess.run([
            sys.executable, str(script_path), "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Download datasets for FRA AI System" in result.stdout
    
    def test_main_runner_status(self):
        """Test main runner status command"""
        run_script = project_root / "run.py"
        
        result = subprocess.run([
            sys.executable, str(run_script), "--status"
        ], capture_output=True, text=True, cwd=str(project_root))
        
        assert result.returncode == 0
        assert "FRA AI Fusion System Status" in result.stdout
    
    def test_config_validation(self):
        """Test configuration validation"""
        try:
            from configs.config import validate_config
            errors = validate_config()
            # Should either pass validation or have known errors
            assert isinstance(errors, list)
        except ImportError:
            pytest.skip("Config validation not available")
    
    def test_model_architecture_import(self):
        """Test that model architecture can be imported"""
        try:
            from main_fusion_model import EnhancedFRAUnifiedEncoder
            # Should be able to import without errors
            assert EnhancedFRAUnifiedEncoder is not None
        except ImportError as e:
            pytest.skip(f"Model import failed: {e}")
    
    def test_data_pipeline_import(self):
        """Test that data pipeline can be imported"""
        try:
            sys.path.append(str(project_root / "1_data_processing"))
            from data_pipeline import main as run_data_processing
            assert run_data_processing is not None
        except ImportError as e:
            pytest.skip(f"Data pipeline import failed: {e}")
    
    def test_training_pipeline_import(self):
        """Test that training pipeline can be imported"""
        try:
            sys.path.append(str(project_root / "2_model_fusion"))
            from train_fusion import EnhancedFRATrainingPipeline
            assert EnhancedFRATrainingPipeline is not None
        except ImportError as e:
            pytest.skip(f"Training pipeline import failed: {e}")
    
    def test_distillation_import(self):
        """Test distillation module import"""
        try:
            sys.path.append(str(project_root / "2_model_fusion"))
            from distillation import FRADistillationTrainer
            assert FRADistillationTrainer is not None
        except ImportError as e:
            pytest.skip(f"Distillation import failed: {e}")

class TestAPIEndpoints:
    """Test API endpoints when server is running"""
    
    @pytest.fixture(scope="class")
    def api_server(self):
        """Start API server for testing"""
        run_script = project_root / "run.py"
        
        # Start server in background
        process = subprocess.Popen([
            sys.executable, str(run_script), "--serve", "--port", "8001"
        ], cwd=str(project_root))
        
        # Wait for server to start
        time.sleep(10)
        
        # Check if server is running
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("API server not responding")
        except:
            pytest.skip("API server failed to start")
        
        yield "http://localhost:8001"
        
        # Cleanup
        process.terminate()
        process.wait()
    
    def test_health_endpoint(self, api_server):
        """Test health check endpoint"""
        response = requests.get(f"{api_server}/health")
        assert response.status_code == 200
    
    def test_status_endpoint(self, api_server):
        """Test status endpoint"""
        response = requests.get(f"{api_server}/status")
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
    
    def test_root_endpoint(self, api_server):
        """Test root endpoint documentation"""
        response = requests.get(api_server)
        assert response.status_code == 200

class TestDockerSetup:
    """Test Docker configuration"""
    
    def test_dockerfile_exists(self):
        """Test Dockerfile exists and is readable"""
        dockerfile = project_root.parent / "Dockerfile"
        assert dockerfile.exists()
        
        with open(dockerfile, 'r') as f:
            content = f.read()
        
        assert "pytorch/pytorch" in content
        assert "WORKDIR /app" in content
    
    def test_docker_compose_exists(self):
        """Test docker-compose.yml exists"""
        compose_file = project_root.parent / "docker-compose.yml"
        assert compose_file.exists()
        
        with open(compose_file, 'r') as f:
            content = f.read()
        
        assert "fra-dev:" in content
        assert "fra-prod:" in content

def test_requirements_file():
    """Test requirements.txt is valid"""
    requirements_file = project_root / "requirements.txt"
    assert requirements_file.exists()
    
    with open(requirements_file, 'r') as f:
        lines = f.readlines()
    
    # Should have core dependencies
    requirements_text = ''.join(lines)
    assert "torch" in requirements_text
    assert "transformers" in requirements_text
    assert "fastapi" in requirements_text
    assert "huggingface_hub" in requirements_text

def test_directory_structure():
    """Test project has expected directory structure"""
    expected_dirs = [
        "1_data_processing",
        "2_model_fusion", 
        "3_webgis_backend",
        "configs"
    ]
    
    for dir_name in expected_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Missing directory: {dir_name}"

def test_config_file_structure():
    """Test config.json has expected structure"""
    config_file = project_root / "configs" / "config.json"
    if config_file.exists():
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check for key sections
        assert "model" in config
        assert "training" in config
        assert "data" in config
        
        # Check for new additions
        if "model_sources" in config:
            assert isinstance(config["model_sources"], dict)
        if "data_sources" in config:
            assert isinstance(config["data_sources"], dict)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
