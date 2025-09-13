#!/usr/bin/env python3
"""
Controlled Model Lifecycle Management System
Download → Train → Finetune → Distill → Store → Serve Pipeline
"""

import os
import json
import torch
import logging
import requests
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import shutil
from huggingface_hub import snapshot_download, login, HfApi
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelStage(Enum):
    """Model lifecycle stages"""
    DOWNLOAD = "download"
    TRAIN = "train"
    FINETUNE = "finetune" 
    DISTILL = "distill"
    STORE = "store"
    SERVE = "serve"
    FAILED = "failed"

@dataclass
class ModelInfo:
    """Model information and metadata"""
    name: str
    hf_model_id: str
    local_path: str
    stage: ModelStage
    size_gb: float
    requires_auth: bool = False
    download_time: Optional[str] = None
    train_time: Optional[str] = None
    finetune_time: Optional[str] = None
    distill_time: Optional[str] = None
    final_size_gb: Optional[float] = None
    endpoints: List[str] = None
    checksum: Optional[str] = None

class ModelLifecycleManager:
    """Complete model lifecycle management"""
    
    def __init__(self, base_dir: str = "models_lifecycle"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        
        # Model registry
        self.registry_file = self.base_dir / "model_registry.json"
        self.models = {}
        self.load_registry()
        
        # HuggingFace setup
        self.hf_api = None
        self.hf_token = None
        self.setup_hf_auth()
        
        # Define model configurations
        self.model_configs = {
            # Primary LLMs
            "llama3.1-8b": ModelInfo(
                name="llama3.1-8b",
                hf_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                local_path=str(self.base_dir / "downloads" / "llama3.1-8b"),
                stage=ModelStage.DOWNLOAD,
                size_gb=16.0,
                requires_auth=True,
                endpoints=["/predict/llm/llama", "/predict/fusion/llama"]
            ),
            "mistral-7b": ModelInfo(
                name="mistral-7b", 
                hf_model_id="mistralai/Mistral-7B-Instruct-v0.3",
                local_path=str(self.base_dir / "downloads" / "mistral-7b"),
                stage=ModelStage.DOWNLOAD,
                size_gb=14.0,
                requires_auth=False,
                endpoints=["/predict/llm/mistral", "/predict/fusion/mistral"]
            ),
            "falcon-7b": ModelInfo(
                name="falcon-7b",
                hf_model_id="tiiuae/falcon-7b-instruct", 
                local_path=str(self.base_dir / "downloads" / "falcon-7b"),
                stage=ModelStage.DOWNLOAD,
                size_gb=13.5,
                requires_auth=False,
                endpoints=["/predict/llm/falcon", "/predict/fusion/falcon"]
            ),
            
            # Vision/OCR Models
            "trocr-base": ModelInfo(
                name="trocr-base",
                hf_model_id="microsoft/trocr-base-handwritten",
                local_path=str(self.base_dir / "downloads" / "trocr-base"),
                stage=ModelStage.DOWNLOAD,
                size_gb=1.5,
                requires_auth=False,
                endpoints=["/predict/ocr", "/predict/document"]
            ),
            "layoutlmv3": ModelInfo(
                name="layoutlmv3",
                hf_model_id="microsoft/layoutlmv3-base",
                local_path=str(self.base_dir / "downloads" / "layoutlmv3"),
                stage=ModelStage.DOWNLOAD,
                size_gb=1.2,
                requires_auth=False,
                endpoints=["/predict/layout", "/predict/document"]
            ),
            "clip-vit": ModelInfo(
                name="clip-vit",
                hf_model_id="openai/clip-vit-base-patch32",
                local_path=str(self.base_dir / "downloads" / "clip-vit"),
                stage=ModelStage.DOWNLOAD,
                size_gb=0.6,
                requires_auth=False,
                endpoints=["/predict/vision", "/predict/satellite"]
            ),
            
            # Specialized Models  
            "swin-transformer": ModelInfo(
                name="swin-transformer",
                hf_model_id="microsoft/swinv2-base-patch4-window16-256",
                local_path=str(self.base_dir / "downloads" / "swin-transformer"),
                stage=ModelStage.DOWNLOAD,
                size_gb=0.8,
                requires_auth=False,
                endpoints=["/predict/satellite", "/predict/vision"]
            )
        }
    
    def setup_directories(self):
        """Create directory structure"""
        directories = [
            "downloads",           # Raw model downloads
            "training",           # Training workspace
            "finetuned",         # Finetuned models
            "distilled",         # Distilled models
            "production",        # Final production models
            "checkpoints",       # Training checkpoints
            "exports",           # Exported models (ONNX, TorchScript)
            "logs",              # Training/process logs
            "cache"              # Temporary cache
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model lifecycle directories created at {self.base_dir}")
    
    def setup_hf_auth(self):
        """Setup HuggingFace authentication"""
        # Try multiple authentication methods
        auth_methods = [
            lambda: os.getenv("HUGGINGFACE_HUB_TOKEN"),
            lambda: os.getenv("HF_TOKEN"),
            lambda: self._read_hf_token_file(),
            lambda: self._interactive_hf_login()
        ]
        
        for method in auth_methods:
            try:
                token = method()
                if token:
                    login(token=token, add_to_git_credential=True)
                    self.hf_token = token
                    self.hf_api = HfApi(token=token)
                    logger.info("HuggingFace authentication successful")
                    return
            except Exception as e:
                logger.debug(f"Auth method failed: {e}")
                continue
        
        logger.warning("HuggingFace authentication not configured. Gated models may fail.")
    
    def _read_hf_token_file(self) -> Optional[str]:
        """Read HF token from file"""
        token_paths = [
            Path.home() / ".huggingface" / "token",
            Path.home() / ".cache" / "huggingface" / "token",
            Path(".hf_token")
        ]
        
        for path in token_paths:
            if path.exists():
                return path.read_text().strip()
        return None
    
    def _interactive_hf_login(self) -> Optional[str]:
        """Interactive HuggingFace login"""
        try:
            print("\\n🔐 HuggingFace Authentication Required")
            print("Some models (like Llama 3.1) require authentication.")
            print("Please visit: https://huggingface.co/settings/tokens")
            print("Create a token and paste it here:")
            
            token = input("Enter HF Token (or press Enter to skip): ").strip()
            if token:
                # Test token validity
                login(token=token, add_to_git_credential=True)
                return token
        except KeyboardInterrupt:
            print("\\nSkipping HuggingFace authentication.")
        except Exception as e:
            logger.error(f"Interactive login failed: {e}")
        
        return None
    
    def download_model(self, model_name: str, force: bool = False) -> bool:
        """Download model with automatic retry and authentication"""
        if model_name not in self.model_configs:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        model_info = self.model_configs[model_name]
        
        # Check if already downloaded
        if not force and Path(model_info.local_path).exists():
            logger.info(f"Model {model_name} already downloaded")
            return True
        
        logger.info(f"Downloading {model_name} ({model_info.size_gb}GB)...")
        
        # Handle authentication for gated models
        if model_info.requires_auth and not self.hf_token:
            logger.error(f"Model {model_name} requires authentication but no token available")
            return False
        
        try:
            start_time = time.time()
            
            # Download with progress
            local_path = snapshot_download(
                repo_id=model_info.hf_model_id,
                local_dir=model_info.local_path,
                token=self.hf_token if model_info.requires_auth else None,
                resume_download=True,
                local_dir_use_symlinks=False
            )
            
            # Verify download
            if not self._verify_download(model_name):
                logger.error(f"Download verification failed for {model_name}")
                return False
            
            # Update model info
            download_time = time.time() - start_time
            model_info.download_time = datetime.now().isoformat()
            model_info.stage = ModelStage.TRAIN
            
            logger.info(f"Downloaded {model_name} in {download_time:.2f}s")
            self.save_registry()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            model_info.stage = ModelStage.FAILED
            self.save_registry()
            return False
    
    def train_model(self, model_name: str, config: Dict[str, Any] = None) -> bool:
        """Train/Finetune model for FRA tasks"""
        if model_name not in self.model_configs:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        model_info = self.model_configs[model_name]
        
        if model_info.stage != ModelStage.TRAIN:
            logger.error(f"Model {model_name} not ready for training. Current stage: {model_info.stage}")
            return False
        
        logger.info(f"Starting training for {model_name}...")
        
        try:
            # Prepare training config
            training_config = self._get_training_config(model_name, config)
            
            # Create training workspace
            training_dir = self.base_dir / "training" / model_name
            training_dir.mkdir(parents=True, exist_ok=True)
            
            # Run training script
            success = self._run_training_pipeline(model_name, training_config)
            
            if success:
                model_info.stage = ModelStage.FINETUNE
                model_info.train_time = datetime.now().isoformat()
                logger.info(f"Training completed for {model_name}")
            else:
                model_info.stage = ModelStage.FAILED
                logger.error(f"Training failed for {model_name}")
            
            self.save_registry()
            return success
            
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}")
            model_info.stage = ModelStage.FAILED
            self.save_registry()
            return False
    
    def finetune_model(self, model_name: str) -> bool:
        """Finetune model with FRA-specific data"""
        model_info = self.model_configs[model_name]
        
        if model_info.stage != ModelStage.FINETUNE:
            logger.info(f"Skipping finetune for {model_name} - stage: {model_info.stage}")
            return True
        
        logger.info(f"Finetuning {model_name} for FRA tasks...")
        
        try:
            # Run FRA-specific finetuning
            success = self._run_finetuning_pipeline(model_name)
            
            if success:
                model_info.stage = ModelStage.DISTILL
                model_info.finetune_time = datetime.now().isoformat()
                
                # Move to finetuned directory
                finetuned_path = self.base_dir / "finetuned" / model_name
                self._move_model_to_stage(model_name, finetuned_path)
                
            self.save_registry()
            return success
            
        except Exception as e:
            logger.error(f"Finetuning failed for {model_name}: {e}")
            return False
    
    def distill_model(self, model_name: str) -> bool:
        """Create distilled version of the model"""
        model_info = self.model_configs[model_name]
        
        if model_info.stage != ModelStage.DISTILL:
            logger.info(f"Skipping distill for {model_name} - stage: {model_info.stage}")
            return True
        
        logger.info(f"Distilling {model_name}...")
        
        try:
            # Run distillation process
            success = self._run_distillation_pipeline(model_name)
            
            if success:
                model_info.stage = ModelStage.STORE
                model_info.distill_time = datetime.now().isoformat()
                
                # Calculate final size
                distilled_path = self.base_dir / "distilled" / model_name
                model_info.final_size_gb = self._calculate_model_size(distilled_path)
                
            self.save_registry()
            return success
            
        except Exception as e:
            logger.error(f"Distillation failed for {model_name}: {e}")
            return False
    
    def store_production_model(self, model_name: str) -> bool:
        """Store final production-ready model"""
        model_info = self.model_configs[model_name]
        
        if model_info.stage != ModelStage.STORE:
            logger.info(f"Model {model_name} not ready for production storage - stage: {model_info.stage}")
            return True
        
        try:
            # Copy to production directory
            production_path = self.base_dir / "production" / model_name
            source_path = self.base_dir / "distilled" / model_name
            
            if source_path.exists():
                shutil.copytree(source_path, production_path, dirs_exist_ok=True)
            
            # Generate checksum
            model_info.checksum = self._generate_checksum(production_path)
            model_info.stage = ModelStage.SERVE
            
            # Create endpoint configuration
            self._create_endpoint_config(model_name)
            
            logger.info(f"Model {model_name} ready for production at {production_path}")
            self.save_registry()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store production model {model_name}: {e}")
            return False
    
    def run_full_lifecycle(self, model_names: List[str] = None, parallel: bool = False) -> Dict[str, bool]:
        """Run complete lifecycle for specified models"""
        if model_names is None:
            model_names = list(self.model_configs.keys())
        
        results = {}
        
        logger.info(f"Starting full lifecycle for models: {model_names}")
        
        if parallel:
            # Run in parallel (for independent models)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = {executor.submit(self._run_single_lifecycle, name): name for name in model_names}
                for future in concurrent.futures.as_completed(futures):
                    model_name = futures[future]
                    results[model_name] = future.result()
        else:
            # Sequential processing
            for model_name in model_names:
                results[model_name] = self._run_single_lifecycle(model_name)
        
        logger.info(f"Lifecycle completed. Results: {results}")
        return results
    
    def _run_single_lifecycle(self, model_name: str) -> bool:
        """Run complete lifecycle for a single model"""
        steps = [
            ("Download", self.download_model),
            ("Train", self.train_model), 
            ("Finetune", self.finetune_model),
            ("Distill", self.distill_model),
            ("Store", self.store_production_model)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"{model_name}: Starting {step_name}")
            
            if not step_func(model_name):
                logger.error(f"{model_name}: Failed at {step_name}")
                return False
                
            logger.info(f"{model_name}: Completed {step_name}")
        
        return True
    
    def get_production_models(self) -> Dict[str, ModelInfo]:
        """Get all production-ready models"""
        production_models = {}
        
        for name, model_info in self.model_configs.items():
            if model_info.stage == ModelStage.SERVE:
                production_models[name] = model_info
        
        return production_models
    
    def get_model_endpoints(self) -> Dict[str, str]:
        """Get endpoint mappings for production models"""
        endpoints = {}
        production_models = self.get_production_models()
        
        for model_name, model_info in production_models.items():
            model_path = self.base_dir / "production" / model_name
            for endpoint in model_info.endpoints:
                endpoints[endpoint] = str(model_path)
        
        return endpoints
    
    # Helper methods
    def _verify_download(self, model_name: str) -> bool:
        """Verify model download integrity"""
        model_path = Path(self.model_configs[model_name].local_path)
        
        # Check if key files exist
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        
        for required_file in required_files:
            file_path = model_path / required_file
            if not file_path.exists():
                # Try alternative patterns
                alternatives = list(model_path.glob(f"*{required_file}*")) or list(model_path.glob(f"{required_file.split('.')[0]}*"))
                if not alternatives:
                    logger.debug(f"Missing {required_file} for {model_name}")
        
        # Basic size check
        total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        
        expected_size = self.model_configs[model_name].size_gb
        if size_gb < expected_size * 0.8:  # Allow 20% variance
            logger.warning(f"Model {model_name} size {size_gb:.1f}GB less than expected {expected_size}GB")
            return False
        
        return True
    
    def _get_training_config(self, model_name: str, user_config: Dict = None) -> Dict[str, Any]:
        """Get training configuration for model"""
        base_config = {
            "output_dir": str(self.base_dir / "training" / model_name),
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "learning_rate": 5e-5,
            "warmup_ratio": 0.1,
            "logging_steps": 100,
            "save_steps": 1000,
            "eval_steps": 500,
            "fp16": True,
            "dataloader_num_workers": 4
        }
        
        if user_config:
            base_config.update(user_config)
        
        return base_config
    
    def _run_training_pipeline(self, model_name: str, config: Dict[str, Any]) -> bool:
        """Execute training pipeline"""
        # This would integrate with main_fusion_model.py training
        try:
            # Simulate training process
            logger.info(f"Training {model_name} with config: {config}")
            time.sleep(2)  # Placeholder for actual training time
            return True
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return False
    
    def _run_finetuning_pipeline(self, model_name: str) -> bool:
        """Execute FRA-specific finetuning"""
        try:
            logger.info(f"Finetuning {model_name} for FRA tasks")
            time.sleep(1)  # Placeholder
            return True
        except Exception as e:
            logger.error(f"Finetuning pipeline failed: {e}")
            return False
    
    def _run_distillation_pipeline(self, model_name: str) -> bool:
        """Execute model distillation"""
        try:
            logger.info(f"Distilling {model_name}")
            time.sleep(1)  # Placeholder  
            return True
        except Exception as e:
            logger.error(f"Distillation pipeline failed: {e}")
            return False
    
    def _move_model_to_stage(self, model_name: str, target_path: Path):
        """Move model between lifecycle stages"""
        # Implementation for moving models between directories
        target_path.mkdir(parents=True, exist_ok=True)
    
    def _calculate_model_size(self, model_path: Path) -> float:
        """Calculate model size in GB"""
        if not model_path.exists():
            return 0.0
        
        total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        return total_size / (1024**3)
    
    def _generate_checksum(self, model_path: Path) -> str:
        """Generate checksum for model verification"""
        hasher = hashlib.sha256()
        
        for file_path in sorted(model_path.rglob('*')):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _create_endpoint_config(self, model_name: str):
        """Create endpoint configuration for model"""
        model_info = self.model_configs[model_name]
        endpoint_config = {
            'model_name': model_name,
            'model_path': str(self.base_dir / "production" / model_name),
            'endpoints': model_info.endpoints,
            'checksum': model_info.checksum,
            'created': datetime.now().isoformat()
        }
        
        config_path = self.base_dir / "production" / model_name / "endpoint_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(endpoint_config, f, indent=2)
    
    def load_registry(self):
        """Load model registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)
                    
                # Reconstruct ModelInfo objects
                for name, data in registry_data.get('models', {}).items():
                    if name in self.model_configs:
                        # Update existing config with saved data
                        for key, value in data.items():
                            if hasattr(self.model_configs[name], key):
                                setattr(self.model_configs[name], key, value)
                                
                logger.info(f"Loaded registry with {len(registry_data.get('models', {}))} models")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
    
    def save_registry(self):
        """Save model registry to disk"""
        registry_data = {
            'last_updated': datetime.now().isoformat(),
            'models': {name: asdict(info) for name, info in self.model_configs.items()},
            'total_models': len(self.model_configs)
        }
        
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        stages_count = {}
        total_size_gb = 0
        
        for model_info in self.model_configs.values():
            stage = model_info.stage.value
            stages_count[stage] = stages_count.get(stage, 0) + 1
            
            if model_info.final_size_gb:
                total_size_gb += model_info.final_size_gb
            else:
                total_size_gb += model_info.size_gb
        
        return {
            'total_models': len(self.model_configs),
            'stages_breakdown': stages_count,
            'total_size_gb': round(total_size_gb, 2),
            'production_ready': len(self.get_production_models()),
            'available_endpoints': len(self.get_model_endpoints()),
            'last_updated': datetime.now().isoformat()
        }

# Global instance
lifecycle_manager = ModelLifecycleManager()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Lifecycle Management")
    parser.add_argument("--action", choices=["download", "train", "full", "status"], 
                       default="status", help="Action to perform")
    parser.add_argument("--models", nargs="*", help="Specific models to process")
    parser.add_argument("--parallel", action="store_true", help="Run in parallel")
    
    args = parser.parse_args()
    
    if args.action == "status":
        status = lifecycle_manager.get_status_report()
        print(json.dumps(status, indent=2))
    
    elif args.action == "download":
        models = args.models or list(lifecycle_manager.model_configs.keys())
        for model in models:
            lifecycle_manager.download_model(model)
    
    elif args.action == "train":
        models = args.models or list(lifecycle_manager.model_configs.keys())
        for model in models:
            lifecycle_manager.train_model(model)
    
    elif args.action == "full":
        models = args.models or list(lifecycle_manager.model_configs.keys())
        results = lifecycle_manager.run_full_lifecycle(models, args.parallel)
        print(f"Results: {results}")