#!/usr/bin/env python3
"""
Model Weights Management System
Handles loading and managing trained model weights that will be generated after training
"""

import os
import torch
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelWeightsManager:
    """Manages trained model weights and their metadata"""
    
    def __init__(self, base_models_dir: str = None):
        # Default model storage paths (will be created by training pipeline)
        if base_models_dir is None:
            self.base_models_dir = Path("trained_models")
        else:
            self.base_models_dir = Path(base_models_dir)
        
        # Expected model weight paths after training
        self.model_paths = {
            'fusion_model': self.base_models_dir / "fusion" / "best_fusion_model.pth",
            'fusion_tokenizer': self.base_models_dir / "fusion" / "tokenizer",
            'fusion_config': self.base_models_dir / "fusion" / "config.json",
            
            # Individual model components
            'llama_adapter': self.base_models_dir / "components" / "llama_adapter.pth",
            'mistral_adapter': self.base_models_dir / "components" / "mistral_adapter.pth", 
            'falcon_adapter': self.base_models_dir / "components" / "falcon_adapter.pth",
            'trocr_finetuned': self.base_models_dir / "components" / "trocr_finetuned.pth",
            'layoutlm_finetuned': self.base_models_dir / "components" / "layoutlm_finetuned.pth",
            
            # Task-specific models
            'dss_model': self.base_models_dir / "tasks" / "dss_model.pth",
            'ner_model': self.base_models_dir / "tasks" / "ner_model.pth",
            'satellite_model': self.base_models_dir / "tasks" / "satellite_analysis.pth",
            'scheme_recommender': self.base_models_dir / "tasks" / "scheme_recommender.pth",
            
            # Distilled models (smaller, faster)
            'fusion_distilled': self.base_models_dir / "distilled" / "fusion_distilled.pth",
            'mobile_model': self.base_models_dir / "distilled" / "mobile_model.pth"
        }
        
        # Model metadata storage
        self.metadata_path = self.base_models_dir / "model_registry.json"
        self.loaded_models = {}
        self.model_metadata = {}
        
        # Create directories if they don't exist
        self._create_model_directories()
        
    def _create_model_directories(self):
        """Create model storage directories"""
        directories = [
            self.base_models_dir / "fusion",
            self.base_models_dir / "components", 
            self.base_models_dir / "tasks",
            self.base_models_dir / "distilled",
            self.base_models_dir / "checkpoints",
            self.base_models_dir / "exports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created model directory: {directory}")
    
    def register_trained_model(self, model_name: str, model_path: str, 
                             metadata: Dict[str, Any]):
        """Register a newly trained model"""
        self.model_paths[model_name] = Path(model_path)
        self.model_metadata[model_name] = {
            'path': str(model_path),
            'trained_at': datetime.now().isoformat(),
            'size_mb': self._get_model_size(model_path),
            'metadata': metadata
        }
        self._save_registry()
        logger.info(f"Registered model: {model_name} at {model_path}")
    
    def load_model_weights(self, model_name: str, device: str = 'cpu') -> Optional[torch.nn.Module]:
        """Load trained model weights"""
        if model_name in self.loaded_models:
            logger.info(f"Using cached model: {model_name}")
            return self.loaded_models[model_name]
        
        model_path = self.model_paths.get(model_name)
        if not model_path or not model_path.exists():
            logger.warning(f"Model weights not found: {model_name} at {model_path}")
            return None
        
        try:
            # Load model weights
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model_weights = checkpoint['model_state_dict']
                    metadata = checkpoint.get('metadata', {})
                elif 'state_dict' in checkpoint:
                    model_weights = checkpoint['state_dict']
                    metadata = checkpoint.get('metadata', {})
                else:
                    model_weights = checkpoint
                    metadata = {}
            else:
                model_weights = checkpoint
                metadata = {}
            
            # Store metadata
            self.model_metadata[model_name] = metadata
            
            logger.info(f"Successfully loaded model weights: {model_name}")
            return model_weights
            
        except Exception as e:
            logger.error(f"Failed to load model weights {model_name}: {e}")
            return None
    
    def get_best_available_model(self, task_type: str) -> Optional[str]:
        """Get the best available model for a specific task"""
        task_model_priority = {
            'fusion': ['fusion_model', 'fusion_distilled'],
            'ocr': ['trocr_finetuned', 'fusion_model'],
            'ner': ['ner_model', 'fusion_model'],
            'dss': ['dss_model', 'scheme_recommender', 'fusion_model'],
            'satellite': ['satellite_model', 'fusion_model'],
            'mobile': ['mobile_model', 'fusion_distilled', 'fusion_model']
        }
        
        candidates = task_model_priority.get(task_type, ['fusion_model'])
        
        for model_name in candidates:
            if self.is_model_available(model_name):
                logger.info(f"Selected {model_name} for task: {task_type}")
                return model_name
        
        logger.warning(f"No suitable model found for task: {task_type}")
        return None
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if model weights are available"""
        model_path = self.model_paths.get(model_name)
        return model_path and model_path.exists()
    
    def get_available_models(self) -> List[str]:
        """Get list of available trained models"""
        available = []
        for model_name, model_path in self.model_paths.items():
            if model_path.exists():
                available.append(model_name)
        return available
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        if model_name not in self.model_paths:
            return {'error': 'Model not registered'}
        
        model_path = self.model_paths[model_name]
        info = {
            'name': model_name,
            'path': str(model_path),
            'exists': model_path.exists(),
            'size_mb': self._get_model_size(model_path) if model_path.exists() else 0,
            'metadata': self.model_metadata.get(model_name, {})
        }
        
        if model_path.exists():
            info['last_modified'] = datetime.fromtimestamp(
                model_path.stat().st_mtime
            ).isoformat()
        
        return info
    
    def create_model_endpoint_config(self) -> Dict[str, Any]:
        """Create configuration for model serving endpoints"""
        available_models = self.get_available_models()
        
        endpoint_config = {
            'model_server_url': 'http://localhost:8000',
            'endpoints': {
                '/predict/fusion': {
                    'model': 'fusion_model',
                    'available': 'fusion_model' in available_models,
                    'fallback': 'fusion_distilled' if 'fusion_distilled' in available_models else None
                },
                '/predict/ocr': {
                    'model': 'trocr_finetuned',
                    'available': 'trocr_finetuned' in available_models,
                    'fallback': 'fusion_model' if 'fusion_model' in available_models else None
                },
                '/predict/ner': {
                    'model': 'ner_model', 
                    'available': 'ner_model' in available_models,
                    'fallback': 'fusion_model' if 'fusion_model' in available_models else None
                },
                '/predict/dss': {
                    'model': 'dss_model',
                    'available': 'dss_model' in available_models,
                    'fallback': 'scheme_recommender' if 'scheme_recommender' in available_models else 'fusion_model'
                },
                '/predict/satellite': {
                    'model': 'satellite_model',
                    'available': 'satellite_model' in available_models,
                    'fallback': 'fusion_model' if 'fusion_model' in available_models else None
                }
            },
            'health_check': '/health',
            'model_info': '/models/info',
            'available_models': available_models,
            'total_models': len(available_models)
        }
        
        return endpoint_config
    
    def _get_model_size(self, model_path: Path) -> float:
        """Get model file size in MB"""
        if not model_path.exists():
            return 0.0
        return model_path.stat().st_size / (1024 * 1024)  # Convert to MB
    
    def _save_registry(self):
        """Save model registry to disk"""
        try:
            registry_data = {
                'models': self.model_metadata,
                'last_updated': datetime.now().isoformat(),
                'total_registered': len(self.model_metadata)
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def _load_registry(self):
        """Load model registry from disk"""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    registry_data = json.load(f)
                    self.model_metadata = registry_data.get('models', {})
                    logger.info(f"Loaded {len(self.model_metadata)} models from registry")
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            self.model_metadata = {}

    def export_model_for_deployment(self, model_name: str, export_format: str = 'torchscript'):
        """Export trained model for production deployment"""
        if not self.is_model_available(model_name):
            logger.error(f"Model not available for export: {model_name}")
            return None
        
        export_dir = self.base_models_dir / "exports" / model_name
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if export_format == 'torchscript':
            export_path = export_dir / f"{model_name}_scripted.pt"
            # This will be implemented when models are actually loaded
            logger.info(f"Model export prepared: {export_path}")
            return str(export_path)
        elif export_format == 'onnx':
            export_path = export_dir / f"{model_name}.onnx"
            logger.info(f"ONNX export prepared: {export_path}")
            return str(export_path)
        
        return None

# Global model weights manager instance
weights_manager = ModelWeightsManager()

# Load existing registry on module import
weights_manager._load_registry()

if __name__ == "__main__":
    # Test the model weights manager
    print("Model Weights Manager Test")
    print("=" * 40)
    
    print(f"Available models: {weights_manager.get_available_models()}")
    print(f"Model directories created: {list(weights_manager.base_models_dir.iterdir())}")
    
    # Create endpoint configuration
    config = weights_manager.create_model_endpoint_config()
    print(f"\nEndpoint configuration:")
    print(json.dumps(config, indent=2))
    
    # Show expected model paths
    print(f"\nExpected model paths:")
    for name, path in weights_manager.model_paths.items():
        print(f"  {name}: {path}")