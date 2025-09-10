#!/usr/bin/env python3
"""
Model Download Script for FRA AI System
Automatically downloads required models from Hugging Face Hub
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from huggingface_hub import snapshot_download, hf_hub_download, login
    from huggingface_hub.utils import HfFolder
except ImportError:
    logger.error("huggingface_hub not found. Please install: pip install huggingface_hub")
    sys.exit(1)

def setup_hf_authentication():
    """Setup Hugging Face authentication"""
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        logger.info("Found HF_TOKEN in environment, logging in...")
        login(token=hf_token)
        return True
    else:
        logger.warning("No HF_TOKEN found. Some private models may not be accessible.")
        return False

def download_model(model_name: str, model_id: str, cache_dir: str = None) -> str:
    """
    Download a model from Hugging Face Hub
    
    Args:
        model_name: Local name for the model
        model_id: Hugging Face model identifier
        cache_dir: Optional cache directory
    
    Returns:
        Path to downloaded model
    """
    logger.info(f"Downloading {model_name}: {model_id}")
    
    try:
        # Use snapshot_download for complete model repos
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False
        )
        logger.info(f"Successfully downloaded {model_name} to: {local_dir}")
        return local_dir
        
    except Exception as e:
        logger.error(f"Failed to download {model_name} ({model_id}): {str(e)}")
        raise

def load_model_config(config_path: str) -> Dict:
    """Load model configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        raise

def update_config_with_model_paths(config_path: str, model_paths: Dict[str, str]):
    """Update config file with downloaded model paths"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Add model_paths section to config
        if 'model_paths' not in config:
            config['model_paths'] = {}
        
        config['model_paths'].update(model_paths)
        
        # Write back to config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Updated config file with model paths: {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to update config: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download models for FRA AI System")
    parser.add_argument('--config', default='Full prototype/configs/config.json', 
                       help='Path to config file')
    parser.add_argument('--cache-dir', default=None,
                       help='Custom cache directory for models')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to download (default: all)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if cached')
    
    args = parser.parse_args()
    
    # Setup authentication
    setup_hf_authentication()
    
    # Default model sources - these are the core models needed for FRA system
    default_models = {
        'layoutlm': 'microsoft/layoutlmv3-base',
        'trocr': 'microsoft/trocr-base-stage1',
        'distilgpt2': 'distilgpt2',
        'bert_base': 'bert-base-uncased',
        'roberta_base': 'roberta-base',
        'detr': 'facebook/detr-resnet-50',
        'clip': 'openai/clip-vit-base-patch32'
    }
    
    # Load config if exists
    config_path = args.config
    if os.path.exists(config_path):
        config = load_model_config(config_path)
        model_sources = config.get('model_sources', default_models)
    else:
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Using default model sources")
        model_sources = default_models
    
    # Filter models if specific ones requested
    if args.models:
        model_sources = {k: v for k, v in model_sources.items() if k in args.models}
    
    logger.info(f"Will download {len(model_sources)} models:")
    for name, repo in model_sources.items():
        logger.info(f"  {name}: {repo}")
    
    # Download models
    model_paths = {}
    failed_downloads = []
    
    for model_name, model_id in model_sources.items():
        try:
            local_path = download_model(model_name, model_id, args.cache_dir)
            model_paths[model_name] = local_path
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")
            failed_downloads.append(model_name)
            continue
    
    # Update config with model paths if config exists
    if os.path.exists(config_path) and model_paths:
        try:
            update_config_with_model_paths(config_path, model_paths)
        except Exception as e:
            logger.warning(f"Failed to update config: {str(e)}")
    
    # Summary
    logger.info(f"\nDownload Summary:")
    logger.info(f"✓ Successfully downloaded: {len(model_paths)} models")
    if failed_downloads:
        logger.error(f"✗ Failed downloads: {len(failed_downloads)} models")
        logger.error(f"  Failed: {', '.join(failed_downloads)}")
    
    # Print model paths for reference
    if model_paths:
        logger.info("\nModel Paths:")
        for name, path in model_paths.items():
            logger.info(f"  {name}: {path}")
    
    return 0 if not failed_downloads else 1

if __name__ == "__main__":
    sys.exit(main())
