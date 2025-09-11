#!/usr/bin/env python3
"""
Model Download Script for FRA AI System
Automatically downloads required models from Hugging Face Hub
Supports all models specified in readme and project plan
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

# Model priority groups for selective downloading
MODEL_PRIORITY_GROUPS = {
    "essential": [
        "primary_llm", "layoutlm", "trocr", "deeplabv3", "detr", 
        "clip", "indic_bert", "legal_ner"
    ],
    "standard": [
        "secondary_llm", "layoutlm_large", "trocr_large", "deeplabv3_satellite",
        "segment_anything", "legal_translation", "bert_multilingual"
    ],
    "advanced": [
        "backup_llm", "sam_model", "geospatial_fm", "clip_large", 
        "swin_large", "vit_large"
    ]
}

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

def get_model_size(model_id: str) -> str:
    """Estimate model size for download planning"""
    size_estimates = {
        "7B": "~14GB", "3B": "~6GB", "1B": "~2GB", 
        "base": "~500MB", "large": "~1GB", "huge": "~2.5GB"
    }
    
    for size_key, estimate in size_estimates.items():
        if size_key.lower() in model_id.lower():
            return estimate
    return "~1GB"

def check_disk_space(required_gb: float = 50.0) -> bool:
    """Check available disk space"""
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)  # GB
        if free_space < required_gb:
            logger.warning(f"Low disk space: {free_space:.1f}GB available, {required_gb}GB recommended")
            return False
        return True
    except Exception:
        logger.warning("Could not check disk space")
        return True

def download_model(model_name: str, model_id: str, cache_dir: str = None) -> str:
    """
    Download a model from Hugging Face Hub with error handling and progress
    
    Args:
        model_name: Local name for the model
        model_id: Hugging Face model identifier
        cache_dir: Optional cache directory
    
    Returns:
        Path to downloaded model
    """
    logger.info(f"Downloading {model_name}: {model_id}")
    logger.info(f"Estimated size: {get_model_size(model_id)}")
    
    try:
        # Use snapshot_download for complete model repos
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False,
            allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt", "*.py", "README.md"],
            ignore_patterns=["*.git*", "*.DS_Store", "__pycache__/*"]
        )
        logger.info(f"✓ Successfully downloaded {model_name} to: {local_dir}")
        return local_dir
        
    except Exception as e:
        logger.error(f"✗ Failed to download {model_name} ({model_id}): {str(e)}")
        
        # Try downloading just config and tokenizer for partial functionality
        try:
            logger.info(f"Attempting to download config files only for {model_name}")
            config_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
            partial_dir = os.path.join(cache_dir or "models", model_name + "_config_only")
            os.makedirs(partial_dir, exist_ok=True)
            
            for file in config_files:
                try:
                    hf_hub_download(
                        repo_id=model_id,
                        filename=file,
                        local_dir=partial_dir,
                        resume_download=True
                    )
                except Exception:
                    continue  # File might not exist
            
            logger.info(f"Downloaded config files for {model_name} to: {partial_dir}")
            return partial_dir
            
        except Exception:
            logger.error(f"Complete failure downloading {model_name}")
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
    parser.add_argument('--cache-dir', default='models',
                       help='Custom cache directory for models')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to download (default: essential)')
    parser.add_argument('--priority', choices=['essential', 'standard', 'advanced', 'all'], 
                       default='essential', help='Model priority group to download')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if cached')
    parser.add_argument('--dry-run', action='store_true',
3                       help='Show what would be downloaded without downloading')
    parser.add_argument('--check-space', action='store_true',
                       help='Check available disk space before downloading')
    
    args = parser.parse_args()
    
    # Check disk space if requested
    if args.check_space:
        if not check_disk_space():
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                logger.info("Download cancelled by user")
                return 1
    
    # Setup authentication
    setup_hf_authentication()
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    config = load_model_config(args.config)
    model_sources = config.get('model_sources', {})
    
    if not model_sources:
        logger.error("No model_sources found in config")
        return 1
    
    # Determine which models to download
    if args.models:
        # Specific models requested
        models_to_download = {k: v for k, v in model_sources.items() if k in args.models}
        missing_models = set(args.models) - set(models_to_download.keys())
        if missing_models:
            logger.warning(f"Models not found in config: {missing_models}")
    elif args.priority == 'all':
        models_to_download = model_sources
    else:
        # Use priority group
        priority_models = MODEL_PRIORITY_GROUPS.get(args.priority, [])
        models_to_download = {k: v for k, v in model_sources.items() if k in priority_models}
        
        # Add higher priority models if they exist
        if args.priority == 'standard':
            essential_models = {k: v for k, v in model_sources.items() 
                              if k in MODEL_PRIORITY_GROUPS['essential']}
            models_to_download.update(essential_models)
        elif args.priority == 'advanced':
            for priority in ['essential', 'standard']:
                priority_models = {k: v for k, v in model_sources.items() 
                                 if k in MODEL_PRIORITY_GROUPS[priority]}
                models_to_download.update(priority_models)
    
    if not models_to_download:
        logger.error("No models to download based on selection criteria")
        return 1
    
    logger.info(f"Will download {len(models_to_download)} models:")
    total_estimated_size = 0
    for name, model_id in models_to_download.items():
        size_est = get_model_size(model_id)
        logger.info(f"  {name}: {model_id} ({size_est})")
        # Rough size calculation for planning
        if "14GB" in size_est:
            total_estimated_size += 14
        elif "6GB" in size_est:
            total_estimated_size += 6
        elif "2GB" in size_est:
            total_estimated_size += 2
        else:
            total_estimated_size += 1
    
    logger.info(f"Total estimated download size: ~{total_estimated_size}GB")
    
    if args.dry_run:
        logger.info("Dry run complete - no models downloaded")
        return 0
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Download models
    successful_downloads = {}
    failed_downloads = []
    
    for model_name, model_id in models_to_download.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Downloading {model_name} ({len(successful_downloads)+1}/{len(models_to_download)})")
        logger.info(f"{'='*50}")
        
        try:
            local_path = download_model(model_name, model_id, args.cache_dir)
            successful_downloads[model_name] = local_path
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")
            failed_downloads.append(model_name)
            continue
    
    # Update config with model paths
    if successful_downloads:
        try:
            update_config_with_model_paths(args.config, successful_downloads)
        except Exception as e:
            logger.warning(f"Could not update config file: {str(e)}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✓ Successfully downloaded: {len(successful_downloads)} models")
    
    if successful_downloads:
        logger.info("\nSuccessful downloads:")
        for name, path in successful_downloads.items():
            logger.info(f"  ✓ {name}: {path}")
    
    if failed_downloads:
        logger.error(f"\n✗ Failed downloads: {len(failed_downloads)} models")
        for name in failed_downloads:
            logger.error(f"  ✗ {name}")
    
    logger.info(f"\nModels cached in: {args.cache_dir}")
    logger.info("You can now run the FRA AI system with downloaded models!")
    
    return 0 if not failed_downloads else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
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
