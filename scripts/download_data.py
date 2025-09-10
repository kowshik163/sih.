#!/usr/bin/env python3
"""
Data Download Script for FRA AI System
Handles dataset ingestion from various sources (HuggingFace, HTTP, S3, Google Drive)
"""

import os
import json
import argparse
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_http_file(url: str, output_path: str, chunk_size: int = 8192):
    """Download file from HTTP/HTTPS URL"""
    logger.info(f"Downloading from HTTP: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Progress: {progress:.1f}%")
        
        logger.info(f"Downloaded: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"HTTP download failed: {str(e)}")
        raise

def download_google_drive_file(file_id: str, output_path: str):
    """Download file from Google Drive using gdown"""
    try:
        import gdown
    except ImportError:
        logger.error("gdown not installed. Install with: pip install gdown")
        raise
    
    logger.info(f"Downloading from Google Drive: {file_id}")
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        logger.info(f"Downloaded from Google Drive: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Google Drive download failed: {str(e)}")
        raise

def download_git_repository(url: str, output_dir: str):
    """Download git repository"""
    try:
        import git
    except ImportError:
        logger.error("GitPython not installed. Install with: pip install gitpython")
        # Fallback to git command
        import subprocess
        try:
            subprocess.run(['git', 'clone', url, output_dir], check=True)
            logger.info(f"Downloaded git repo using command: {output_dir}")
            return output_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"Git clone failed: {str(e)}")
            raise
    
    logger.info(f"Cloning git repository: {url}")
    
    try:
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        git.Repo.clone_from(url, output_dir)
        logger.info(f"Cloned git repository to: {output_dir}")
        return output_dir
        
    except Exception as e:
        logger.error(f"Git repository download failed: {str(e)}")
        raise

def download_huggingface_dataset(dataset_id: str, output_dir: str, split: str = None):
    """Download dataset from Hugging Face Hub"""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Install with: pip install datasets")
        raise
    
    logger.info(f"Loading Hugging Face dataset: {dataset_id}")
    
    try:
        # Load dataset
        if split:
            dataset = load_dataset(dataset_id, split=split)
        else:
            dataset = load_dataset(dataset_id)
        
        # Save to disk
        os.makedirs(output_dir, exist_ok=True)
        dataset.save_to_disk(output_dir)
        logger.info(f"Saved Hugging Face dataset to: {output_dir}")
        return output_dir
        
    except Exception as e:
        logger.error(f"Hugging Face dataset download failed: {str(e)}")
        raise

def extract_archive(archive_path: str, extract_to: str):
    """Extract zip or tar archive"""
    logger.info(f"Extracting archive: {archive_path}")
    
    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            logger.warning(f"Unknown archive format: {archive_path}")
            return archive_path
        
        logger.info(f"Extracted to: {extract_to}")
        return extract_to
        
    except Exception as e:
        logger.error(f"Archive extraction failed: {str(e)}")
        raise

def download_dataset(source_info: Dict, output_dir: str) -> str:
    """
    Download dataset from various sources
    
    Args:
        source_info: Dictionary with source information
        output_dir: Output directory for downloaded data
    
    Returns:
        Path to downloaded/extracted data
    """
    source_type = source_info.get('type', 'http')
    url = source_info.get('url')
    
    if not url:
        raise ValueError("No URL provided in source info")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if source_type == 'git':
        # Git repository
        return download_git_repository(url, output_dir)
    
    elif source_type == 'huggingface':
        # Hugging Face dataset
        return download_huggingface_dataset(url, output_dir, source_info.get('split'))
    
    elif source_type == 'google_drive':
        # Extract file ID from Google Drive URL
        if 'drive.google.com' in url:
            file_id = url.split('/d/')[1].split('/')[0] if '/d/' in url else url.split('id=')[1].split('&')[0]
        else:
            file_id = url  # Assume it's just the file ID
        
        filename = source_info.get('filename', f'gdrive_download_{file_id}')
        output_path = os.path.join(output_dir, filename)
        download_google_drive_file(file_id, output_path)
        
        # Extract if it's an archive
        if filename.endswith(('.zip', '.tar', '.tar.gz', '.tgz')):
            return extract_archive(output_path, output_dir)
        return output_path
    
    else:  # HTTP/HTTPS
        parsed_url = urlparse(url)
        filename = source_info.get('filename') or os.path.basename(parsed_url.path) or 'download'
        output_path = os.path.join(output_dir, filename)
        
        download_http_file(url, output_path)
        
        # Extract if it's an archive
        if filename.endswith(('.zip', '.tar', '.tar.gz', '.tgz')):
            return extract_archive(output_path, output_dir)
        return output_path

def download_dataset_collection(collection_info: Dict, output_dir: str) -> List[str]:
    """
    Download a collection of datasets
    
    Args:
        collection_info: Dictionary with collection information
        output_dir: Output directory for downloaded data
    
    Returns:
        List of paths to downloaded data
    """
    sources = collection_info.get('sources', [])
    downloaded_paths = []
    
    logger.info(f"Downloading collection with {len(sources)} sources")
    
    for idx, source in enumerate(sources):
        source_name = source.get('name', f'source_{idx}')
        source_output_dir = os.path.join(output_dir, source_name)
        
        logger.info(f"Downloading source: {source_name}")
        try:
            result_path = download_dataset(source, source_output_dir)
            downloaded_paths.append(result_path)
            logger.info(f"✓ Downloaded {source_name} to {result_path}")
        except Exception as e:
            logger.error(f"✗ Failed to download {source_name}: {str(e)}")
            continue
    
    return downloaded_paths

def load_data_config(config_path: str) -> Dict:
    """Load data configuration"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download datasets for FRA AI System")
    parser.add_argument('--config', default='Full prototype/configs/config.json',
                       help='Path to config file')
    parser.add_argument('--data-url', type=str,
                       help='Single dataset URL to download')
    parser.add_argument('--data-type', choices=['http', 'huggingface', 'google_drive'], 
                       default='http', help='Type of data source')
    parser.add_argument('--out-dir', default='data/raw',
                       help='Output directory for downloaded data')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to download from config')
    
    args = parser.parse_args()
    
    # Single URL download mode
    if args.data_url:
        source_info = {
            'type': args.data_type,
            'url': args.data_url
        }
        
        try:
            result_path = download_dataset(source_info, args.out_dir)
            logger.info(f"Successfully downloaded to: {result_path}")
            return 0
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            return 1
    
    # Config-based download mode
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        
        # Create a sample config
        sample_config = {
            "data_sources": {
                "sample_fra_documents": {
                    "type": "http",
                    "url": "https://example.com/fra_documents.zip",
                    "filename": "fra_documents.zip",
                    "description": "Sample FRA documents for testing"
                },
                "village_boundaries": {
                    "type": "huggingface",
                    "url": "example/village-boundaries-india",
                    "description": "Village boundary shapefiles"
                },
                "satellite_imagery": {
                    "type": "google_drive",
                    "url": "1ABC123DEF456GHI789JKL",
                    "filename": "satellite_data.tar.gz",
                    "description": "Satellite imagery samples"
                }
            }
        }
        
        logger.info(f"Creating sample config at: {config_path}")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        logger.info("Please update the config file with actual dataset URLs and re-run")
        return 1
    
    # Load config
    config = load_data_config(config_path)
    data_sources = config.get('data_sources', {})
    
    if not data_sources:
        logger.error("No data_sources found in config")
        return 1
    
    # Filter datasets if specific ones requested
    if args.datasets:
        data_sources = {k: v for k, v in data_sources.items() if k in args.datasets}
    
    logger.info(f"Will download {len(data_sources)} datasets:")
    for name, info in data_sources.items():
        logger.info(f"  {name}: {info.get('description', 'No description')}")
    
    # Download datasets
    successful_downloads = []
    failed_downloads = []
    
    for dataset_name, source_info in data_sources.items():
        logger.info(f"\nDownloading dataset: {dataset_name}")
        dataset_output_dir = os.path.join(args.out_dir, dataset_name)
        
        try:
            if source_info.get('type') == 'collection':
                # Handle collection of datasets
                result_paths = download_dataset_collection(source_info, dataset_output_dir)
                if result_paths:
                    successful_downloads.append((dataset_name, result_paths))
                    logger.info(f"✓ Successfully downloaded collection {dataset_name}")
                else:
                    failed_downloads.append(dataset_name)
            else:
                # Handle single dataset
                result_path = download_dataset(source_info, dataset_output_dir)
                successful_downloads.append((dataset_name, result_path))
                logger.info(f"✓ Successfully downloaded {dataset_name}")
        except Exception as e:
            logger.error(f"✗ Failed to download {dataset_name}: {str(e)}")
            failed_downloads.append(dataset_name)
            continue
    
    # Summary
    logger.info(f"\nDownload Summary:")
    logger.info(f"✓ Successfully downloaded: {len(successful_downloads)} datasets")
    if failed_downloads:
        logger.error(f"✗ Failed downloads: {len(failed_downloads)} datasets")
        logger.error(f"  Failed: {', '.join(failed_downloads)}")
    
    # Print paths
    if successful_downloads:
        logger.info("\nDownload Paths:")
        for name, path in successful_downloads:
            logger.info(f"  {name}: {path}")
    
    return 0 if not failed_downloads else 1

if __name__ == "__main__":
    sys.exit(main())
