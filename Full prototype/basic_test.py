#!/usr/bin/env python3
"""
Simple Test Script for FRA AI System
Tests basic functionality without heavy dependencies
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config_loading():
    """Test configuration loading"""
    logger.info("Testing configuration loading...")
    
    try:
        config_path = project_root / "configs" / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check key sections
            required_sections = ['model', 'training', 'data', 'model_sources', 'llm_config']
            for section in required_sections:
                if section in config:
                    logger.info(f"✓ Found config section: {section}")
                else:
                    logger.warning(f"✗ Missing config section: {section}")
            
            # Check model sources
            model_sources = config.get('model_sources', {})
            logger.info(f"✓ Found {len(model_sources)} model sources")
            
            return True
        else:
            logger.error("Config file not found")
            return False
            
    except Exception as e:
        logger.error(f"Config loading failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure"""
    logger.info("Testing directory structure...")
    
    required_dirs = [
        "configs",
        "1_data_processing", 
        "2_model_fusion",
        "3_webgis_backend"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            logger.info(f"✓ Found directory: {dir_name}")
        else:
            logger.error(f"✗ Missing directory: {dir_name}")
            all_exist = False
    
    return all_exist

def test_python_imports():
    """Test critical Python imports"""
    logger.info("Testing Python imports...")
    
    critical_imports = [
        ('json', 'json'),
        ('pathlib', 'Path'),
        ('logging', 'logging')
    ]
    
    optional_imports = [
        ('torch', 'PyTorch'),
        ('transformers', 'Hugging Face Transformers'),
        ('fastapi', 'FastAPI'),
        ('PIL', 'Pillow')
    ]
    
    all_critical = True
    
    # Test critical imports
    for module, name in critical_imports:
        try:
            __import__(module)
            logger.info(f"✓ Critical import available: {name}")
        except ImportError:
            logger.error(f"✗ Critical import missing: {name}")
            all_critical = False
    
    # Test optional imports
    for module, name in optional_imports:
        try:
            __import__(module)
            logger.info(f"✓ Optional import available: {name}")
        except ImportError:
            logger.warning(f"⚠ Optional import missing: {name}")
    
    return all_critical

def create_directories():
    """Create necessary directories"""
    logger.info("Creating necessary directories...")
    
    dirs_to_create = [
        "data/raw",
        "data/processed",
        "data/sample_small",
        "models",
        "logs",
        "outputs",
        "2_model_fusion/checkpoints"
    ]
    
    for dir_path in dirs_to_create:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created/verified directory: {dir_path}")
    
    return True

def run_basic_test():
    """Run basic system test"""
    logger.info("="*60)
    logger.info("FRA AI SYSTEM - BASIC TEST")
    logger.info("="*60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Configuration Loading", test_config_loading), 
        ("Python Imports", test_python_imports),
        ("Directory Creation", create_directories)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("="*60)
    logger.info("BASIC TEST RESULTS")
    logger.info("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:25}: {status}")
        if result:
            passed += 1
    
    logger.info("="*60)
    logger.info(f"OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 Basic tests passed! System structure is ready.")
        return True
    else:
        logger.warning("⚠️  Some basic tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = run_basic_test()
    sys.exit(0 if success else 1)
