#!/usr/bin/env python3
"""
Production Readiness Checker for FRA AI System
Checks system status and readiness for deployment
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionChecker:
    """Check system readiness for production"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues = []
        self.warnings = []
        self.passed = []
    
    def check_environment(self) -> bool:
        """Check environment configuration"""
        logger.info("Checking environment configuration...")
        
        env_file = self.project_root / ".env"
        if not env_file.exists():
            self.issues.append("No .env file found - secrets not configured")
            return False
        
        # Check required environment variables
        required_vars = ['SECRET_KEY', 'JWT_SECRET_KEY']
        missing = []
        
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            
            for var in required_vars:
                if not os.environ.get(var):
                    missing.append(var)
            
            if missing:
                self.issues.append(f"Missing required environment variables: {missing}")
                return False
            
            # Check for default values
            if os.environ.get('SECRET_KEY') == 'your-super-secret-key-change-this-in-production':
                self.warnings.append("SECRET_KEY is using default value - change for production")
            
            self.passed.append("Environment configuration")
            return True
            
        except ImportError:
            self.warnings.append("python-dotenv not installed - using system environment")
            return True
    
    def check_models(self) -> bool:
        """Check model availability"""
        logger.info("Checking model availability...")
        
        checkpoint_dir = self.project_root / "2_model_fusion" / "checkpoints"
        model_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.bin"))
        
        if not model_files:
            self.warnings.append("No model checkpoints found - API will run in mock mode")
            return False
        
        # Check if final model exists
        final_model = checkpoint_dir / "final_model.pth"
        if final_model.exists():
            self.passed.append("Primary model checkpoint found")
        else:
            self.warnings.append("final_model.pth not found - using alternative checkpoint")
        
        # Check model directory
        models_dir = self.project_root.parent / "models"
        if models_dir.exists() and list(models_dir.iterdir()):
            self.passed.append("Downloaded models directory")
        else:
            self.warnings.append("No downloaded models found - run download_models.py")
        
        return True
    
    def check_data(self) -> bool:
        """Check data availability"""
        logger.info("Checking data availability...")
        
        data_dir = self.project_root / "data"
        if not data_dir.exists():
            self.issues.append("Data directory not found")
            return False
        
        # Check for training data
        training_data = data_dir / "training_data.json"
        if training_data.exists():
            self.passed.append("Training data found")
        else:
            self.warnings.append("No training data found - run data pipeline")
        
        # Check for sample data
        sample_dir = data_dir / "sample_small"
        if sample_dir.exists():
            self.passed.append("Sample data available for testing")
        else:
            self.warnings.append("No sample data - smoke test will create it")
        
        return True
    
    def check_dependencies(self) -> bool:
        """Check Python dependencies"""
        logger.info("Checking Python dependencies...")
        
        critical_deps = [
            'torch', 'transformers', 'fastapi', 
            'uvicorn', 'PIL', 'numpy', 'pandas'
        ]
        
        missing_deps = []
        for dep in critical_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            self.issues.append(f"Missing critical dependencies: {missing_deps}")
            return False
        
        self.passed.append("Core dependencies")
        
        # Check optional dependencies
        optional_deps = ['accelerate', 'bitsandbytes', 'psycopg2']
        missing_optional = []
        
        for dep in optional_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_optional.append(dep)
        
        if missing_optional:
            self.warnings.append(f"Missing optional dependencies: {missing_optional}")
        
        return True
    
    def check_ports(self) -> bool:
        """Check if required ports are available"""
        logger.info("Checking port availability...")
        
        import socket
        
        ports_to_check = [8000, 5432]  # API and PostgreSQL
        busy_ports = []
        
        for port in ports_to_check:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                busy_ports.append(port)
        
        if busy_ports:
            self.warnings.append(f"Ports already in use: {busy_ports}")
        else:
            self.passed.append("Required ports available")
        
        return True
    
    def check_gpu(self) -> bool:
        """Check GPU availability"""
        logger.info("Checking GPU availability...")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                self.passed.append(f"GPU available: {gpu_name} ({gpu_count} devices)")
                return True
            else:
                self.warnings.append("No GPU available - using CPU only")
                return False
        except ImportError:
            self.warnings.append("PyTorch not available for GPU check")
            return False
    
    def check_disk_space(self) -> bool:
        """Check available disk space"""
        logger.info("Checking disk space...")
        
        try:
            import shutil
            free_space_gb = shutil.disk_usage('.').free / (1024**3)
            
            if free_space_gb < 10:
                self.issues.append(f"Low disk space: {free_space_gb:.1f}GB available")
                return False
            elif free_space_gb < 50:
                self.warnings.append(f"Limited disk space: {free_space_gb:.1f}GB available")
            else:
                self.passed.append(f"Adequate disk space: {free_space_gb:.1f}GB available")
            
            return True
        except Exception:
            self.warnings.append("Could not check disk space")
            return True
    
    def run_comprehensive_check(self) -> Dict:
        """Run all production readiness checks"""
        logger.info("="*60)
        logger.info("FRA AI SYSTEM - PRODUCTION READINESS CHECK")
        logger.info("="*60)
        
        checks = [
            ("Environment", self.check_environment),
            ("Dependencies", self.check_dependencies),
            ("Models", self.check_models),
            ("Data", self.check_data),
            ("Ports", self.check_ports),
            ("GPU", self.check_gpu),
            ("Disk Space", self.check_disk_space)
        ]
        
        results = {}
        
        for check_name, check_func in checks:
            logger.info(f"\nRunning: {check_name}")
            try:
                result = check_func()
                results[check_name] = result
                status = "✓ PASS" if result else "⚠ WARN"
                logger.info(f"{check_name}: {status}")
            except Exception as e:
                logger.error(f"{check_name}: ✗ ERROR - {e}")
                results[check_name] = False
                self.issues.append(f"{check_name} check failed: {e}")
        
        return self.generate_report(results)
    
    def generate_report(self, results: Dict) -> Dict:
        """Generate comprehensive report"""
        logger.info("="*60)
        logger.info("PRODUCTION READINESS REPORT")
        logger.info("="*60)
        
        # Count results
        passed_count = len(self.passed)
        warnings_count = len(self.warnings)
        issues_count = len(self.issues)
        
        # Determine overall status
        if issues_count == 0 and warnings_count == 0:
            overall_status = "READY"
            status_emoji = "🚀"
        elif issues_count == 0:
            overall_status = "READY_WITH_WARNINGS"
            status_emoji = "⚠️"
        else:
            overall_status = "NOT_READY"
            status_emoji = "❌"
        
        logger.info(f"\nOverall Status: {status_emoji} {overall_status}")
        logger.info(f"Passed: {passed_count} | Warnings: {warnings_count} | Issues: {issues_count}")
        
        # Show details
        if self.passed:
            logger.info(f"\n✓ PASSED ({len(self.passed)}):")
            for item in self.passed:
                logger.info(f"  ✓ {item}")
        
        if self.warnings:
            logger.info(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.info(f"  ⚠ {warning}")
        
        if self.issues:
            logger.info(f"\n❌ ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                logger.info(f"  ❌ {issue}")
        
        # Recommendations
        logger.info("\n📋 RECOMMENDATIONS:")
        
        if issues_count > 0:
            logger.info("  1. Fix critical issues listed above before deployment")
        
        if "No model checkpoints found" in str(self.warnings):
            logger.info("  2. Run 'python smoke_test.py' to create dummy model for testing")
            logger.info("  3. Run 'python ../scripts/download_models.py' for real models")
        
        if "No training data found" in str(self.warnings):
            logger.info("  4. Run 'python run.py --data-pipeline' to process data")
        
        if "Missing optional dependencies" in str(self.warnings):
            logger.info("  5. Install optional dependencies: pip install accelerate bitsandbytes")
        
        if "No GPU available" in str(self.warnings):
            logger.info("  6. For better performance, consider GPU-enabled deployment")
        
        logger.info("="*60)
        
        return {
            'overall_status': overall_status,
            'passed': self.passed,
            'warnings': self.warnings,
            'issues': self.issues,
            'results': results,
            'ready_for_production': issues_count == 0
        }

def main():
    checker = ProductionChecker()
    report = checker.run_comprehensive_check()
    
    # Return appropriate exit code
    return 0 if report['ready_for_production'] else 1

if __name__ == "__main__":
    sys.exit(main())
