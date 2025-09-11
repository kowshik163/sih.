#!/usr/bin/env python3
"""
Environment and Secrets Management for FRA AI System
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EnvironmentManager:
    """Manages environment variables and secrets"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.env_file = self.project_root / ".env"
        self.env_example = self.project_root / ".env.example"
        self._load_environment()
    
    def _load_environment(self):
        """Load environment variables from .env file"""
        try:
            # Try to import python-dotenv
            from dotenv import load_dotenv
            
            if self.env_file.exists():
                load_dotenv(self.env_file)
                logger.info(f"Loaded environment from: {self.env_file}")
            elif self.env_example.exists():
                logger.warning(f"No .env file found. Copy {self.env_example} to .env and configure")
            else:
                logger.warning("No environment file found. Using system environment only")
                
        except ImportError:
            logger.warning("python-dotenv not installed. Using system environment only")
            logger.info("Install with: pip install python-dotenv")
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable"""
        return os.environ.get(key, default)
    
    def get_required(self, key: str) -> str:
        """Get required environment variable"""
        value = os.environ.get(key)
        if value is None:
            raise ValueError(f"Required environment variable not set: {key}")
        return value
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean environment variable"""
        value = self.get(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer environment variable"""
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer value for {key}: {value}, using default {default}")
            return default
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            'host': self.get('DB_HOST', 'localhost'),
            'port': self.get_int('DB_PORT', 5432),
            'name': self.get('DB_NAME', 'fra_gis'),
            'user': self.get('DB_USER', 'fra_user'),
            'password': self.get('DB_PASSWORD', 'fra_password'),
            'url': self.get('DATABASE_URL')
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            'secret_key': self.get_required('SECRET_KEY'),
            'jwt_secret_key': self.get_required('JWT_SECRET_KEY'),
            'jwt_expire_hours': self.get_int('JWT_EXPIRE_HOURS', 24)
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return {
            'host': self.get('API_HOST', '0.0.0.0'),
            'port': self.get_int('API_PORT', 8000),
            'debug': self.get_bool('DEBUG', False),
            'cors_origins': self.get('CORS_ORIGINS', '*').split(',')
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'cache_dir': self.get('MODEL_CACHE_DIR', './models'),
            'checkpoint_dir': self.get('CHECKPOINT_DIR', './checkpoints'),
            'hf_token': self.get('HF_TOKEN')
        }
    
    def validate_required_vars(self) -> Dict[str, str]:
        """Validate required environment variables"""
        required_vars = [
            'SECRET_KEY',
            'JWT_SECRET_KEY'
        ]
        
        missing_vars = {}
        for var in required_vars:
            if not self.get(var):
                missing_vars[var] = f"Required environment variable {var} not set"
        
        return missing_vars
    
    def setup_secrets(self, interactive: bool = True) -> bool:
        """Setup secrets interactively"""
        logger.info("Setting up secrets for FRA AI System...")
        
        # Check if .env exists
        if not self.env_file.exists() and self.env_example.exists():
            if interactive:
                response = input(f"Copy {self.env_example} to .env? (y/N): ")
                if response.lower() == 'y':
                    import shutil
                    shutil.copy(self.env_example, self.env_file)
                    logger.info(f"Copied {self.env_example} to {self.env_file}")
                    logger.info("Please edit .env file and configure your secrets")
                    return False
            else:
                logger.warning("No .env file found and not in interactive mode")
                return False
        
        # Validate current configuration
        missing_vars = self.validate_required_vars()
        
        if missing_vars:
            logger.error("Missing required environment variables:")
            for var, msg in missing_vars.items():
                logger.error(f"  - {msg}")
            
            if interactive:
                logger.info("Please update your .env file with the required variables")
            
            return False
        
        logger.info("✓ All required environment variables are set")
        return True

# Global environment manager instance
env = EnvironmentManager()

def setup_environment():
    """Setup environment for the application"""
    return env.setup_secrets()

def get_env_config() -> Dict[str, Any]:
    """Get complete environment configuration"""
    return {
        'database': env.get_database_config(),
        'security': env.get_security_config(),
        'api': env.get_api_config(),
        'model': env.get_model_config()
    }

if __name__ == "__main__":
    # Run environment setup
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        success = setup_environment()
        if success:
            print("✓ Environment setup completed successfully")
            sys.exit(0)
        else:
            print("✗ Environment setup failed")
            sys.exit(1)
    else:
        # Show current environment
        print("Current environment configuration:")
        config = get_env_config()
        
        for section, values in config.items():
            print(f"\n{section.upper()}:")
            for key, value in values.items():
                # Hide sensitive values
                if 'secret' in key.lower() or 'password' in key.lower() or 'token' in key.lower():
                    display_value = "*" * len(str(value)) if value else "NOT_SET"
                else:
                    display_value = value
                print(f"  {key}: {display_value}")
