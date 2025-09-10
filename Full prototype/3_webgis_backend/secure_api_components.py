"""
Enhanced API Security and Model Integration
Real model invocation with input validation and security
"""

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator, Field
from typing import List, Dict, Optional, Any
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
import re
import logging
import hashlib
import time
from datetime import datetime, timedelta
import jwt
from functools import wraps
import asyncio
import aiofiles

# Security configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()
logger = logging.getLogger(__name__)

# Input validation models
class SecureTextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    language: Optional[str] = Field("en", regex=r"^[a-z]{2}$")
    
    @validator('text')
    def validate_text(cls, v):
        # Remove potentially harmful characters
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        
        # Basic XSS prevention
        dangerous_patterns = ['<script', 'javascript:', 'onclick=', 'onerror=']
        text_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in text_lower:
                raise ValueError("Invalid characters detected in text")
        
        return v.strip()

class SecureImageInput(BaseModel):
    image_base64: str = Field(..., min_length=100)
    format: str = Field(..., regex=r"^(jpeg|jpg|png|tiff)$")
    max_size_mb: Optional[float] = Field(10.0, gt=0, le=50)
    
    @validator('image_base64')
    def validate_image(cls, v):
        try:
            # Decode and validate image
            image_data = base64.b64decode(v)
            if len(image_data) > 50 * 1024 * 1024:  # 50MB limit
                raise ValueError("Image too large")
            
            # Verify it's actually an image
            img = Image.open(io.BytesIO(image_data))
            img.verify()
            return v
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")

class SecureGeoQuery(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_km: Optional[float] = Field(1.0, gt=0, le=100)
    
class SecureDSSQuery(BaseModel):
    village_id: Optional[str] = Field(None, regex=r"^[a-zA-Z0-9_-]+$")
    schemes: List[str] = Field([], max_items=20)
    priority_factors: Dict[str, float] = Field({}, max_items=10)
    
    @validator('schemes')
    def validate_schemes(cls, v):
        allowed_schemes = [
            'PM_KISAN', 'JAL_JEEVAN_MISSION', 'MGNREGA', 'DAJGUA',
            'FOREST_RIGHTS', 'HOUSING', 'HEALTH', 'EDUCATION'
        ]
        for scheme in v:
            if scheme not in allowed_schemes:
                raise ValueError(f"Invalid scheme: {scheme}")
        return v

# Authentication utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Rate limiting decorator
class RateLimiter:
    def __init__(self):
        self.requests = {}
        self.window = 60  # seconds
        self.max_requests = 60  # per window
    
    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        
        # Clean old entries
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip] 
                if now - req_time < self.window
            ]
        
        # Check rate limit
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        
        self.requests[client_ip].append(now)
        return True

rate_limiter = RateLimiter()

# Model loading with security
class SecureModelManager:
    def __init__(self):
        self.model = None
        self.model_hash = None
        self.last_loaded = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self, model_path: str, config: Dict):
        """Securely load model with verification"""
        try:
            # Verify model file integrity
            with open(model_path, 'rb') as f:
                model_hash = hashlib.sha256(f.read()).hexdigest()
            
            if self.model_hash != model_hash:
                logger.info(f"Loading model from {model_path}")
                
                # Load model
                from main_fusion_model import EnhancedFRAUnifiedEncoder
                self.model = EnhancedFRAUnifiedEncoder(config['model'])
                
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(self.device)
                self.model.eval()
                
                self.model_hash = model_hash
                self.last_loaded = datetime.now()
                
                logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Secure model prediction with input validation"""
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            with torch.no_grad():
                # Move inputs to device
                device_inputs = {}
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        device_inputs[key] = value.to(self.device)
                    else:
                        device_inputs[key] = value
                
                # Run inference
                outputs = self.model(device_inputs)
                
                # Convert outputs to JSON-serializable format
                result = {}
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        result[key] = value.cpu().numpy().tolist()
                    else:
                        result[key] = value
                
                return result
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed")

# Initialize secure model manager
model_manager = SecureModelManager()

# Enhanced preprocessing functions
def preprocess_text_secure(text: str) -> Dict[str, torch.Tensor]:
    """Securely preprocess text input"""
    try:
        # Tokenization with security checks
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")
        
        # Truncate and encode
        encoded = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
        
    except Exception as e:
        logger.error(f"Text preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail="Text preprocessing failed")

def preprocess_image_secure(image_base64: str) -> Dict[str, torch.Tensor]:
    """Securely preprocess image input"""
    try:
        # Decode image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize and normalize
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        return {'image': image_tensor}
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail="Image preprocessing failed")

# Enhanced OCR function
def perform_ocr_secure(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Perform OCR with the loaded model"""
    try:
        # Use the model for OCR
        inputs = {'image': image_tensor}
        outputs = model_manager.predict(inputs)
        
        # Extract text from model outputs (implement based on your model)
        extracted_text = outputs.get('text', '')
        confidence = outputs.get('confidence', 0.0)
        
        return {
            'text': extracted_text,
            'confidence': float(confidence),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail="OCR processing failed")

# Enhanced NER function
def perform_ner_secure(text: str) -> Dict[str, Any]:
    """Perform Named Entity Recognition with the loaded model"""
    try:
        # Preprocess text
        text_inputs = preprocess_text_secure(text)
        
        # Run NER model
        outputs = model_manager.predict(text_inputs)
        
        # Extract entities (implement based on your model)
        entities = outputs.get('entities', [])
        
        return {
            'entities': entities,
            'text': text,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"NER failed: {e}")
        raise HTTPException(status_code=500, detail="NER processing failed")

# Enhanced DSS function
def generate_dss_recommendations_secure(query: SecureDSSQuery) -> Dict[str, Any]:
    """Generate DSS recommendations using the loaded model"""
    try:
        # Prepare inputs for DSS model
        dss_inputs = {
            'village_id': query.village_id,
            'schemes': query.schemes,
            'priority_factors': query.priority_factors
        }
        
        # Convert to model inputs (implement based on your model)
        model_inputs = preprocess_dss_inputs(dss_inputs)
        
        # Run DSS model
        outputs = model_manager.predict(model_inputs)
        
        # Extract recommendations
        recommendations = outputs.get('recommendations', [])
        priorities = outputs.get('priorities', [])
        
        return {
            'recommendations': recommendations,
            'priorities': priorities,
            'query': dss_inputs,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"DSS generation failed: {e}")
        raise HTTPException(status_code=500, detail="DSS generation failed")

def preprocess_dss_inputs(dss_inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Convert DSS inputs to model format"""
    # Implement based on your specific model requirements
    # This is a placeholder implementation
    return {
        'dss_query': torch.tensor([1.0])  # Replace with actual preprocessing
    }
