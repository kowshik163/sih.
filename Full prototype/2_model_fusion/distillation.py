#!/usr/bin/env python3
"""
Knowledge Distillation for FRA AI System
Implements teacher-student distillation to create smaller, deployable models
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining KL divergence and supervised loss
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, targets=None):
        """
        Compute distillation loss
        
        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions  
            targets: Ground truth labels (optional)
        """
        # Soft target loss (KL divergence between teacher and student)
        soft_loss = self.kl_loss(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1)
        ) * (self.temperature ** 2)
        
        # Hard target loss (if targets provided)
        if targets is not None:
            hard_loss = self.ce_loss(student_logits, targets)
            total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        else:
            total_loss = soft_loss
        
        return total_loss

class FRADistillationTrainer:
    """
    Trainer for knowledge distillation in FRA system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.temperature = config.get('distillation', {}).get('temperature', 4.0)
        self.alpha = config.get('distillation', {}).get('alpha', 0.5)
        
        # Initialize loss function
        self.distillation_loss = DistillationLoss(
            temperature=self.temperature,
            alpha=self.alpha
        )
        
        # Models will be loaded later
        self.teacher_model = None
        self.student_model = None
    
    def load_teacher_model(self, teacher_path: str):
        """Load pre-trained teacher model"""
        logger.info(f"Loading teacher model from: {teacher_path}")
        
        try:
            # Import the main fusion model
            from main_fusion_model import EnhancedFRAUnifiedEncoder
            
            # Load teacher model
            self.teacher_model = EnhancedFRAUnifiedEncoder(self.config)
            checkpoint = torch.load(teacher_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.teacher_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.teacher_model.load_state_dict(checkpoint)
            
            self.teacher_model.to(self.device)
            self.teacher_model.eval()  # Set to evaluation mode
            
            # Freeze teacher parameters
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            logger.info("Teacher model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            return False
    
    def create_student_model(self, student_config: Dict[str, Any] = None):
        """Create smaller student model"""
        logger.info("Creating student model...")
        
        try:
            from main_fusion_model import EnhancedFRAUnifiedEncoder
            
            # Student configuration - smaller architecture
            if student_config is None:
                student_config = self.config.copy()
                # Reduce model size
                student_config['model']['hidden_size'] = self.config['model']['hidden_size'] // 2
                student_config['model']['graph_neural_network']['hidden_dim'] = \
                    self.config['model']['graph_neural_network']['hidden_dim'] // 2
                student_config['model']['memory_module']['capacity'] = \
                    self.config['model']['memory_module']['capacity'] // 2
                student_config['model']['temporal_modeling']['d_model'] = \
                    self.config['model']['temporal_modeling']['d_model'] // 2
            
            self.student_model = EnhancedFRAUnifiedEncoder(student_config)
            self.student_model.to(self.device)
            self.student_model.train()  # Set to training mode
            
            logger.info("Student model created successfully")
            logger.info(f"Student model parameters: {sum(p.numel() for p in self.student_model.parameters()):,}")
            logger.info(f"Teacher model parameters: {sum(p.numel() for p in self.teacher_model.parameters()):,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create student model: {e}")
            return False
    
    def distill_model(self, train_dataloader: DataLoader, 
                     val_dataloader: DataLoader = None,
                     num_epochs: int = 10,
                     learning_rate: float = 1e-4) -> Dict[str, Any]:
        """
        Perform knowledge distillation
        
        Args:
            train_dataloader: Training data
            val_dataloader: Validation data (optional)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for student model
        
        Returns:
            Training history and metrics
        """
        logger.info("Starting knowledge distillation...")
        
        if self.teacher_model is None or self.student_model is None:
            raise ValueError("Teacher and student models must be loaded/created first")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.get('training', {}).get('weight_decay', 0.01)
        )
        
        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            train_loss = self._distill_epoch(train_dataloader, optimizer, train=True)
            history['train_loss'].append(train_loss)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Validation phase
            if val_dataloader is not None:
                val_loss = self._distill_epoch(val_dataloader, optimizer, train=False)
                history['val_loss'].append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.student_model.state_dict().copy()
                    logger.info(f"New best validation loss: {val_loss:.6f}")
            
            # Update learning rate
            scheduler.step()
            
            logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {history['val_loss'][-1] if val_dataloader else 'N/A':.6f}")
        
        # Load best model if validation was used
        if best_model_state is not None:
            self.student_model.load_state_dict(best_model_state)
            logger.info("Loaded best model weights")
        
        logger.info("Knowledge distillation completed")
        
        return history
    
    def _distill_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                      train: bool = True) -> float:
        """Run single distillation epoch"""
        
        if train:
            self.student_model.train()
        else:
            self.student_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.set_grad_enabled(train):
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device (assuming batch is a dict with tensors)
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(batch)
                    teacher_logits = teacher_outputs.get('logits', teacher_outputs.get('predictions'))
                
                # Get student predictions
                student_outputs = self.student_model(batch)
                student_logits = student_outputs.get('logits', student_outputs.get('predictions'))
                
                # Compute distillation loss
                targets = batch.get('labels')  # May be None for unsupervised distillation
                loss = self.distillation_loss(student_logits, teacher_logits, targets)
                
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(), 
                        max_norm=1.0
                    )
                    
                    optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Progress logging
                if batch_idx % 100 == 0:
                    logger.info(f"Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_student_model(self, save_path: str, include_config: bool = True):
        """Save the distilled student model"""
        logger.info(f"Saving student model to: {save_path}")
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': self.student_model.state_dict(),
            'model_type': 'FRA_student_model',
            'distillation_config': {
                'temperature': self.temperature,
                'alpha': self.alpha
            }
        }
        
        if include_config:
            checkpoint['config'] = self.config
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
        logger.info("Student model saved successfully")
    
    def evaluate_compression_ratio(self) -> Dict[str, Any]:
        """Evaluate compression achieved through distillation"""
        if self.teacher_model is None or self.student_model is None:
            return {}
        
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        
        compression_ratio = teacher_params / student_params
        size_reduction = (1 - student_params / teacher_params) * 100
        
        metrics = {
            'teacher_parameters': teacher_params,
            'student_parameters': student_params,
            'compression_ratio': f"{compression_ratio:.2f}x",
            'size_reduction_percent': f"{size_reduction:.1f}%"
        }
        
        logger.info("Compression Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        return metrics

def main():
    """Example usage of distillation trainer"""
    # Example configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model': {
            'hidden_size': 1024,
            'graph_neural_network': {'hidden_dim': 512},
            'memory_module': {'capacity': 4096},
            'temporal_modeling': {'d_model': 256}
        },
        'distillation': {
            'temperature': 4.0,
            'alpha': 0.5
        },
        'training': {
            'weight_decay': 0.01
        }
    }
    
    # Initialize trainer
    trainer = FRADistillationTrainer(config)
    
    # Load teacher model (this would be your trained model)
    teacher_path = "path/to/teacher/model.pth"
    if os.path.exists(teacher_path):
        trainer.load_teacher_model(teacher_path)
        trainer.create_student_model()
        
        # Note: You would need to provide actual dataloaders here
        # history = trainer.distill_model(train_dataloader, val_dataloader)
        # trainer.save_student_model("path/to/student/model.pth")
        # metrics = trainer.evaluate_compression_ratio()
        
        logger.info("Distillation setup completed successfully")
    else:
        logger.error(f"Teacher model not found at: {teacher_path}")

if __name__ == "__main__":
    main()
