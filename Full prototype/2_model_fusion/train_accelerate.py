#!/usr/bin/env python3
"""
Accelerate Training Script for FRA AI Fusion Model
Supports distributed training, mixed precision, and DeepSpeed
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))
sys.path.append(str(current_dir))

def main():
    parser = argparse.ArgumentParser(description="Accelerate training for FRA AI Fusion")
    parser.add_argument("--config", type=str, default="../configs/config.json", 
                       help="Path to training configuration")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--accelerate-config", type=str, 
                       choices=["single_gpu", "multi_gpu", "deepspeed"],
                       default="single_gpu",
                       help="Accelerate configuration to use")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--use-8bit", action="store_true",
                       help="Use 8-bit optimization")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Update config with command line arguments
    if args.use_8bit:
        config.setdefault('training', {})['use_8bit'] = True
    
    # Set accelerate config path
    accelerate_config_path = current_dir / ".." / "configs" / "accelerate" / f"{args.accelerate_config}.yaml"
    os.environ['ACCELERATE_CONFIG_FILE'] = str(accelerate_config_path)
    
    print(f"Using accelerate config: {accelerate_config_path}")
    print(f"Training configuration: {args.config}")
    print(f"Training data: {args.data}")
    print(f"8-bit optimization: {args.use_8bit}")
    
    # Import and run training
    from train_fusion import EnhancedFRATrainingPipeline
    
    # Initialize trainer
    trainer = EnhancedFRATrainingPipeline(config)
    
    # Resume from checkpoint if specified
    if args.resume and Path(args.resume).exists():
        print(f"Resuming training from: {args.resume}")
        # Load checkpoint logic would go here
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train_enhanced_pipeline(args.data)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
