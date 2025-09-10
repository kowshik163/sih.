#!/bin/bash
"""
Advanced Training Launch Scripts for FRA AI Fusion System
Supports multiple training configurations with accelerate, deepspeed, and 8-bit
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Check if we're in the correct directory
if [ ! -f "run.py" ]; then
    print_error "Please run this script from the 'Full prototype' directory"
    exit 1
fi

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi -L | wc -l)
        print_status "Found $gpu_count GPU(s)"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        return $gpu_count
    else
        print_warning "No NVIDIA GPU detected. Using CPU training."
        return 0
    fi
}

# Function to check and install requirements
check_requirements() {
    print_status "Checking Python requirements..."
    
    # Check for essential packages
    python -c "import torch, transformers, accelerate, datasets" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "Installing missing requirements..."
        pip install -r requirements.txt
    fi
    
    # Check for optional packages
    python -c "import bitsandbytes" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "bitsandbytes not found. Install for 8-bit training: pip install bitsandbytes"
    fi
    
    python -c "import deepspeed" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "deepspeed not found. Install for ZeRO optimization: pip install deepspeed"
    fi
}

# Function to setup HuggingFace token
setup_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        print_warning "HF_TOKEN not set. Some models may not be accessible."
        echo -n "Enter your HuggingFace token (or press Enter to skip): "
        read -r hf_token
        if [ ! -z "$hf_token" ]; then
            export HF_TOKEN="$hf_token"
            print_status "HF_TOKEN set for this session"
        fi
    else
        print_status "HF_TOKEN detected"
    fi
}

# Function to run single GPU training
single_gpu_training() {
    print_header "Single GPU Training"
    
    local data_path=${1:-"../data/processed/training_data.json"}
    local use_8bit=${2:-false}
    
    local cmd="python 2_model_fusion/train_accelerate.py"
    cmd="$cmd --config configs/config.json"
    cmd="$cmd --data $data_path"
    cmd="$cmd --accelerate-config single_gpu"
    
    if [ "$use_8bit" = true ]; then
        cmd="$cmd --use-8bit"
        print_status "Enabled 8-bit optimization"
    fi
    
    print_status "Command: $cmd"
    eval $cmd
}

# Function to run multi-GPU training
multi_gpu_training() {
    print_header "Multi-GPU Distributed Training"
    
    local data_path=${1:-"../data/processed/training_data.json"}
    local num_gpus=${2:-2}
    local use_8bit=${3:-false}
    
    print_status "Training on $num_gpus GPUs"
    
    local cmd="accelerate launch"
    cmd="$cmd --config_file configs/accelerate/multi_gpu.yaml"
    cmd="$cmd --num_processes $num_gpus"
    cmd="$cmd 2_model_fusion/train_accelerate.py"
    cmd="$cmd --config configs/config.json"
    cmd="$cmd --data $data_path"
    cmd="$cmd --accelerate-config multi_gpu"
    
    if [ "$use_8bit" = true ]; then
        cmd="$cmd --use-8bit"
        print_status "Enabled 8-bit optimization"
    fi
    
    print_status "Command: $cmd"
    eval $cmd
}

# Function to run DeepSpeed training
deepspeed_training() {
    print_header "DeepSpeed ZeRO Training"
    
    local data_path=${1:-"../data/processed/training_data.json"}
    local zero_stage=${2:-2}
    
    print_status "Using DeepSpeed ZeRO Stage $zero_stage"
    
    # Update DeepSpeed config with the specified stage
    sed -i "s/\"zero_stage\": [0-9]/\"zero_stage\": $zero_stage/" configs/accelerate/deepspeed.yaml
    
    local cmd="accelerate launch"
    cmd="$cmd --config_file configs/accelerate/deepspeed.yaml"
    cmd="$cmd 2_model_fusion/train_accelerate.py"
    cmd="$cmd --config configs/config.json"
    cmd="$cmd --data $data_path"
    cmd="$cmd --accelerate-config deepspeed"
    
    print_status "Command: $cmd"
    eval $cmd
}

# Function to run knowledge distillation
run_distillation() {
    print_header "Knowledge Distillation"
    
    local teacher_path=${1:-"2_model_fusion/checkpoints/final_model.pth"}
    local data_path=${2:-"../data/processed/training_data.json"}
    
    if [ ! -f "$teacher_path" ]; then
        print_error "Teacher model not found at $teacher_path"
        print_status "Please train the main model first or specify a valid teacher model path"
        return 1
    fi
    
    print_status "Teacher model: $teacher_path"
    print_status "Training data: $data_path"
    
    python 2_model_fusion/distillation.py \
        --teacher-model "$teacher_path" \
        --data-path "$data_path" \
        --output-dir "2_model_fusion/checkpoints/distilled" \
        --epochs 10 \
        --compression-ratio 4
}

# Function to run complete automated pipeline
run_complete_pipeline() {
    print_header "Complete Automated Pipeline"
    
    # Step 1: Setup and downloads
    print_status "Step 1: Environment setup and downloads"
    python run.py --setup
    
    if [ $? -ne 0 ]; then
        print_error "Setup failed"
        return 1
    fi
    
    # Step 2: Download models and data
    print_status "Step 2: Downloading models and datasets"
    python run.py --download-models
    python run.py --download-data
    
    # Step 3: Data processing
    print_status "Step 3: Data processing"
    python run.py --data-pipeline
    
    # Step 4: Determine training strategy based on available GPUs
    local gpu_count=$(check_gpu)
    
    if [ $gpu_count -gt 1 ]; then
        print_status "Step 4: Multi-GPU training"
        multi_gpu_training "../data/processed/training_data.json" $gpu_count false
    else
        print_status "Step 4: Single GPU training"  
        single_gpu_training "../data/processed/training_data.json" false
    fi
    
    # Step 5: Knowledge distillation
    print_status "Step 5: Knowledge distillation"
    run_distillation
    
    # Step 6: Evaluation
    print_status "Step 6: Model evaluation"
    python run.py --eval
    
    print_status "Complete pipeline finished!"
}

# Function to monitor training
monitor_training() {
    print_header "Training Monitor"
    
    # Start tensorboard if available
    if command -v tensorboard &> /dev/null; then
        print_status "Starting TensorBoard on port 6006"
        tensorboard --logdir logs/tensorboard --port 6006 &
        local tensorboard_pid=$!
        print_status "TensorBoard PID: $tensorboard_pid"
    fi
    
    # Monitor GPU usage
    print_status "Monitoring GPU usage (Ctrl+C to stop):"
    watch -n 2 nvidia-smi
}

# Main menu
show_menu() {
    clear
    print_header "FRA AI Fusion Training Launcher"
    echo "1) Complete Automated Pipeline"
    echo "2) Single GPU Training"
    echo "3) Multi-GPU Training" 
    echo "4) DeepSpeed Training"
    echo "5) Knowledge Distillation Only"
    echo "6) Monitor Training"
    echo "7) API Server"
    echo "8) System Status"
    echo "9) Exit"
    echo
}

# Main execution
main() {
    print_header "FRA AI Fusion Training System"
    
    # Initial checks
    check_requirements
    setup_hf_token
    check_gpu
    
    while true; do
        show_menu
        echo -n "Please select an option [1-9]: "
        read -r choice
        
        case $choice in
            1)
                run_complete_pipeline
                ;;
            2)
                echo -n "Enter data path (default: ../data/processed/training_data.json): "
                read -r data_path
                data_path=${data_path:-"../data/processed/training_data.json"}
                
                echo -n "Use 8-bit optimization? (y/n, default: n): "
                read -r use_8bit
                use_8bit=${use_8bit:-n}
                
                if [ "$use_8bit" = "y" ]; then
                    single_gpu_training "$data_path" true
                else
                    single_gpu_training "$data_path" false
                fi
                ;;
            3)
                echo -n "Enter data path (default: ../data/processed/training_data.json): "
                read -r data_path
                data_path=${data_path:-"../data/processed/training_data.json"}
                
                echo -n "Number of GPUs (default: 2): "
                read -r num_gpus
                num_gpus=${num_gpus:-2}
                
                echo -n "Use 8-bit optimization? (y/n, default: n): "
                read -r use_8bit
                use_8bit=${use_8bit:-n}
                
                if [ "$use_8bit" = "y" ]; then
                    multi_gpu_training "$data_path" "$num_gpus" true
                else
                    multi_gpu_training "$data_path" "$num_gpus" false
                fi
                ;;
            4)
                echo -n "Enter data path (default: ../data/processed/training_data.json): "
                read -r data_path
                data_path=${data_path:-"../data/processed/training_data.json"}
                
                echo -n "DeepSpeed ZeRO stage (1/2/3, default: 2): "
                read -r zero_stage
                zero_stage=${zero_stage:-2}
                
                deepspeed_training "$data_path" "$zero_stage"
                ;;
            5)
                echo -n "Enter teacher model path (default: 2_model_fusion/checkpoints/final_model.pth): "
                read -r teacher_path
                teacher_path=${teacher_path:-"2_model_fusion/checkpoints/final_model.pth"}
                
                echo -n "Enter data path (default: ../data/processed/training_data.json): "
                read -r data_path  
                data_path=${data_path:-"../data/processed/training_data.json"}
                
                run_distillation "$teacher_path" "$data_path"
                ;;
            6)
                monitor_training
                ;;
            7)
                print_status "Starting API server..."
                python run.py --serve
                ;;
            8)
                python run.py --status
                ;;
            9)
                print_status "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please select 1-9."
                ;;
        esac
        
        echo
        echo -n "Press Enter to continue..."
        read -r
    done
}

# Run main function
main "$@"
