#!/bin/bash
# ============================================================================
# Training Launcher for DGX (8x V100)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PACKAGE_DIR"

# Check environment
if [ ! -d "dgx_env" ]; then
    echo "ERROR: Virtual environment not found. Run setup_env.sh first."
    exit 1
fi

source dgx_env/bin/activate

# Check training data
if [ ! -f "training_data/train.jsonl" ]; then
    echo "ERROR: Training data not found. Run copy_data.sh first."
    exit 1
fi

# Create output directory
mkdir -p output

echo "=============================================="
echo "Starting DGX Training: Clinical Factor Extraction"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  GPUs: $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"
echo "  Model: microsoft/Phi-3.5-mini-instruct"
echo "  Mixed Precision: FP16"
echo ""

# ClearML credentials
export CLEARML_WEB_HOST=https://app.clear.ml/
export CLEARML_API_HOST=https://api.clear.ml
export CLEARML_FILES_HOST=https://files.clear.ml
export CLEARML_API_ACCESS_KEY=CIXHLPFI1GE34MQMG7FC1SREQWND6L
export CLEARML_API_SECRET_KEY=QzBh_QD7WF38cQDjEknMCZ2G-qf3VHarngveIXvvmxbd12vh5-SQVvGqu7s6_zBH6tM

# Environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=1,2,3,4,6,7
export NCCL_P2P_LEVEL=NVL  # Use NVLink
export NCCL_IB_DISABLE=0   # Enable InfiniBand if available
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Enable TF32 for faster matmul on V100
export NVIDIA_TF32_OVERRIDE=1

# Cache settings
export HF_HOME="$PACKAGE_DIR/.cache/huggingface"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_CACHE="$PACKAGE_DIR/.cache/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# Launch with accelerate for multi-GPU
echo "Launching training with accelerate..."
echo "Log file: output/training.log"
echo ""

accelerate launch \
    --config_file "$SCRIPT_DIR/accelerate_config.yaml" \
    --num_processes 6 \
    --mixed_precision fp16 \
    "$SCRIPT_DIR/train_dgx.py" \
    2>&1 | tee output/training.log

echo ""
echo "Training complete!"
echo "Models saved to: output/best_model/ and output/latest_model/"
echo "View experiment at: https://app.clear.ml/"
