#!/bin/bash
# ============================================================================
# DGX Environment Setup Script
# Clinical Factor Extraction - Mental Health EHR Analysis
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "DGX Environment Setup for BioMistral Training"
echo "=============================================="
echo ""

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA drivers not installed."
    exit 1
fi

echo "Detected GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.10" && "$PYTHON_VERSION" != "3.11" ]]; then
    echo "WARNING: Recommended Python 3.10 or 3.11"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
cd "$PACKAGE_DIR"

if [ -d "dgx_env" ]; then
    echo "Virtual environment 'dgx_env' already exists."
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf dgx_env
        python3 -m venv dgx_env
    fi
else
    python3 -m venv dgx_env
fi

source dgx_env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo ""
echo "Installing training dependencies..."
pip install \
    transformers>=4.40.0 \
    datasets>=2.18.0 \
    accelerate>=0.28.0 \
    peft>=0.10.0 \
    trl>=0.8.0 \
    bitsandbytes>=0.43.0 \
    scipy \
    sentencepiece \
    protobuf \
    clearml>=1.14.0 \
    psutil \
    tensorboard

# Install flash-attention for V100 (optional, improves speed)
echo ""
echo "Installing Flash Attention (optional)..."
pip install flash-attn --no-build-isolation 2>/dev/null || {
    echo "Flash Attention installation failed (may not be critical)"
    echo "Training will use standard attention."
}

# Install deepspeed (optional, for ZeRO optimization)
echo ""
echo "Installing DeepSpeed..."
pip install deepspeed

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
import transformers
import peft
import accelerate

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'Accelerate: {accelerate.__version__}')

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)')
"

# Create output directory
mkdir -p "$PACKAGE_DIR/output"
mkdir -p "$PACKAGE_DIR/training_data"

# Create accelerate config
echo ""
echo "Creating accelerate configuration..."
cat > "$SCRIPT_DIR/accelerate_config.yaml" << 'EOF'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source dgx_env/bin/activate"
echo "  2. Copy training data: ./scripts/copy_data.sh /path/to/training_data"
echo "  3. Start training: ./scripts/run_training.sh"
echo ""
