#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PACKAGE_DIR"

source dgx_env/bin/activate

# ================= CLEARML =================
export CLEARML_WEB_HOST=https://app.clear.ml/
export CLEARML_API_HOST=https://api.clear.ml
export CLEARML_FILES_HOST=https://files.clear.ml
export CLEARML_API_ACCESS_KEY=CIXHLPFI1GE34MQMG7FC1SREQWND6L
export CLEARML_API_SECRET_KEY=QzBh_QD7WF38cQDjEknMCZ2G-qf3VHarngveIXvvmxbd12vh5-SQVvGqu7s6_zBH6tM

# ================= GPU CONFIG =================
# ✅ Use GPUs 2, 6, 7
export CUDA_VISIBLE_DEVICES=2,6,7

# Optional DGX performance tuning
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export NVIDIA_TF32_OVERRIDE=1

echo "=============================================="
echo "Starting Evaluation on GPUs: 2,6,7"
echo "=============================================="

# Show GPU mapping
nvidia-smi

echo ""
echo "Running evaluation..."
echo ""

python scripts/evaluate.py \
    --model_path output/checkpoint-1500 \
    --test_data training_data/test.jsonl \
    --output_dir output/eval_results_1500 \
    --max_samples 100

echo ""
echo "Evaluation completed!"