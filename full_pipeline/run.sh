#!/bin/bash
# ============================================================================
# Full Pipeline Runner Script
# ============================================================================
# Usage:
#   bash full_pipeline/run.sh single one_sample.jsonl 0
#   bash full_pipeline/run.sh batch one_sample_test10.jsonl 0 5
#   bash full_pipeline/run.sh single one_sample.jsonl 0 --use-existing-labels
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PACKAGE_DIR"

# Activate venv if it exists
if [ -f "dgx_env/bin/activate" ]; then
    source dgx_env/bin/activate
fi

MODE="${1:-single}"
INPUT="${2:-one_sample.jsonl}"
GPU="${3:-0}"
LIMIT="${4:-}"
EXTRA_ARGS="${@:5}"

echo "============================================"
echo "  Full Inference Pipeline"
echo "  Mode:  $MODE"
echo "  Input: $INPUT"
echo "  GPU:   $GPU"
echo "============================================"

CMD="python -m full_pipeline.run --mode $MODE --input $INPUT --gpu $GPU"

if [ -n "$LIMIT" ] && [ "$LIMIT" != "--use-existing-labels" ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Pass through any extra args
if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

echo "Running: $CMD"
eval $CMD
