#!/bin/bash
# ============================================================================
# Full Pipeline V3 Runner Script
# ============================================================================
# Usage:
#   bash full_pipeline_v3/run.sh single one_sample.jsonl 0
#   bash full_pipeline_v3/run.sh single input-output/input.txt 0
#   bash full_pipeline_v3/run.sh batch one_sample_test10.jsonl 0,1

MODE=${1:-single}
INPUT=${2:-one_sample.jsonl}
GPU=${3:-0}
EXTRA_ARGS="${@:4}"

cd "$(dirname "$0")/.."

echo "============================================"
echo "  Full Pipeline V3"
echo "  Mode: $MODE"
echo "  Input: $INPUT"
echo "  GPU(s): $GPU"
echo "============================================"

python -m full_pipeline_v3 --mode "$MODE" --input "$INPUT" --gpu "$GPU" $EXTRA_ARGS
