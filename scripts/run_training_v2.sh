#!/bin/bash
# ============================================================================
# Training Launcher V2 for DGX (8x V100)
# Uses aligned prompts with full_pipeline inference
# Output: output_v2/
#
# Fault-tolerant: handles SIGTERM, OOM, and unexpected crashes.
#   - Catches signals → forwards to training process for emergency save
#   - Detects free GPUs → auto-resumes on available hardware
#   - Retries up to MAX_RETRIES times after recoverable failures
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PACKAGE_DIR"

MAX_RETRIES=${MAX_RETRIES:-3}
RETRY_DELAY=${RETRY_DELAY:-30}          # seconds between retries
MIN_FREE_MEM_MB=${MIN_FREE_MEM_MB:-12000}  # 12 GB free threshold
RUN_STATE_FILE="output_v2/run_state.json"

# ── helpers ──────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log_err() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

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
mkdir -p output_v2

# ── GPU discovery ────────────────────────────────────────────────────────────

detect_free_gpus() {
    # Returns a comma-separated list of GPU IDs that have >= MIN_FREE_MEM_MB
    # free memory. Existing compute processes are logged, but not used for filtering.
    local free_ids=""
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)

    log_err "GPU scan (threshold: ${MIN_FREE_MEM_MB} MiB free):"
    for (( i=0; i<gpu_count; i++ )); do
        # GPU name
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$i" | tr -d '\n')
        # Total / Free memory in MiB
        local total_mem free_mem
        total_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$i" | tr -d ' ')
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$i" | tr -d ' ')
        # Number of compute processes on this GPU
        local gpu_procs
        gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$i" 2>/dev/null | grep -c '[0-9]' || true)

        local status
        if [[ "$free_mem" -ge "$MIN_FREE_MEM_MB" ]]; then
            status="SELECTED"
            if [[ -n "$free_ids" ]]; then
                free_ids="${free_ids},$i"
            else
                free_ids="$i"
            fi
        else
            status="LOW MEM"
        fi
        log_err "  GPU $i: $gpu_name | ${free_mem}/${total_mem} MiB free | ${gpu_procs} procs | $status"
    done
    echo "$free_ids"
}

describe_gpu_ids() {
    local ids="$1"
    local details=""
    local id
    for id in $(echo "$ids" | tr ',' ' '); do
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$id" | tr -d '\n')
        if [[ -n "$details" ]]; then
            details="${details}; GPU ${id}: ${gpu_name}"
        else
            details="GPU ${id}: ${gpu_name}"
        fi
    done
    echo "$details"
}

# ── Signal forwarding ───────────────────────────────────────────────────────

TRAIN_PID=""

forward_signal() {
    local sig=$1
    log "Launcher caught SIG${sig} — forwarding to training PID ${TRAIN_PID:-??}"
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -s "$sig" "$TRAIN_PID"
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
    exit 1
}

trap 'forward_signal TERM' SIGTERM
trap 'forward_signal INT'  SIGINT

# ── ClearML credentials ─────────────────────────────────────────────────────

export CLEARML_WEB_HOST=https://app.clear.ml/
export CLEARML_API_HOST=https://api.clear.ml
export CLEARML_FILES_HOST=https://files.clear.ml
export CLEARML_API_ACCESS_KEY=CIXHLPFI1GE34MQMG7FC1SREQWND6L
export CLEARML_API_SECRET_KEY=QzBh_QD7WF38cQDjEknMCZ2G-qf3VHarngveIXvvmxbd12vh5-SQVvGqu7s6_zBH6tM

# Environment variables for optimal performance
export NCCL_P2P_LEVEL=NVL  # Use NVLink
export NCCL_IB_DISABLE=0   # Enable InfiniBand if available

# ── Main retry loop ─────────────────────────────────────────────────────────

attempt=0

while (( attempt < MAX_RETRIES )); do
    attempt=$((attempt + 1))

    # If a prior run already completed successfully, skip
    if [[ -f "$RUN_STATE_FILE" ]]; then
        prior_status=$(python3 -c "import json; print(json.load(open('$RUN_STATE_FILE'))['status'])" 2>/dev/null || echo "unknown")
        if [[ "$prior_status" == "completed" ]]; then
            log "Previous run already completed. Nothing to do."
            exit 0
        fi
        log "Previous run status: $prior_status — will attempt resume."
    fi

    # Discover free GPUs
    FREE_GPUS=$(detect_free_gpus)
    if [[ -z "$FREE_GPUS" ]]; then
        log "ERROR: No GPUs matched selection rule (need >= ${MIN_FREE_MEM_MB} MiB free). Retrying in ${RETRY_DELAY}s..."
        sleep "$RETRY_DELAY"
        continue
    fi

    NUM_GPUS=$(echo "$FREE_GPUS" | tr ',' '\n' | wc -l)
    export CUDA_VISIBLE_DEVICES="$FREE_GPUS"
    SELECTED_GPU_DETAILS=$(describe_gpu_ids "$FREE_GPUS")

    log "=============================================="
    log "Attempt $attempt / $MAX_RETRIES"
    log "=============================================="
    log ""
    log "Configuration:"
    log "  Selected GPUs: $FREE_GPUS  ($NUM_GPUS devices)"
    log "  Selected GPU details: $SELECTED_GPU_DETAILS"
    log "  Selection rule: >= ${MIN_FREE_MEM_MB} MiB free"
    log "  Model: microsoft/Phi-3.5-mini-instruct"
    log "  Mixed Precision: FP16"
    log "  Output: output_v2/"
    log "  Prompt Version: V2 (aligned with full_pipeline)"
    log ""

    # Launch training in background so we can capture PID for signal forwarding
    accelerate launch \
        --config_file scripts/accelerate_config.yaml \
        --gpu_ids all \
        --num_processes "$NUM_GPUS" \
        scripts/train_dgx_v2.py &
    TRAIN_PID=$!

    # Wait for training to finish and capture exit code
    set +e
    wait "$TRAIN_PID"
    EXIT_CODE=$?
    set -e
    TRAIN_PID=""

    if [[ "$EXIT_CODE" -eq 0 ]]; then
        log ""
        log "=============================================="
        log "Training complete!"
        log "Models saved to: output_v2/"
        log "=============================================="
        exit 0
    fi

    # -- Diagnose failure --
    log "Training exited with code $EXIT_CODE"

    run_status="unknown"
    if [[ -f "$RUN_STATE_FILE" ]]; then
        run_status=$(python3 -c "import json; print(json.load(open('$RUN_STATE_FILE'))['status'])" 2>/dev/null || echo "unknown")
    fi

    case "$run_status" in
        oom)
            log "OOM detected. Will retry with freshly detected free GPUs."
            ;;
        interrupted)
            log "Interrupted (SIGTERM/SIGINT). Will attempt resume from checkpoint."
            ;;
        *)
            log "Run state: $run_status (exit code $EXIT_CODE). Will attempt resume."
            ;;
    esac

    if (( attempt < MAX_RETRIES )); then
        log "Waiting ${RETRY_DELAY}s before retry..."
        sleep "$RETRY_DELAY"
    fi
done

log "Exhausted $MAX_RETRIES retries. Check output_v2/run_state.json for details."
exit 1
