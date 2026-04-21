#!/bin/bash
# =============================================================================
# Eval Watcher — watches training server for new checkpoints, runs eval
#
# Usage:
#   bash scripts/eval_watcher.sh <remote_path> [checkpoint_pattern]
#
# Args:
#   remote_path          - Full SSH path: server:/path/to/checkpoint/dir
#                          e.g. mars3:/home/mars_rover/model_disk/Qwen3-VL/qwen-vl-finetune/output/chotavlm-v0-2b
#   checkpoint_pattern   - Glob pattern for checkpoints (default: checkpoint-*)
#
# Environment variables:
#   LOCAL_CHECKPOINT    - Local dir to sync checkpoints (default: ./checkpoints)
#   EVAL_OUTPUT_DIR     - Where to write eval results (default: ./eval_results)
#   POLL_INTERVAL       - Seconds between polls (default: 300 = 5 min)
#   EVAL_SAMPLES        - Samples per benchmark (default: 10, 0=all)
#   EVAL_BENCHMARKS     - Benchmarks to run (default: all)
#   EVAL_BATCH_SIZE     - Batch size (default: 4)
#
# Examples:
#   bash scripts/eval_watcher.sh mars3:/home/mars_rover/model_disk/Qwen3-VL/qwen-vl-finetune/output/chotavlm-v0-2b
#   bash scripts/eval_watcher.sh mars3:/data/training/run1 "checkpoint-*"
#   bash scripts/eval_watcher.sh mars3:/data/training/run1 "step-*"
#   POLL_INTERVAL=60 EVAL_SAMPLES=0 bash scripts/eval_watcher.sh mars3:/path/to/checkpoints
# =============================================================================
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/eval_watcher.sh <server:/path/to/checkpoint/dir> [checkpoint_pattern]"
    echo "Example: bash scripts/eval_watcher.sh mars3:/home/mars_rover/model_disk/Qwen3-VL/qwen-vl-finetune/output/chotavlm-v0-2b"
    exit 1
fi

# Parse remote_path into server and directory
REMOTE_FULL="$1"
TRAIN_SERVER="${REMOTE_FULL%%:*}"
REMOTE_DIR="${REMOTE_FULL#*:}"
CHECKPOINT_REGEX="${2:-checkpoint-*}"

LOCAL_CHECKPOINT=${LOCAL_CHECKPOINT:-"./checkpoints"}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"./eval_results"}
POLL_INTERVAL=${POLL_INTERVAL:-300}
EVAL_SAMPLES=${EVAL_SAMPLES:-10}
EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-"all"}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
EVALUATED_LOG="${EVAL_OUTPUT_DIR}/.evaluated_checkpoints"

mkdir -p "${LOCAL_CHECKPOINT}" "${EVAL_OUTPUT_DIR}"
touch "${EVALUATED_LOG}"

# Source .env if it exists (benchmark paths, HF_HOME, API keys)
ENV_FILE="${ENV_FILE:-$(dirname "$0")/../.env}"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
    echo "Sourced env from ${ENV_FILE}"
fi

echo "=== Eval Watcher ==="
echo "Training server: ${TRAIN_SERVER}"
echo "Remote dir: ${REMOTE_DIR}"
echo "Checkpoint pattern: ${CHECKPOINT_REGEX}"
echo "Local checkpoint dir: ${LOCAL_CHECKPOINT}"
echo "Poll interval: ${POLL_INTERVAL}s"
echo "Eval config: samples=${EVAL_SAMPLES}, benchmarks=${EVAL_BENCHMARKS}, batch_size=${EVAL_BATCH_SIZE}"
echo "===================="

get_remote_checkpoints() {
    ssh "${TRAIN_SERVER}" "ls -d ${REMOTE_DIR}/${CHECKPOINT_REGEX} 2>/dev/null | sort -t- -k2 -n" 2>/dev/null || true
}

is_evaluated() {
    local ckpt_name="$1"
    grep -qF "${ckpt_name}" "${EVALUATED_LOG}" 2>/dev/null
}

mark_evaluated() {
    local ckpt_name="$1"
    echo "${ckpt_name}" >> "${EVALUATED_LOG}"
}

is_checkpoint_complete() {
    # Check if checkpoint has finished writing (trainer_state.json exists)
    local remote_path="$1"
    ssh "${TRAIN_SERVER}" "test -f ${remote_path}/trainer_state.json" 2>/dev/null
}

sync_checkpoint() {
    local remote_path="$1"
    local ckpt_name="$2"
    local local_path="${LOCAL_CHECKPOINT}/${ckpt_name}"

    echo "[SYNC] ${ckpt_name} from ${TRAIN_SERVER}..."
    rsync -az --progress \
        "${TRAIN_SERVER}:${remote_path}/" \
        "${local_path}/"
    echo "[SYNC] Done: ${local_path}"
}

run_eval() {
    local ckpt_name="$1"
    local local_path="${LOCAL_CHECKPOINT}/${ckpt_name}"
    local output_file="${EVAL_OUTPUT_DIR}/${ckpt_name}.md"
    local step=$(echo "${ckpt_name}" | grep -oP '\d+')

    echo ""
    echo "============================================"
    echo "[EVAL] Running: ${ckpt_name} (step ${step})"
    echo "============================================"

    python run_eval.py \
        --model "${local_path}" \
        --samples "${EVAL_SAMPLES}" \
        --benchmarks "${EVAL_BENCHMARKS}" \
        --batch-size "${EVAL_BATCH_SIZE}" \
        --verbose \
        --output "${output_file}" \
        2>&1 | tee "${EVAL_OUTPUT_DIR}/${ckpt_name}.log"

    echo ""
    echo "[EVAL] Results written to: ${output_file}"
    echo "[EVAL] Scores:"
    grep "Scores:" "${output_file}" 2>/dev/null || echo "  (no scores found)"
    echo ""
}

cleanup_old_checkpoints() {
    # Keep only the last 3 synced checkpoints to save disk
    local keep=3
    local count=$(ls -d "${LOCAL_CHECKPOINT}"/checkpoint-* 2>/dev/null | wc -l)
    if [ "${count}" -gt "${keep}" ]; then
        ls -dt "${LOCAL_CHECKPOINT}"/checkpoint-* | tail -n +$((keep + 1)) | while read old; do
            echo "[CLEANUP] Removing old checkpoint: $(basename ${old})"
            rm -rf "${old}"
        done
    fi
}

# === Main loop ===
echo ""
echo "Starting watcher loop (Ctrl+C to stop)..."
echo ""

while true; do
    remote_checkpoints=$(get_remote_checkpoints)

    if [ -z "${remote_checkpoints}" ]; then
        echo "[$(date '+%H:%M:%S')] No checkpoints found. Waiting ${POLL_INTERVAL}s..."
        sleep "${POLL_INTERVAL}"
        continue
    fi

    new_found=0
    while IFS= read -r remote_path; do
        ckpt_name=$(basename "${remote_path}")

        # Skip already evaluated
        if is_evaluated "${ckpt_name}"; then
            continue
        fi

        # Check if checkpoint is fully written
        if ! is_checkpoint_complete "${remote_path}"; then
            echo "[$(date '+%H:%M:%S')] ${ckpt_name} still being written, skipping..."
            continue
        fi

        new_found=1
        echo "[$(date '+%H:%M:%S')] New checkpoint found: ${ckpt_name}"

        # Sync from training server
        sync_checkpoint "${remote_path}" "${ckpt_name}"

        # Run eval
        run_eval "${ckpt_name}"

        # Mark as done
        mark_evaluated "${ckpt_name}"

        # Cleanup old
        cleanup_old_checkpoints

    done <<< "${remote_checkpoints}"

    if [ "${new_found}" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] No new checkpoints. Waiting ${POLL_INTERVAL}s..."
    fi

    sleep "${POLL_INTERVAL}"
done
