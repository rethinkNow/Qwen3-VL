#!/bin/bash
set -e

# === Distributed training configuration ===
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
NNODES=${WORLD_SIZE:-1}

# === DeepSpeed configuration ===
deepspeed=./scripts/zero2_2gpu.json

# === Model configuration ===
llm=${MODEL:-"Qwen/Qwen3-VL-2B-Instruct"}

# === Training hyperparameters ===
lr=${LR:-2e-5}
batch_size=${BATCH_SIZE:-2}
grad_accum_steps=${GRAD_ACCUM:-8}
num_epochs=${EPOCHS:-2}

# === Video / context configuration ===
# video_max_pixels = per-frame H*W cap. 200704 = 448*448.
video_max_frames=${VIDEO_MAX_FRAMES:-720}
video_min_frames=${VIDEO_MIN_FRAMES:-4}
video_fps=${VIDEO_FPS:-2}
max_pixels=${MAX_PIXELS:-200704}
min_pixels=${MIN_PIXELS:-784}
video_max_pixels=${VIDEO_MAX_PIXELS:-200704}
video_min_pixels=${VIDEO_MIN_PIXELS:-784}
model_max_length=${MODEL_MAX_LENGTH:-131072}

# === Training entry point ===
entry_file=qwenvl/train/train_qwen.py

# === Dataset configuration ===
datasets=${DATASETS:-"chotavlm_webvid,chotavlm_charades_ego,chotavlm_charades"}

# === Output configuration ===
run_name=${RUN_NAME:-"chotavlm-v0-2b"}
output_dir=${OUTPUT_DIR:-"/home/mars_rover/model_disk/checkpoints/${run_name}"}

# === Checkpoint parameters ===
save_steps=${SAVE_STEPS:-500}
save_total_limit=${SAVE_TOTAL_LIMIT:-}       # empty: keep all checkpoints

# === Cache ===
export HF_HOME=${HF_HOME:-"/home/mars_rover/model_disk/hf_cache"}
export TMPDIR=${TMPDIR:-"/home/mars_rover/model_disk/tmp"}
export WANDB_DIR=${WANDB_DIR:-"/home/mars_rover/model_disk/wandb"}
mkdir -p "${WANDB_DIR}"

# Reclaim fragmented CUDA memory (frees ~9GB of "reserved but unallocated" seen in prior OOM).
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

# === Training arguments ===
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --video_max_pixels ${video_max_pixels} \
    --video_min_pixels ${video_min_pixels} \
    --video_max_frames ${video_max_frames} \
    --video_min_frames ${video_min_frames} \
    --video_fps ${video_fps} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps ${save_steps} \
    ${save_total_limit:+--save_total_limit $save_total_limit} \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

echo "=== Training Configuration ==="
echo "Model: ${llm}"
echo "GPUs: ${NPROC_PER_NODE}"
echo "Batch: ${batch_size}/gpu x ${grad_accum_steps} accum x ${NPROC_PER_NODE} gpus = $((batch_size * grad_accum_steps * NPROC_PER_NODE)) effective"
echo "LR: ${lr}, Epochs: ${num_epochs}"
echo "Video: fps=${video_fps}, max_frames=${video_max_frames}, per-frame cap=${video_max_pixels} (448x448)"
echo "Checkpoints: every ${save_steps} steps, keep last ${save_total_limit:-all}"
echo "Output: ${output_dir}"
echo "=============================="

# === Launch training ===
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
