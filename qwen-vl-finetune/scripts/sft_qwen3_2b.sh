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
batch_size=${BATCH_SIZE:-8}
grad_accum_steps=${GRAD_ACCUM:-2}
num_epochs=${EPOCHS:-2}

# === Training entry point ===
entry_file=qwenvl/train/train_qwen.py

# === Dataset configuration ===
datasets=${DATASETS:-"tarsier2_recap"}
data_path=${DATA_PATH:-"/home/mars_rover/model_disk/data/tarsier2_recap_train.json"}

# === Output configuration ===
run_name=${RUN_NAME:-"chotavlm-v0-2b"}
output_dir=${OUTPUT_DIR:-"./output/${run_name}"}

# === Eval parameters (configurable) ===
eval_steps=${EVAL_STEPS:-0}                 # 0: disabled (eval runs on separate server)
eval_mode=${EVAL_MODE:-"subset"}            # "subset" or "full"
eval_subset_size=${EVAL_SUBSET_SIZE:-200}   # samples per benchmark in subset mode
eval_benchmarks=${EVAL_BENCHMARKS:-"dream1k,carebench,et_bench_dvc,et_bench_tvg,activitynet,next_gqa,videomme,mmmu,realworldqa,mathvision,odinw13"}

# === Checkpoint parameters ===
save_steps=${SAVE_STEPS:-2000}              # R: save checkpoint every R steps
save_total_limit=${SAVE_TOTAL_LIMIT:-}       # empty: keep all checkpoints

# === Cache ===
export HF_HOME=${HF_HOME:-"/home/mars_rover/model_disk/hf_cache"}
export TMPDIR=${TMPDIR:-"/home/mars_rover/model_disk/tmp"}

# === Training arguments ===
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_path ${data_path} \
    --data_flatten True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --video_max_frames 16 \
    --video_min_frames 4 \
    --video_fps 2 \
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
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb \
    --eval_steps_custom ${eval_steps} \
    --eval_mode ${eval_mode} \
    --eval_subset_size ${eval_subset_size} \
    --eval_benchmarks ${eval_benchmarks} \
    --sample_log_steps 500 \
    --sample_log_count 3"

echo "=== Training Configuration ==="
echo "Model: ${llm}"
echo "GPUs: ${NPROC_PER_NODE}"
echo "Batch: ${batch_size}/gpu × ${grad_accum_steps} accum × ${NPROC_PER_NODE} gpus = $((batch_size * grad_accum_steps * NPROC_PER_NODE)) effective"
echo "LR: ${lr}, Epochs: ${num_epochs}"
echo "Eval: every ${eval_steps} steps, mode=${eval_mode}, subset_size=${eval_subset_size}"
echo "Checkpoints: every ${save_steps} steps, keep last ${save_total_limit}"
echo "Output: ${output_dir}"
echo "=============================="

# === Launch training ===
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
