#!/bin/bash
# =============================================================================
# Setup script for the eval server
#
# Installs everything needed to run eval on a fresh Ubuntu server with GPU.
#
# Usage:
#   bash scripts/setup_eval_server.sh <work_dir> [train_server] [remote_base]
#
# Args:
#   work_dir        - Local directory to set up eval workspace
#   train_server    - SSH host for training server (default: mars3)
#   remote_base     - Remote path to Qwen3-VL dir (default: /home/mars_rover/model_disk/Qwen3-VL)
#
# Examples:
#   bash scripts/setup_eval_server.sh ~/eval_workspace
#   bash scripts/setup_eval_server.sh /mnt/data/eval mars3 /home/mars_rover/model_disk/Qwen3-VL
#
# Environment variables (optional overrides):
#   CONDA_DIR       - Where to install miniconda (default: ~/miniconda3)
#   ENV_NAME        - Conda env name (default: qwen3vl)
#   HF_CACHE        - HuggingFace cache dir (default: ~/hf_cache)
# =============================================================================
set -eo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/setup_eval_server.sh <work_dir> [train_server] [remote_base]"
    echo "Example: bash scripts/setup_eval_server.sh ~/eval_workspace mars3 /home/mars_rover/model_disk/Qwen3-VL"
    exit 1
fi

WORK_DIR="$1"
TRAIN_SERVER="${2:-mars3}"
REMOTE_BASE="${3:-/home/mars_rover/model_disk/Qwen3-VL}"
CONDA_DIR=${CONDA_DIR:-"$HOME/miniconda3"}
ENV_NAME=${ENV_NAME:-"qwen3vl"}
HF_CACHE=${HF_CACHE:-"$HOME/hf_cache"}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-"${WORK_DIR}/pip_cache"}

export HF_HOME="${HF_CACHE}"
export TMPDIR="${PIP_CACHE_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
mkdir -p "${PIP_CACHE_DIR}"

echo "=== Eval Server Setup ==="
echo "Conda: ${CONDA_DIR}"
echo "Env: ${ENV_NAME}"
echo "Workspace: ${WORK_DIR}"
echo "HF Cache: ${HF_CACHE}"
echo "PIP Cache: ${PIP_CACHE_DIR}"
echo "========================="

# --- Step 1: Install Miniconda ---
if [ ! -f "${CONDA_DIR}/bin/conda" ]; then
    echo ""
    echo "[1/8] Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
    rm /tmp/miniconda.sh
    eval "$(${CONDA_DIR}/bin/conda shell.bash hook)"
    conda init bash
    echo "[1/8] Done"
else
    echo "[1/8] Miniconda already installed"
    eval "$(${CONDA_DIR}/bin/conda shell.bash hook)"
fi

# --- Step 2: Create conda environment ---
if ! conda env list | grep -q "${ENV_NAME}"; then
    echo ""
    echo "[2/8] Creating conda env: ${ENV_NAME}..."
    conda create -n "${ENV_NAME}" python=3.12 -y
    echo "[2/8] Done"
else
    echo "[2/8] Conda env ${ENV_NAME} already exists"
fi

conda activate "${ENV_NAME}"

# --- Step 3: Install PyTorch ---
echo ""
echo "[3/8] Installing PyTorch..."
pip install torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0

# --- Step 4: Install dependencies ---
echo ""
echo "[4/8] Installing dependencies..."
pip install transformers accelerate
pip install qwen-vl-utils
pip install decord
pip install openai   # for GPT-4o judge

# --- Step 5: Install flash-attention ---
echo ""
echo "[5/8] Installing flash-attention (this takes 10-15 min)..."
# Detect CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "Detected CUDA: ${CUDA_VERSION}"

# Install matching CUDA toolkit for compilation
CUDA_MAJOR=$(echo "${CUDA_VERSION}" | cut -d. -f1)
CUDA_MINOR=$(echo "${CUDA_VERSION}" | cut -d. -f2)
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
echo "PyTorch CUDA: ${TORCH_CUDA}"

# If CUDA versions differ, install matching toolkit
if [ "${TORCH_CUDA}" != "${CUDA_MAJOR}.${CUDA_MINOR}" ]; then
    echo "Installing CUDA toolkit ${TORCH_CUDA} to match PyTorch..."
    TORCH_CUDA_MAJOR=$(echo "${TORCH_CUDA}" | cut -d. -f1)
    TORCH_CUDA_MINOR=$(echo "${TORCH_CUDA}" | cut -d. -f2)
    conda install -c nvidia "cuda-toolkit=${TORCH_CUDA_MAJOR}.${TORCH_CUDA_MINOR}" -y
    export CUDA_HOME="${CONDA_PREFIX}"
fi

MAX_JOBS=8 pip install flash-attn --no-cache-dir --no-binary flash-attn --no-build-isolation --no-deps

# --- Step 6: Sync entire Qwen3-VL directory from training server ---
echo ""
echo "[6/8] Syncing ${REMOTE_BASE} from ${TRAIN_SERVER}..."
mkdir -p "${WORK_DIR}"

rsync -az --progress \
    --exclude="output/*/checkpoint-*" \
    --exclude="*.pt" \
    --exclude="*.bin" \
    --exclude="__pycache__" \
    --exclude=".git" \
    "${TRAIN_SERVER}:${REMOTE_BASE}/" \
    "${WORK_DIR}/"

echo "[6/8] Synced to ${WORK_DIR}"

# --- Step 7: Download base model ---
echo ""
echo "[7/8] Downloading Qwen3-VL-2B-Instruct..."
export HF_HOME="${HF_CACHE}"
python -c "
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
print('Downloading model...')
Qwen3VLForConditionalGeneration.from_pretrained('Qwen/Qwen3-VL-2B-Instruct', cache_dir='${HF_CACHE}')
AutoProcessor.from_pretrained('Qwen/Qwen3-VL-2B-Instruct', cache_dir='${HF_CACHE}')
print('Done')
"

# --- Step 8: Verify eval datasets ---
echo ""
LOCAL_EVAL="${WORK_DIR}/evaluation_set"
if [ -d "${LOCAL_EVAL}" ]; then
    echo "[8/8] Eval datasets found at ${LOCAL_EVAL}"
    ls -d "${LOCAL_EVAL}"/*/ 2>/dev/null | while read d; do echo "  $(basename $d)"; done
else
    echo "[8/8] WARNING: Eval datasets not found at ${LOCAL_EVAL}"
    echo "  They should have been synced in step 6. Check REMOTE_BASE structure."
fi

# --- Create env file ---
LOCAL_EVAL="${WORK_DIR}/evaluation_set"
cat > "${WORK_DIR}/.env" << EOF
export HF_HOME="${HF_CACHE}"
export CONDA_DIR="${CONDA_DIR}"
export ENV_NAME="${ENV_NAME}"
export TRAIN_SERVER="${TRAIN_SERVER}"
export REMOTE_BASE="${REMOTE_BASE}"

# Eval dataset paths
export DREAM1K_PATH="${LOCAL_EVAL}/dream1k"
export CAREBENCH_PATH="${LOCAL_EVAL}/carebench"
export ET_BENCH_PATH="${LOCAL_EVAL}/et_bench"
export ACTIVITYNET_PATH="${LOCAL_EVAL}/activitynet"
export NEXT_GQA_PATH="${LOCAL_EVAL}/next_gqa"

# OpenAI API key for GPT-4o judge (fill in)
export OPENAI_API_KEY=""
export OPENAI_ENDPOINT="https://banim.cognitiveservices.azure.com/openai/v1/"
export OPENAI_DEPLOYMENT="gpt-4o"
EOF

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit ${WORK_DIR}/.env and set OPENAI_API_KEY"
echo "  2. Source the env:  source ${WORK_DIR}/.env"
echo "  3. Activate conda:  conda activate ${ENV_NAME}"
echo "  4. Start eval watcher:"
echo "     cd ${WORK_DIR}"
echo "     bash scripts/eval_watcher.sh"
echo ""
echo "  Or run eval manually:"
echo "     cd ${WORK_DIR}"
echo "     python run_eval.py --model /path/to/checkpoint --samples 10 --verbose"
echo ""
