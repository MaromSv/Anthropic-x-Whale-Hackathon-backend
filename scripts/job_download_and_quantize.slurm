#!/bin/bash
#SBATCH --partition=gpu_h100     # Use the MIG GPU partition
#SBATCH --gpus=1                 # Request 1 GPU
#SBATCH --job-name=anthropic # Name of the job
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=9        # Number of CPU cores per task
#SBATCH --time=02:00:00          # Maximum runtime
#SBATCH --output=outputs/slurm_outputs_A.out # Output log file

# Optional: load a Python module if your cluster requires it
# module load python/3.11
module purge
module load 2024

source "$HOME/whale-venv/bin/activate"

set -euo pipefail

export HF_TOKEN=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

# SLURM copies the script to its spool dir, so BASH_SOURCE[0] is unreliable.
# SLURM_SUBMIT_DIR is always set to the directory where sbatch was run.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL_ID="google/gemma-4-E2B-it"
MODEL_NAME="gemma-4-E2B-it"
MODEL_DIR="$REPO_DIR/models/$MODEL_NAME"
QUANT_DIR="$REPO_DIR/models/${MODEL_NAME}-quanto-int8"

mkdir -p "$REPO_DIR/outputs"
cd "$REPO_DIR"

TORCH_CUDA=$(python -c "import torch; print('.'.join(torch.version.cuda.split('.')[:2]))" 2>/dev/null || echo "")
if [ -n "$TORCH_CUDA" ] && [ -d "/usr/local/cuda-$TORCH_CUDA" ]; then
    export CUDA_HOME="/usr/local/cuda-$TORCH_CUDA"
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
fi
[ -n "${CUDA_HOME:-}" ] && echo "CUDA_HOME=$CUDA_HOME"

# ── Install deps ──────────────────────────────────────────
echo "=== [0/2] Installing Python dependencies ==="
pip install --quiet "numpy>=1.24,<2.0" optimum-quanto safetensors datasets


# HF_TOKEN must be set — either export it before submitting, or hardcode a
# read-only token here (never a write token in a shared script).
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Run: export HF_TOKEN=hf_..." >&2
    exit 1
fi

echo "======================================================"
echo "Job: $SLURM_JOB_ID   Host: $(hostname)   Date: $(date)"
echo "Repo: $REPO_DIR"
echo "======================================================"

# ── Step 1: Download ──────────────────────────────────────
# echo ""
# echo "=== [1/2] Downloading $MODEL_ID ==="
# python scripts/download_gemma.py \
#     --model "$MODEL_ID" \
#     --output "$REPO_DIR/models"

# echo "Download complete. Size on disk:"
# du -sh "$MODEL_DIR"

# ── Step 2: Quantize to int4 ──────────────────────────────
echo ""
echo "=== [2/2] Quantizing to int4 with Quanto ==="
# Peaks at ~2x FP16 size (~10-12 GB) during quantization.
# Use CUDA since we're on a GPU node; fall back to cpu if not available.
DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
echo "Using device: $DEVICE"
python -m scripts.quantize_quanto \
    --model "$MODEL_DIR" \
    --output "$QUANT_DIR" \
    --bits int4 \
    --device "$DEVICE"

echo ""
echo "======================================================"
echo "Done. Quantized model at: $QUANT_DIR"
echo ""
echo "Next step — run the eval:"
echo "  python scripts/eval_quantized.py --model $QUANT_DIR"
echo "======================================================"
