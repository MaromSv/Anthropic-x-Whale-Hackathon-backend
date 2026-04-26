#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=anthropic-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm_eval_%A.out

module purge
module load 2024

source "$HOME/whale-venv/bin/activate"

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# Set CUDA_HOME using the CUDA version torch was built against.
TORCH_CUDA=$(python -c "import torch; print('.'.join(torch.version.cuda.split('.')[:2]))" 2>/dev/null || echo "")
if [ -n "$TORCH_CUDA" ] && [ -d "/usr/local/cuda-$TORCH_CUDA" ]; then
    export CUDA_HOME="/usr/local/cuda-$TORCH_CUDA"
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
else
    echo "WARNING: CUDA_HOME not found; quanto will fall back to CPU inference"
fi
[ -n "${CUDA_HOME:-}" ] && echo "CUDA_HOME=$CUDA_HOME"
# QUANT_MODEL="$REPO_DIR/models/gemma-4-E2B-it-quanto-int4"
# QUANT_MODEL="$REPO_DIR/models/gemma-4-E2B-it"
QUANT_MODEL="$REPO_DIR/models/gemma-4-E2B-it-bnb-nf4"
RESULTS="$REPO_DIR/outputs/eval_gemma4_int4.json"

mkdir -p "$REPO_DIR/outputs"
cd "$REPO_DIR"

echo "======================================================"
echo "Job: $SLURM_JOB_ID   Host: $(hostname)   Date: $(date)"
echo "Model: $QUANT_MODEL"
echo "======================================================"

python scripts/eval_quantized.py \
    --model "$QUANT_MODEL" \
    --n 20 \
    --max-tokens 200 \
    --show-prompt \
    --output "$RESULTS"

echo ""
echo "======================================================"
echo "Done. Results at: $RESULTS"
echo "======================================================"
