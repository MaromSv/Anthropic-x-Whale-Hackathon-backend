#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=anthropic-rag-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:30:00
#SBATCH --output=outputs/slurm_rag_eval_%A.out

module purge
module load 2024

source "$HOME/whale-venv/bin/activate"

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

TORCH_CUDA=$(python -c "import torch; print('.'.join(torch.version.cuda.split('.')[:2]))" 2>/dev/null || echo "")
if [ -n "$TORCH_CUDA" ] && [ -d "/usr/local/cuda-$TORCH_CUDA" ]; then
    export CUDA_HOME="/usr/local/cuda-$TORCH_CUDA"
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
else
    echo "WARNING: CUDA_HOME not found; quanto will fall back to CPU inference"
fi
[ -n "${CUDA_HOME:-}" ] && echo "CUDA_HOME=$CUDA_HOME"

# MODEL="$REPO_DIR/models/gemma-4-E2B-it"
MODEL="$REPO_DIR/models/gemma-4-E2B-it-bnb-nf4"
RESULTS="$REPO_DIR/outputs/eval_rag_gemma4_nf4.json"

mkdir -p "$REPO_DIR/outputs"
cd "$REPO_DIR"

echo "======================================================"
echo "Job: $SLURM_JOB_ID   Host: $(hostname)   Date: $(date)"
echo "Model: $MODEL"
echo "======================================================"

python scripts/eval_rag.py \
    --model "$MODEL" \
    --n 20 \
    --k 5 \
    --max-tokens 200 \
    --output "$RESULTS"

echo ""
echo "======================================================"
echo "Done. Results at: $RESULTS"
echo "======================================================"
