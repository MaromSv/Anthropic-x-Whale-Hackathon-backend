#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=anthropic-bnb-quant
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm_bnb_quant_%A.out

module purge
module load 2024

source "$HOME/whale-venv/bin/activate"

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL="$REPO_DIR/models/gemma-4-E2B-it"
OUTPUT="$REPO_DIR/models/gemma-4-E2B-it-bnb-nf4"

mkdir -p "$REPO_DIR/outputs"
cd "$REPO_DIR"

# Point LD_LIBRARY_PATH at the nvjitlink that came with torch
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
if [ -d "$SITE_PKGS/nvidia/nvjitlink/lib" ]; then
    export LD_LIBRARY_PATH="$SITE_PKGS/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
fi

echo "======================================================"
echo "Job: $SLURM_JOB_ID   Host: $(hostname)   Date: $(date)"
echo "Model : $MODEL"
echo "Output: $OUTPUT"
echo "bitsandbytes: $(python -c 'import bitsandbytes; print(bitsandbytes.__version__)')"
echo "torch CUDA:   $(python -c 'import torch; print(torch.version.cuda)')"
echo "======================================================"

python -m scripts.quantize_bnb_4bit \
    --model  "$MODEL" \
    --output "$OUTPUT" \
    --max-new-tokens 64

echo ""
echo "======================================================"
echo "Done. Model at: $OUTPUT"
echo "======================================================"
