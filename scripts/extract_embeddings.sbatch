#!/bin/bash
#SBATCH --job-name=wpv-embed
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Exit on error
set -e

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Load modules
module purge
module load python/3.10 2>/dev/null || true

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate wpv
fi

# Verify PyTorch can see CUDA
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

# Verify GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

export PYTHONUNBUFFERED=1

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"
echo ""

# Run embedding extraction
echo "Extracting embeddings..."
python scripts/extract_embeddings.py --device cuda

echo ""
echo "=========================================="
echo "Embedding extraction complete!"
echo "End time: $(date)"
echo "=========================================="
