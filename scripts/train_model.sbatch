#!/bin/bash
#SBATCH --job-name=wpv-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@example.com

# Exit on error
set -e

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust based on your cluster)
module purge
module load python/3.10 2>/dev/null || true

# Activate virtual environment
# Option 1: venv (PyTorch includes CUDA runtime)
if [ -d "venv" ]; then
    source venv/bin/activate
# Option 2: conda
elif command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate wpv
fi

# Verify PyTorch can see CUDA
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

# Verify GPU is available
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"
echo ""

# Check if preprocessed data exists
if [ ! -f "preprocessed_data.pkl" ]; then
    echo "Preprocessed data not found. Running preprocessing first..."
    python scripts/preprocess_data.py
    echo ""
fi

# Run training
echo "Starting training..."
echo "=========================================="

python scripts/train_model.py \
    --epochs 100 \
    --batch-size 64 \
    --device cuda

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=========================================="
