#!/bin/bash
#SBATCH --job-name=wpv-preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
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
module load python/3.10

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -n "$CONDA_PREFIX" ] || command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate wpv
fi

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"
echo ""

# Run preprocessing
echo "Starting preprocessing..."
python scripts/preprocess_data.py

echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo "End time: $(date)"
echo "=========================================="
