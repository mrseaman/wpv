#!/bin/bash
# Submit the full training pipeline to SLURM
# Usage: ./submit_pipeline.sh [--skip-preprocess]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Create logs directory
mkdir -p logs

SKIP_PREPROCESS=false
if [ "$1" == "--skip-preprocess" ]; then
    SKIP_PREPROCESS=true
fi

echo "=========================================="
echo "Wikipedia Pageview Recommender Pipeline"
echo "=========================================="
echo ""

if [ "$SKIP_PREPROCESS" = false ]; then
    # Submit preprocessing job
    echo "Submitting preprocessing job..."
    PREPROCESS_JOB=$(sbatch --parsable scripts/preprocess_data.sbatch)
    echo "  Preprocessing job ID: $PREPROCESS_JOB"

    # Submit training job with dependency
    echo "Submitting training job (depends on preprocessing)..."
    TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$PREPROCESS_JOB scripts/train_model.sbatch)
    echo "  Training job ID: $TRAIN_JOB"
else
    # Submit training job without dependency
    echo "Submitting training job..."
    TRAIN_JOB=$(sbatch --parsable scripts/train_model.sbatch)
    echo "  Training job ID: $TRAIN_JOB"
fi

# Submit embedding extraction job with dependency
echo "Submitting embedding extraction job (depends on training)..."
EMBED_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB scripts/extract_embeddings.sbatch)
echo "  Embedding job ID: $EMBED_JOB"

echo ""
echo "=========================================="
echo "Pipeline submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f logs/train_${TRAIN_JOB}.out"
echo "=========================================="
