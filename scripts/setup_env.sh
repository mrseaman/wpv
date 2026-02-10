#!/bin/bash
# Setup Python environment on cluster with PyTorch CUDA support
# Run this once on a login node before submitting jobs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=========================================="
echo "Setting up Python environment"
echo "Project directory: $PROJECT_DIR"
echo "=========================================="

# Detect CUDA version on the system (if nvidia-smi is available)
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "Detected NVIDIA driver: $CUDA_VERSION"
fi

# Default PyTorch CUDA version
PYTORCH_CUDA=${PYTORCH_CUDA:-cu118}
echo "Using PyTorch with CUDA: $PYTORCH_CUDA"
echo ""

# Option 1: Create venv
setup_venv() {
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate

    pip install --upgrade pip

    echo "Installing PyTorch with CUDA support..."
    pip install torch --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}

    echo "Installing other dependencies..."
    pip install -r requirements.txt

    echo ""
    echo "Virtual environment created at: $PROJECT_DIR/venv"
    echo "Activate with: source venv/bin/activate"
}

# Option 2: Create conda environment
setup_conda() {
    echo "Creating conda environment..."

    conda create -n wpv python=3.10 -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate wpv

    echo "Installing PyTorch with CUDA support..."
    pip install torch --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}

    echo "Installing other dependencies..."
    pip install -r requirements.txt

    echo ""
    echo "Conda environment 'wpv' created"
    echo "Activate with: conda activate wpv"
}

# Choose method
echo "Select environment type:"
echo "  1) venv (recommended for NFS)"
echo "  2) conda"
read -p "Choice [1]: " choice
choice=${choice:-1}

case $choice in
    1) setup_venv ;;
    2) setup_conda ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
