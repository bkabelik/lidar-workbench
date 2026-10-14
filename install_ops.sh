#!/usr/bin/env bash
set -e

# Auto-detect CUDA toolkit if not already set
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif [ -d "/usr/local/cuda-12.9" ]; then
        export CUDA_HOME="/usr/local/cuda-12.9"
    elif [ -d "/usr/local/cuda-12.8" ]; then
        export CUDA_HOME="/usr/local/cuda-12.8"
    fi
fi

if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# Multi-architecture support:
# 8.0: A100
# 8.6: RTX 30xx (Ampere)
# 8.9: RTX 40xx (Ada Lovelace)
# 9.0: Hopper H100
# 12.0+PTX: RTX 50xx (Blackwell) + future forward compatibility
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"

echo "=========================================================="
echo "Installing C++/CUDA Operators for LiDAR Workbench"
echo "=========================================================="
echo "CUDA_HOME            : ${CUDA_HOME:-not found}"
echo "NVCC                 : $(which nvcc 2>/dev/null || echo 'not in PATH')"
if command -v nvcc &>/dev/null; then
    nvcc --version | grep "release"
fi
echo "TORCH_CUDA_ARCH_LIST : $TORCH_CUDA_ARCH_LIST"
echo "=========================================================="

echo "[1/4] Installing flash-attn..."
pip install flash-attn --no-build-isolation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[2/4] Compiling and installing pointops..."
pip install -e "$SCRIPT_DIR/Pointcept/libs/pointops" --no-build-isolation

echo "[3/4] Compiling and installing pointgroup_ops..."
pip install -e "$SCRIPT_DIR/Pointcept/libs/pointgroup_ops" --no-build-isolation

echo "[4/4] Compiling and installing pointrope..."
pip install -e "$SCRIPT_DIR/Pointcept/libs/pointrope" --no-build-isolation

# Register lidar_workbench in the active Python environment
PYTHON_SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [ -n "$PYTHON_SITE_PACKAGES" ] && [ -d "$PYTHON_SITE_PACKAGES" ]; then
    ln -sfn "$SCRIPT_DIR" "$PYTHON_SITE_PACKAGES/lidar_workbench"
fi

echo ""
echo "=== Verifying CUDA Operators in PyTorch ==="
python -c '
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name    :", torch.cuda.get_device_name(0))
    print("Device arch    :", torch.cuda.get_device_capability(0))
import pointops
import pointgroup_ops
import pointrope
import flash_attn
print("SUCCESS: All Pointcept ops and flash-attn imported and ready!")
'

