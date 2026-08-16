#!/bin/bash
# Build EasySteer's forked vLLM (vllm-steer) from source into .venv_sae.
# EasySteer needs python 3.12 and its own torch (torch 2.10.0+cu130, with Blackwell
# kernels). We compile only the local GPU arch for speed (default sm_100 / B200;
# override TORCH_CUDA_ARCH_LIST for other GPUs).
#
# Expects the EasySteer source (https://github.com/ZJU-REAL/EasySteer) checked out at
# $EASYSTEER_SRC (default /tmp/EasySteer), with its vllm-steer submodule:
#   git clone --recursive https://github.com/ZJU-REAL/EasySteer /tmp/EasySteer
#
# Usage: EASYSTEER_SRC=/tmp/EasySteer bash baselines/sae/build_easysteer.sh
set -e

VENV="${SAE_VENV:-.venv_sae}"
EASYSTEER_SRC="${EASYSTEER_SRC:-/tmp/EasySteer}"
ARCH="${TORCH_CUDA_ARCH_LIST:-10.0}"          # 10.0 = sm_100 (B200); 9.0 = H100; 8.0 = A100

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

echo "[build] creating python 3.12 venv at $VENV"
uv venv --python 3.12 "$VENV"

echo "[build] installing torch 2.10.0+cu130 (Blackwell kernels)"
uv pip install --python "$VENV/bin/python" "torch==2.10.0+cu130" \
    --index-url https://download.pytorch.org/whl/cu130

cd "$EASYSTEER_SRC/vllm-steer"
echo "[build] use_existing_torch.py (strip torch pins -> use the cu130 torch above)"
"$VENV/bin/python" use_existing_torch.py

export TORCH_CUDA_ARCH_LIST="$ARCH"
export CMAKE_ARGS="-DTORCH_CUDA_ARCH_LIST=$ARCH"
export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS=$(nproc)
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export VLLM_FA_CMAKE_GPU_ARCHES="$(echo "$ARCH" | tr -d .)-real"

echo "[build] installing build deps (requirements/build.txt, torch stripped)"
uv pip install --python "$VENV/bin/python" -r requirements/build.txt
echo "[build] compiling vllm-steer for arch $ARCH (this is the long part)..."
uv pip install --python "$VENV/bin/python" -e . --no-build-isolation -v

cd "$EASYSTEER_SRC"
echo "[build] installing easysteer package"
uv pip install --python "$VENV/bin/python" -e .

echo "EASYSTEER_BUILD_DONE rc=$?"
"$VENV/bin/python" -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__)"
