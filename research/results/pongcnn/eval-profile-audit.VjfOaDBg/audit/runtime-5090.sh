#!/usr/bin/env bash
# Local, process-only setup; reuse the existing F-Zero NCCL package.
export CUDA_HOME=/usr/local/cuda-13.1
export NCCL_ROOT=/home/keith/Git/ml/fzero-training/PufferLib-5.0/.venv/lib/python3.12/site-packages/nvidia/nccl
source ocean/connect4cnn/runtime_env.sh
export NVCC_ARCH=sm_120
export OPENBLAS_NUM_THREADS=1
# The wheel has only its versioned library. Keep the linker alias in this checkout.
mkdir -p build/connect4cnn/link-libs
ln -sfn "$NCCL_ROOT/lib/libnccl.so.2" build/connect4cnn/link-libs/libnccl.so
export LIBRARY_PATH="$PWD/build/connect4cnn/link-libs:$LIBRARY_PATH"
