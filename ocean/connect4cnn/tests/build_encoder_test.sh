#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
source ocean/connect4cnn/runtime_env.sh
cuda_dir="${CUDA_HOME:-/usr/local/cuda}"
nccl_dir="${NCCL_ROOT:-}"
if [ -z "$nccl_dir" ]; then
    nccl_dir=$(.venv/bin/python -c 'import nvidia.nccl; print(nvidia.nccl.__path__[0])')
fi
mkdir -p build/connect4cnn
test_name="${1:-test_encoder}"
case "$test_name" in test_encoder|test_nature) ;; *) echo "Expected test_encoder or test_nature" >&2; exit 1 ;; esac
"$cuda_dir/bin/nvcc" -shared -O1 -std=c++17 -arch="${NVCC_ARCH:-native}" \
    -Xcompiler=-fPIC -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing \
    --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128 \
    -I. -Isrc -Ivendor -Iraylib-5.5_linux_amd64/include \
    -I"$cuda_dir/include/cccl" -I"$nccl_dir/include" \
    "ocean/connect4cnn/tests/$test_name.cu" \
    raylib-5.5_linux_amd64/lib/libraylib.a \
    -L"$cuda_dir/lib64" -L"$nccl_dir/lib" -L/usr/lib/wsl/lib \
    -lcudart -lcublas -lcurand -lcusolver -lnccl -l:libnvidia-ml.so.1 -lGL -lpthread \
    -o "build/connect4cnn/$test_name.so"
