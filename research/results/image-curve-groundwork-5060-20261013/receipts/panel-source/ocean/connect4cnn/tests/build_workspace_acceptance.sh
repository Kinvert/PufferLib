#!/usr/bin/env bash
# Build only. Explicit architecture prevents native GPU discovery.
set -euo pipefail
cd "$(dirname "$0")/../../.."
if [[ $# != 1 || -e "$1" ]]; then
    echo 'Usage: build_workspace_acceptance.sh FRESH_BUILD_DIR' >&2
    exit 1
fi
workspace_arch="${NVCC_ARCH:-sm_120}"
if [[ "$workspace_arch" == native ]]; then
    echo 'Use an explicit NVCC_ARCH; GPU discovery is not permitted.' >&2
    exit 1
fi
mkdir -p "$1"
workspace_out=$(realpath "$1")
source ocean/connect4cnn/runtime_env.sh
workspace_cuda="${CUDA_HOME:-/usr/local/cuda}"
git rev-parse HEAD > "$workspace_out/revision.txt"
git status --short > "$workspace_out/worktree.txt"
"$workspace_cuda/bin/nvcc" --version > "$workspace_out/compiler.txt"
sha256sum raylib-5.5_linux_amd64/include/raylib.h raylib-5.5_linux_amd64/lib/libraylib.a \
    "$workspace_cuda/include/cublas_v2.h" "$workspace_cuda/include/cuda_runtime.h" \
    "$workspace_cuda/lib64/libcublas.so" "$workspace_cuda/lib64/libcudart.so" \
    "$NCCL_ROOT/include/nccl.h" "$NCCL_ROOT/lib/libnccl.so.2" > "$workspace_out/dependencies.sha256"
rg --files src ocean/connect4cnn | sort | xargs sha256sum > "$workspace_out/source.sha256"
workspace_command=("$workspace_cuda/bin/nvcc" -O2 -std=c++17 -arch="$workspace_arch"
    -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing
    --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128
    -I. -Isrc -Ivendor -Iraylib-5.5_linux_amd64/include
    -I"$workspace_cuda/include/cccl" -I"$NCCL_ROOT/include"
    ocean/connect4cnn/tests/workspace_acceptance.cu
    raylib-5.5_linux_amd64/lib/libraylib.a
    -L"$workspace_cuda/lib64" -L"$NCCL_ROOT/lib" -L/usr/lib/wsl/lib
    -lcudart -lcublas -lcurand -lcusolver -lnccl -l:libnvidia-ml.so.1 -lGL -lpthread
    -o "$workspace_out/workspace_acceptance")
printf '%q ' "${workspace_command[@]}" > "$workspace_out/command.txt"
printf '\n' >> "$workspace_out/command.txt"
"${workspace_command[@]}" > "$workspace_out/build.log" 2>&1
sha256sum -c "$workspace_out/source.sha256" > "$workspace_out/source-check.log"
sha256sum -c "$workspace_out/dependencies.sha256" > "$workspace_out/dependencies-check.log"
sha256sum "$workspace_out/workspace_acceptance" > "$workspace_out/binary.sha256"
echo "Compiled isolated GEMM acceptance; no GPU/model execution: $workspace_out"
