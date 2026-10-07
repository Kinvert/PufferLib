#!/usr/bin/env bash
# Compilation only; explicit architecture avoids querying a GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ $# != 1 || -e "$1" ]]; then
    echo 'Usage: bash ocean/connect4cnn/build_encoder_profile.sh FRESH_OUTPUT_DIR' >&2
    exit 1
fi
mkdir -p "$1"
profile_out=$(realpath "$1")
source ocean/connect4cnn/runtime_env.sh
profile_cuda="${CUDA_HOME:-/usr/local/cuda}"
profile_arch="${NVCC_ARCH:-sm_120}"
if [[ "$profile_arch" == native ]]; then
    echo 'Set an explicit NVCC_ARCH; this build must not query a GPU.' >&2
    exit 1
fi
case "${C4_DENSE_PATCH_ALIAS:-0}" in
    0|1) ;;
    *) echo 'C4_DENSE_PATCH_ALIAS must be 0 or 1.' >&2; exit 1 ;;
esac
git rev-parse HEAD > "$profile_out/revision.txt"
git status --short > "$profile_out/worktree.txt"
"$profile_cuda/bin/nvcc" --version > "$profile_out/compiler.txt"
rg --files src ocean/connect4cnn | sort | xargs sha256sum > "$profile_out/source.sha256"
for profile_family in default impala impoola; do
    profile_def=()
    if [[ "$profile_family" == impala ]]; then profile_def=(-DC4_IMPALA_CNN); fi
    if [[ "$profile_family" == impoola ]]; then profile_def=(-DC4_IMPOOLA_CNN); fi
    if [[ "${C4_DENSE_PATCH_ALIAS:-0}" == 1 ]]; then profile_def+=(-DC4_DENSE_PATCH_ALIAS); fi
    profile_command=("$profile_cuda/bin/nvcc" -O2 -std=c++17 -arch="$profile_arch"
        -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing
        --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128
        -I. -Isrc -Ivendor -Iraylib-5.5_linux_amd64/include
        -I"$profile_cuda/include/cccl" -I"$NCCL_ROOT/include"
        "${profile_def[@]}" ocean/connect4cnn/encoder_profile.cu
        raylib-5.5_linux_amd64/lib/libraylib.a
        -L"$profile_cuda/lib64" -L"$NCCL_ROOT/lib" -L/usr/lib/wsl/lib
        -lcudart -lcublas -lcurand -lcusolver -lnccl -l:libnvidia-ml.so.1 -lGL -lpthread
        -o "$profile_out/$profile_family")
    printf '%q ' "${profile_command[@]}" > "$profile_out/$profile_family.command"
    printf '\n' >> "$profile_out/$profile_family.command"
    "${profile_command[@]}" > "$profile_out/$profile_family.build.log" 2>&1
done
sha256sum -c "$profile_out/source.sha256" > "$profile_out/source-check.log"
sha256sum "$profile_out/default" "$profile_out/impala" "$profile_out/impoola" > "$profile_out/binaries.sha256"
echo "Compiled three profiler targets; no policy/GPU execution: $profile_out"
