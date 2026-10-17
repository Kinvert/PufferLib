#!/usr/bin/env bash
# Three standalone host descriptor targets. Compilation is not GPU execution.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 1 || -e "$1" ]]; then
    echo 'Usage: bash research/build_policy_metadata.sh FRESH_OUTPUT_DIR' >&2
    exit 1
fi
metadata_arch="${NVCC_ARCH:-sm_120}"
if [[ ! "$metadata_arch" =~ ^sm_[0-9]+[a-z]?$ ]]; then
    echo 'Set an explicit sm_N architecture; no native GPU discovery.' >&2
    exit 1
fi
if [[ -n "${NVCC_APPEND_FLAGS:-}" || ( -n "${NVCC_PREPEND_FLAGS:-}" && "${NVCC_PREPEND_FLAGS}" != '--threads 1' ) ]]; then
    echo 'Unrecorded compiler overrides are prohibited; only --threads 1 is allowed.' >&2
    exit 1
fi
source ocean/connect4cnn/runtime_env.sh
metadata_cuda="${CUDA_HOME:-/usr/local/cuda}"
mkdir -p "$1"
metadata_out=$(realpath "$1")
mkdir -p "$metadata_out/source"
git rev-parse HEAD > "$metadata_out/revision.txt"
git status --short > "$metadata_out/worktree.txt"
"$metadata_cuda/bin/nvcc" --version > "$metadata_out/compiler.txt"
printf '%s\n' "NVCC_ARCH=$metadata_arch" "NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}" \
    "NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-}" "CUDA_HOME=$metadata_cuda" "NCCL_ROOT=$NCCL_ROOT" > "$metadata_out/environment.txt"
{ rg --files src ocean/connect4cnn; printf '%s\n' research/policy_metadata.cu research/build_policy_metadata.sh; } | sort -u > "$metadata_out/source-files.txt"
while IFS= read -r metadata_source; do
    sha256sum "$metadata_source"
done < "$metadata_out/source-files.txt" > "$metadata_out/source.sha256"
while IFS= read -r metadata_source; do
    cp --parents "$metadata_source" "$metadata_out/source/"
done < "$metadata_out/source-files.txt"
(cd "$metadata_out/source" && sha256sum -c "$metadata_out/source.sha256") > "$metadata_out/copied-source-check.txt"
for metadata_family in default impala impoola; do
    metadata_def=()
    if [[ "$metadata_family" == impala ]]; then metadata_def=(-DC4_IMPALA_CNN); fi
    if [[ "$metadata_family" == impoola ]]; then metadata_def=(-DC4_IMPOOLA_CNN); fi
    metadata_command=("$metadata_cuda/bin/nvcc" -O1 -std=c++17 -arch="$metadata_arch"
        -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing
        --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128
        -I. -Isrc -Ivendor -Iraylib-5.5_linux_amd64/include
        -I"$metadata_cuda/include/cccl" -I"$NCCL_ROOT/include" "${metadata_def[@]}"
        research/policy_metadata.cu raylib-5.5_linux_amd64/lib/libraylib.a
        -L"$metadata_cuda/lib64" -L"$NCCL_ROOT/lib" -L/usr/lib/wsl/lib
        -lcudart -lcublas -lcurand -lcusolver -lnccl -l:libnvidia-ml.so.1 -lGL -lpthread
        -o "$metadata_out/$metadata_family")
    printf '%q ' "${metadata_command[@]}" > "$metadata_out/$metadata_family.command.txt"
    printf '\n' >> "$metadata_out/$metadata_family.command.txt"
    "${metadata_command[@]}" > "$metadata_out/$metadata_family.compiler.txt" 2>&1
done
sha256sum -c "$metadata_out/source.sha256" > "$metadata_out/source-check.txt"
sha256sum "$metadata_out/default" "$metadata_out/impala" "$metadata_out/impoola" > "$metadata_out/binaries.sha256"
echo "Compiled three policy metadata targets; no GPU/policy execution: $metadata_out"
