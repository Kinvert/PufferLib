#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source research/image_test_flags.sh
[[ $# == 1 && ! -e "$1" && ! -e "$1.build" ]] || { echo 'Supply a fresh output path.' >&2; exit 1; }
[[ "${NVCC_ARCH:-}" =~ ^sm_[0-9]+[a-z]?$ ]] || { echo 'Explicit NVCC_ARCH required.' >&2; exit 1; }
[[ -z "${NVCC_APPEND_FLAGS:-}" && -z "${NVCC_PREPEND_FLAGS:-}" ]] || exit 1
ref_output="$1"
ref_receipt="$ref_output.build"
mkdir -p "$(dirname "$ref_output")" "$ref_receipt"
sha256sum research/native_flex_reference.cu research/native_encoder_reference.cu research/build_native_flex_reference.sh research/image_test_flags.sh > "$ref_receipt/source.sha256"
git rev-parse HEAD > "$ref_receipt/revision.txt"
git status --short > "$ref_receipt/worktree.txt"
"${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version > "$ref_receipt/compiler.txt"
printf '%s\n' "NVCC_ARCH=$NVCC_ARCH" "CNN_TEST_IMAGE=${CNN_TEST_IMAGE:-legacy}" > "$ref_receipt/environment.txt"
ref_command=("${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" -shared -O1 -std=c++17 "-arch=$NVCC_ARCH" --threads 1 -fmad=false
    -Xcompiler=-fPIC "${image_ref_defs[@]}" research/native_flex_reference.cu -o "$ref_output")
printf '%q ' "${ref_command[@]}" > "$ref_receipt/command.txt"
printf '\n' >> "$ref_receipt/command.txt"
"${ref_command[@]}" > "$ref_receipt/build.txt" 2>&1
sha256sum -c "$ref_receipt/source.sha256" > "$ref_receipt/source-check.txt"
sha256sum "$ref_output" > "$ref_receipt/binary.sha256"
