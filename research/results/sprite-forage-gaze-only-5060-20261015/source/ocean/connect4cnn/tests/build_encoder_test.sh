#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
source research/image_test_flags.sh
source ocean/connect4cnn/runtime_env.sh
sprite_test_defs=()
sprite_test_sources=()
if [[ -n "${SPRITE_ENCODER_TEST:-}" ]]; then
    case "$SPRITE_ENCODER_TEST" in
        sprite_catch|sprite_platform|sprite_maze|sprite_runner|sprite_key|sprite_push|sprite_forage) ;;
        *) echo 'Unknown sprite test environment.' >&2; exit 1 ;;
    esac
    [[ "${CNN_TEST_IMAGE:-}" == 480,640,3 ]] || { echo 'Sprite tests require480,640,3.' >&2; exit 1; }
    sprite_test_defs=(-DSPRITE_CNN_RESEARCH "-DENV_HEADER=\"ocean/$SPRITE_ENCODER_TEST/$SPRITE_ENCODER_TEST.h\""
        "-DPUFFER_ENV_NAME=\"$SPRITE_ENCODER_TEST\"")
    sprite_test_sources=(research/sprite_encoders.cu "ocean/$SPRITE_ENCODER_TEST/$SPRITE_ENCODER_TEST.h"
        "ocean/$SPRITE_ENCODER_TEST/sprites.h" ocean/sprite_common/sprite.h)
    if [[ -f "ocean/$SPRITE_ENCODER_TEST/imported_sprites.h" ]]; then
        sprite_test_sources+=("ocean/$SPRITE_ENCODER_TEST/imported_sprites.h")
    fi
fi
cuda_dir="${CUDA_HOME:-/usr/local/cuda}"
nccl_dir="${NCCL_ROOT:-}"
if [ -z "$nccl_dir" ]; then
    nccl_dir=$(.venv/bin/python -c 'import nvidia.nccl; print(nvidia.nccl.__path__[0])')
fi
mkdir -p build/connect4cnn
test_name="${1:-test_encoder}"
case "$test_name" in test_encoder|test_nature|test_impala|test_cnn|test_flex2) ;; *) echo "Unknown encoder test: $test_name" >&2; exit 1 ;; esac
if [[ -n "${CNN_TEST_IMAGE:-}" && "$test_name" != test_nature && "$test_name" != test_impala ]]; then
    echo 'Portable image harness supports Nature/Flex/IMPALA/Impoola only.' >&2; exit 1
fi
alias_def=()
case "${C4_DENSE_PATCH_ALIAS:-0}" in
    0) ;;
    1)
        if [[ "$test_name" != test_nature ]]; then echo 'Dense alias candidate requires test_nature.' >&2; exit 1; fi
        if [[ "${NVCC_ARCH:-native}" == native ]]; then echo 'Dense alias candidate requires an explicit NVCC_ARCH.' >&2; exit 1; fi
        if [[ $# != 2 || -e "$2" ]]; then echo 'Dense alias candidate requires a fresh explicit output path.' >&2; exit 1; fi
        alias_def=(-DC4_DENSE_PATCH_ALIAS) ;;
    *) echo 'C4_DENSE_PATCH_ALIAS must be 0 or 1.' >&2; exit 1 ;;
esac
test_output="${2:-build/connect4cnn/$test_name.so}"
if [[ $# -gt 2 ]]; then echo 'Usage: build_encoder_test.sh [TEST_NAME [FRESH_OUTPUT]]' >&2; exit 1; fi
test_receipt="${test_output}.build"
if [[ $# == 2 ]]; then
    if [[ -e "$test_output" || -e "$test_receipt" ]]; then echo 'Explicit test outputs require fresh paths.' >&2; exit 1; fi
    mkdir -p "$test_receipt"
    { rg --files src ocean/connect4cnn; printf '%s\n' research/image_test_flags.sh "${sprite_test_sources[@]}"; } | sed '/^$/d' | sort -u | xargs sha256sum > "$test_receipt/source.sha256"
    git rev-parse HEAD > "$test_receipt/revision.txt"
    git status --short > "$test_receipt/worktree.txt"
    "$cuda_dir/bin/nvcc" --version > "$test_receipt/compiler.txt"
    printf '%s\n' "C4_DENSE_PATCH_ALIAS=${C4_DENSE_PATCH_ALIAS:-0}" \
        "NVCC_ARCH=${NVCC_ARCH:-native}" "NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}" \
        "NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-}" "CNN_TEST_IMAGE=${CNN_TEST_IMAGE:-legacy}" \
        "SPRITE_ENCODER_TEST=${SPRITE_ENCODER_TEST:-}" > "$test_receipt/environment.txt"
fi
mkdir -p "$(dirname "$test_output")"
test_command=("$cuda_dir/bin/nvcc" -shared -O1 -std=c++17 -arch="${NVCC_ARCH:-native}"
    -Xcompiler=-fPIC -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing
    --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128
    -I. -Isrc -Ivendor -Iraylib-5.5_linux_amd64/include
    -I"$cuda_dir/include/cccl" -I"$nccl_dir/include"
    "${alias_def[@]}" "${image_test_defs[@]}" "${sprite_test_defs[@]}" "ocean/connect4cnn/tests/$test_name.cu"
    raylib-5.5_linux_amd64/lib/libraylib.a
    -L"$cuda_dir/lib64" -L"$nccl_dir/lib" -L/usr/lib/wsl/lib
    -lcudart -lcublas -lcurand -lcusolver -lnccl -l:libnvidia-ml.so.1 -lGL -lpthread
    -o "$test_output")
if [[ $# == 2 ]]; then
    printf '%q ' "${test_command[@]}" > "$test_receipt/command.txt"
    printf '\n' >> "$test_receipt/command.txt"
    "${test_command[@]}" > "$test_receipt/build.txt" 2>&1
    sha256sum -c "$test_receipt/source.sha256" > "$test_receipt/source-check.txt"
    sha256sum "$test_output" > "$test_receipt/binary.sha256"
else
    "${test_command[@]}"
fi
