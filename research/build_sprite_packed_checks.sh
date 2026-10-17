#!/usr/bin/env bash
# Compile independent CUDA reference against captured benchmark callbacks; no GPU query.
set -euo pipefail
cd "$(dirname "$0")/.."
sprite_out=$(realpath "${1:?Usage: build_sprite_packed_checks.sh BENCHMARK_BUILD}")
test ! -e "$sprite_out/packed-checks"
cp research/test_sprite_packed.cu research/native_encoder_reference.cu "$sprite_out/source/research/"
source ocean/connect4cnn/runtime_env.sh
cd "$sprite_out/source"
"$CUDA_HOME/bin/nvcc" -O2 -arch="${NVCC_ARCH:-sm_120}" -std=c++17 -fmad=false \
    -I. -Isrc -Ivendor -I"$CUDA_HOME/include/cccl" -I"$NCCL_ROOT/include" \
    -Iraylib-5.5_linux_amd64/include \
    '-DENV_HEADER="ocean/sprite_catch/sprite_catch.h"' -DENV_NAME=sprite_catch \
    '-DPUFFER_ENV_NAME="sprite_catch"' -DPUFFERLIB_BUILD_MAIN -DPRECISION_FLOAT \
    -DSPRITE_ATTENTION -DSPRITE_PACKED_VIEW -DSPRITE_CNN_RESEARCH -DSPRITE_EXACT_RESEARCH \
    -Xcompiler=-DPLATFORM_DESKTOP -Xcompiler=-fopenmp \
    research/test_sprite_packed.cu raylib-5.5_linux_amd64/lib/libraylib.a \
    -L"$CUDA_HOME/lib64" -L"$CUDA_HOME/lib64/stubs" -L"$NCCL_ROOT/lib" \
    -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
    -lGL -lm -lpthread -ldl -lrt -o "$sprite_out/packed-checks" \
    > "$sprite_out/packed-checks-build.log" 2>&1
sha256sum "$sprite_out/packed-checks" "$sprite_out/source/research/test_sprite_packed.cu" \
    "$sprite_out/source/research/native_encoder_reference.cu" > "$sprite_out/packed-checks.sha256"
