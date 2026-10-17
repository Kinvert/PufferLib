#!/usr/bin/env bash
# Test-only independent CUDA reference, compiled against the captured actual encoder.
set -euo pipefail
cd "$(dirname "$0")/.."
forage_out=$(realpath "${1:?Usage: build_forage_attention_checks.sh ATTENTION_BUILD REFERENCE_SOURCE}")
forage_reference=$(realpath "${2:?Usage: build_forage_attention_checks.sh ATTENTION_BUILD REFERENCE_SOURCE}")
source ocean/connect4cnn/runtime_env.sh
test ! -e "$forage_out/checks"
source "$forage_out/attention-build.env"
forage_gaze="${FORAGE_GAZE_MODE:-0}"
forage_relative="${FORAGE_RELATIVE_GAZE:-0}"
forage_onehot="${FORAGE_STATE_ONEHOT:-0}"
forage_player_view="${FORAGE_PLAYER_VIEW:-1}"
cp research/test_forage_attention.cu "$forage_out/source/research/"
cp "$forage_reference" "$forage_out/source/research/native_encoder_reference.cu"
cd "$forage_out/source"
"$CUDA_HOME/bin/nvcc" -O2 -arch=sm_120 -std=c++17 \
    -I. -Isrc -Ivendor -I"$CUDA_HOME/include" -I"$CUDA_HOME/include/cccl" \
    -I"$NCCL_ROOT/include" -Iraylib-5.5_linux_amd64/include \
    '-DENV_HEADER="ocean/sprite_forage/sprite_forage.h"' -DENV_NAME=sprite_forage \
    '-DPUFFER_ENV_NAME="sprite_forage"' -DPUFFERLIB_BUILD_MAIN -DPRECISION_FLOAT \
    -DFORAGE_ATTENTION -DSPRITE_CNN_RESEARCH \
    -DFORAGE_GAZE_MODE="$forage_gaze" -DFORAGE_RELATIVE_GAZE="$forage_relative" \
    -DFORAGE_STATE_ONEHOT="$forage_onehot" -DFORAGE_PLAYER_VIEW="$forage_player_view" \
    -Xcompiler=-DPLATFORM_DESKTOP -Xcompiler=-fopenmp \
    research/test_forage_attention.cu raylib-5.5_linux_amd64/lib/libraylib.a \
    -L"$CUDA_HOME/lib64" -L"$CUDA_HOME/lib64/stubs" -L"$NCCL_ROOT/lib" \
    -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
    -lGL -lm -lpthread -ldl -lrt -o "$forage_out/checks" \
    > "$forage_out/checks-build.log" 2>&1
sha256sum "$forage_out/checks" "$forage_out/source/research/"*.cu \
    > "$forage_out/checks-source.sha256"
