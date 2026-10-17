#!/usr/bin/env bash
# Independent float64 CUDA oracle against the captured 18-tile encoder.
set -euo pipefail
cd "$(dirname "$0")/.."
sprite_out=$(realpath "${1:?Usage: build_sprite_attention_checks.sh ATTENTION_BUILD [ENV]}")
sprite_game="${2:-sprite_catch}"
case "$sprite_game" in
    sprite_catch)
        sprite_test=test_sprite_attention.cu
        sprite_flags=(-DSPRITE_ATTENTION -DSPRITE_CNN_RESEARCH) ;;
    sprite_lanedodge)
        sprite_test=test_lanedodge_tilecnn.cu
        sprite_flags=(-DLD_TILE_CNN -DSPRITE_CNN_RESEARCH) ;;
    *) echo 'Use sprite_catch or sprite_lanedodge for the shared encoder checks.' >&2; exit 1 ;;
esac
test ! -e "$sprite_out/checks"
source ocean/connect4cnn/runtime_env.sh
cp "research/$sprite_test" research/native_encoder_reference.cu "$sprite_out/source/research/"
cd "$sprite_out/source"
"$CUDA_HOME/bin/nvcc" -O2 -arch=sm_120 -std=c++17 \
    -I. -Isrc -Ivendor -I"$CUDA_HOME/include" -I"$CUDA_HOME/include/cccl" \
    -I"$NCCL_ROOT/include" -Iraylib-5.5_linux_amd64/include \
    "-DENV_HEADER=\"ocean/$sprite_game/$sprite_game.h\"" "-DENV_NAME=$sprite_game" \
    "-DPUFFER_ENV_NAME=\"$sprite_game\"" -DPUFFERLIB_BUILD_MAIN -DPRECISION_FLOAT \
    "${sprite_flags[@]}" \
    -Xcompiler=-DPLATFORM_DESKTOP -Xcompiler=-fopenmp \
    "research/$sprite_test" raylib-5.5_linux_amd64/lib/libraylib.a \
    -L"$CUDA_HOME/lib64" -L"$CUDA_HOME/lib64/stubs" -L"$NCCL_ROOT/lib" \
    -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
    -lGL -lm -lpthread -ldl -lrt -o "$sprite_out/checks" \
    > "$sprite_out/checks-build.log" 2>&1
sha256sum "$sprite_out/checks" "$sprite_out/source/research/"*.cu \
    > "$sprite_out/checks-source.sha256"
