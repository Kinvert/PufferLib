#!/usr/bin/env bash
# Build copied main-checkout sprite sources against an explicit captured core.
set -euo pipefail
cd "$(dirname "$0")/.."
sprite_core=$(realpath "${1:?Usage: build_sprite_attention.sh CORE_SOURCE FRESH_OUT [GAMES]}")
sprite_out="${2:?Usage: build_sprite_attention.sh CORE_SOURCE FRESH_OUT [GAMES]}"
sprite_games="${3:-sprite_catch sprite_platform sprite_maze sprite_runner sprite_key sprite_push}"
test ! -e "$sprite_out"
mkdir -p "$sprite_out/source/ocean" "$sprite_out/source/research" "$sprite_out/runtime/config"
sprite_out=$(realpath "$sprite_out")
cp -a "$sprite_core/src" "$sprite_out/source/"
cp "$sprite_core/build.sh" "$sprite_out/source/"
cp -a "$sprite_core/ocean/connect4cnn" "$sprite_out/source/ocean/"
cp -a ocean/sprite_common "$sprite_out/source/ocean/"
cp config/default.ini "$sprite_out/runtime/config/"
for sprite_game in $sprite_games; do
    cp -a "ocean/$sprite_game" "$sprite_out/source/ocean/"
    cp "config/$sprite_game.ini" "$sprite_out/runtime/config/"
done
cp research/forage_attention_encoder.cu research/sprite_attention_encoder.cu "$sprite_out/source/research/"
cp research/sprite_attention_dispatch.cu "$sprite_out/source/research/sprite_encoders.cu"
cp research/sprite_suite_eval.cu "$sprite_out/source/research/sprite_exact_eval.cu"
cp research/sprite_attention_core.patch "$sprite_out/"
ln -s "$PWD/vendor" "$sprite_out/source/vendor"
ln -s "$PWD/raylib-5.5_linux_amd64" "$sprite_out/source/raylib-5.5_linux_amd64"
source ocean/connect4cnn/runtime_env.sh
cd "$sprite_out/source"
patch -p1 < "$sprite_out/sprite_attention_core.patch" > "$sprite_out/patch.log"
for sprite_game in $sprite_games; do
    NVCC_EXTRA='-DSPRITE_ATTENTION -DSPRITE_CNN_RESEARCH -DSPRITE_EXACT_RESEARCH' \
        NVCC_ARCH=sm_120 bash build.sh "$sprite_game" "$sprite_out/$sprite_game" --float \
        > "$sprite_out/$sprite_game.build.log" 2>&1
    sha256sum "$sprite_out/$sprite_game" > "$sprite_out/$sprite_game.sha256"
done
rg --files --no-ignore src ocean research | sort | xargs sha256sum > "$sprite_out/source.sha256"
