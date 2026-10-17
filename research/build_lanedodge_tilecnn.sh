#!/usr/bin/env bash
# Build only. CORE_SOURCE is the retained compatible native research source
# tree; the lane-dodge env header comes from this checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
ld_core=$(realpath "${1:?Usage: build_lanedodge_tilecnn.sh CORE_SOURCE FRESH_OUT}")
ld_out="${2:?Usage: build_lanedodge_tilecnn.sh CORE_SOURCE FRESH_OUT}"
test ! -e "$ld_out"
mkdir -p "$ld_out/source/ocean" "$ld_out/source/research"
ld_out=$(realpath "$ld_out")
cp -a "$ld_core/src" "$ld_out/source/"
cp "$ld_core/build.sh" "$ld_out/source/"
cp -a ocean/sprite_lanedodge "$ld_out/source/ocean/"
cp -a "$ld_core/ocean/connect4cnn" "$ld_out/source/ocean/"
cp research/lanedodge_tilecnn_encoder.cu "$ld_out/source/research/"
cp research/sprite_attention_eval.cu "$ld_out/source/research/sprite_exact_eval.cu"
cp research/lanedodge_tilecnn_dispatch.cu "$ld_out/source/research/sprite_encoders.cu"
cp research/lanedodge_tilecnn_core.patch "$ld_out/"
ln -s "$(realpath "$ld_core/vendor")" "$ld_out/source/vendor"
ln -s "$PWD/raylib-5.5_linux_amd64" "$ld_out/source/raylib-5.5_linux_amd64"
mkdir -p "$ld_out/runtime/config"
cp config/default.ini "$ld_out/runtime/config/"
cp config/sprite_lanedodge.ini "$ld_out/runtime/config/"
source ocean/connect4cnn/runtime_env.sh
cd "$ld_out/source"
patch -p1 < "$ld_out/lanedodge_tilecnn_core.patch" > "$ld_out/patch.log"
NVCC_EXTRA="-DLD_TILE_CNN -DSPRITE_CNN_RESEARCH -DSPRITE_EXACT_RESEARCH" NVCC_ARCH=sm_120 \
    bash build.sh sprite_lanedodge "$ld_out/puffer" --float > "$ld_out/build.log" 2>&1
sha256sum "$ld_out/puffer" > "$ld_out/binary.sha256"
rg --files --no-ignore src ocean research | sort | xargs sha256sum > "$ld_out/source.sha256"
