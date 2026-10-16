#!/usr/bin/env bash
# Build only. CORE_SOURCE is the retained compatible native research source tree.
set -euo pipefail
cd "$(dirname "$0")/.."
forage_core=$(realpath "${1:?Usage: build_forage_attention.sh CORE_SOURCE FRESH_OUT}")
forage_out="${2:?Usage: build_forage_attention.sh CORE_SOURCE FRESH_OUT}"
forage_gaze="${3:-0}"
forage_relative="${4:-0}"
forage_exact="${5:-0}"
forage_onehot="${6:-0}"
[[ "$forage_gaze" =~ ^[012]$ && "$forage_relative" =~ ^[01]$ ]]
[[ "$forage_exact" =~ ^[01]$ ]]
[[ "$forage_onehot" =~ ^[01]$ ]]
test ! -e "$forage_out"
mkdir -p "$forage_out/source/ocean" "$forage_out/source/research"
forage_out=$(realpath "$forage_out")
cp -a "$forage_core/src" "$forage_out/source/"
cp "$forage_core/build.sh" "$forage_out/source/"
cp -a ocean/sprite_forage "$forage_out/source/ocean/"
cp -a "$forage_core/ocean/connect4cnn" "$forage_out/source/ocean/"
cp research/forage_attention_encoder.cu "$forage_out/source/research/"
cp research/forage_attention_dispatch.cu "$forage_out/source/research/sprite_encoders.cu"
cp research/forage_attention_core.patch "$forage_out/"
printf 'FORAGE_GAZE_MODE=%s\nFORAGE_RELATIVE_GAZE=%s\n' \
    "$forage_gaze" "$forage_relative" > "$forage_out/attention-build.env"
forage_extra=""
if [[ "$forage_exact" == 1 ]]; then
    cp research/sprite_attention_eval.cu "$forage_out/source/research/sprite_exact_eval.cu"
    forage_extra="-DSPRITE_EXACT_RESEARCH"
fi
printf 'SPRITE_EXACT_RESEARCH=%s\n' "$forage_exact" >> "$forage_out/attention-build.env"
printf 'FORAGE_STATE_ONEHOT=%s\n' "$forage_onehot" >> "$forage_out/attention-build.env"
ln -s "$(realpath "$forage_core/vendor")" "$forage_out/source/vendor"
ln -s "$PWD/raylib-5.5_linux_amd64" "$forage_out/source/raylib-5.5_linux_amd64"
mkdir -p "$forage_out/runtime/config"
cp config/default.ini "$forage_out/runtime/config/"
cat config/sprite_forage.ini research/recipes/sprite_forage_spatial_cnn.ini \
    research/recipes/sprite_forage_attention.ini > "$forage_out/runtime/config/sprite_forage.ini"
source ocean/connect4cnn/runtime_env.sh
cd "$forage_out/source"
patch -p1 < "$forage_out/forage_attention_core.patch" > "$forage_out/patch.log"
NVCC_EXTRA="-DFORAGE_ATTENTION -DSPRITE_CNN_RESEARCH -DFORAGE_GAZE_MODE=$forage_gaze -DFORAGE_RELATIVE_GAZE=$forage_relative -DFORAGE_STATE_ONEHOT=$forage_onehot $forage_extra" NVCC_ARCH=sm_120 \
    bash build.sh sprite_forage "$forage_out/puffer" --float > "$forage_out/build.log" 2>&1
sha256sum "$forage_out/puffer" > "$forage_out/binary.sha256"
rg --files --no-ignore src ocean research | sort | xargs sha256sum > "$forage_out/source.sha256"
