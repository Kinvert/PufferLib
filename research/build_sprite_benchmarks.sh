#!/usr/bin/env bash
# Capture the committed core and current sprite inputs. Build only; no GPU query.
set -euo pipefail
cd "$(dirname "$0")/.."
sprite_out="${1:?Usage: build_sprite_benchmarks.sh FRESH_OUT [GAMES]}"
sprite_games="${2:-sprite_catch sprite_platform sprite_maze sprite_runner sprite_key sprite_push}"
test ! -e "$sprite_out"
mkdir -p "$sprite_out/source/research" "$sprite_out/runtime/config"
sprite_out=$(realpath "$sprite_out")
git rev-parse HEAD > "$sprite_out/core-revision.txt"
git archive HEAD src build.sh ocean/connect4cnn | tar -x -C "$sprite_out/source"
cp src/protein_feedback.cu "$sprite_out/source/src/"
cp ocean/connect4cnn/{impala,nature,flex}.cu "$sprite_out/source/ocean/connect4cnn/"
cp -a ocean/sprite_common "$sprite_out/source/ocean/"
cp config/default.ini "$sprite_out/runtime/config/"
for sprite_game in $sprite_games; do
    cp -a "ocean/$sprite_game" "$sprite_out/source/ocean/"
    cp "config/$sprite_game.ini" "$sprite_out/runtime/config/"
done
cp research/forage_attention_encoder.cu research/sprite_attention_encoder.cu \
    research/sprite_benchmark_metadata.cu "$sprite_out/source/research/"
cp research/sprite_packed_dispatch.cu "$sprite_out/source/research/sprite_encoders.cu"
cp research/sprite_suite_eval.cu "$sprite_out/source/research/sprite_exact_eval.cu"
cp research/sprite_attention_core.patch research/sprite_benchmark_core.patch "$sprite_out/"
ln -s "$PWD/vendor" "$sprite_out/source/vendor"
ln -s "$PWD/raylib-5.5_linux_amd64" "$sprite_out/source/raylib-5.5_linux_amd64"
source ocean/connect4cnn/runtime_env.sh
cd "$sprite_out/source"
patch -p1 < "$sprite_out/sprite_attention_core.patch" > "$sprite_out/patch.log"
patch -p1 < "$sprite_out/sprite_benchmark_core.patch" >> "$sprite_out/patch.log"
for sprite_game in $sprite_games; do
    NVCC_EXTRA='-DSPRITE_ATTENTION -DSPRITE_PACKED_VIEW -DSPRITE_CNN_RESEARCH -DSPRITE_EXACT_RESEARCH -DPUFFER_RESEARCH_PROTEIN_FEEDBACK' \
        NVCC_ARCH="${NVCC_ARCH:-sm_120}" bash build.sh "$sprite_game" "$sprite_out/$sprite_game" --float \
        > "$sprite_out/$sprite_game.build.log" 2>&1
    sha256sum "$sprite_out/$sprite_game" > "$sprite_out/$sprite_game.sha256"
done
rg --files --no-ignore src ocean research | sort | xargs sha256sum > "$sprite_out/source.sha256"
