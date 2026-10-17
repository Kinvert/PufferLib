#!/usr/bin/env bash
# Captured before/after native targets. Build only; no GPU query or training.
set -euo pipefail
cd "$(dirname "$0")/.."
sprite_out="$PWD/build/sprite-cleanup-20261016"
source ocean/connect4cnn/runtime_env.sh
for sprite_version in before after; do
    sprite_source="$sprite_out/$sprite_version-source"
    test ! -e "$sprite_source"
    mkdir -p "$sprite_source/ocean/connect4cnn" "$sprite_source/research"
    cp -a build/sprite-forage-cleanup-sps-20261015/after-source/src "$sprite_source/"
    cp build/sprite-forage-cleanup-sps-20261015/after-source/build.sh "$sprite_source/"
    if [[ "$sprite_version" == before ]]; then
        cp -a "$sprite_out/before/ocean/sprite_common" "$sprite_source/ocean/"
    else
        cp -a ocean/sprite_common "$sprite_source/ocean/"
    fi
    for sprite_game in sprite_catch sprite_platform sprite_maze sprite_runner sprite_key sprite_push; do
        if [[ "$sprite_version" == before ]]; then
            cp -a "$sprite_out/before/ocean/$sprite_game" "$sprite_source/ocean/"
        else
            cp -a "ocean/$sprite_game" "$sprite_source/ocean/"
        fi
    done
    cp research/sprite_suite_eval.cu "$sprite_source/research/sprite_exact_eval.cu"
    cp build/sprite-forage-onehot-relative-build-20261015/source/ocean/connect4cnn/exact_protocol.h \
        "$sprite_source/ocean/connect4cnn/"
    ln -s "$PWD/vendor" "$sprite_source/vendor"
    ln -s "$PWD/raylib-5.5_linux_amd64" "$sprite_source/raylib-5.5_linux_amd64"
    for sprite_game in sprite_catch sprite_platform sprite_maze sprite_runner sprite_key sprite_push; do
        (cd "$sprite_source"
            NVCC_EXTRA=-DSPRITE_EXACT_RESEARCH NVCC_ARCH=sm_120 bash build.sh "$sprite_game" \
                "$sprite_out/$sprite_version-$sprite_game" --float \
                > "$sprite_out/$sprite_version-$sprite_game.build.log" 2>&1)
    done
    (cd "$sprite_source"
        rg --files --no-ignore src ocean research | sort | xargs sha256sum) \
        > "$sprite_out/$sprite_version-source.sha256"
done
sha256sum "$sprite_out"/{before,after}-sprite_{catch,platform,maze,runner,key,push} \
    > "$sprite_out/binaries.sha256"
