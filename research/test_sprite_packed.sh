#!/usr/bin/env bash
# Scalar raster/game checks, never CPU neural computation or GPU queries.
set -euo pipefail
cd "$(dirname "$0")/.."
sprite_out="${1:?Usage: test_sprite_packed.sh FRESH_OUT}"
test ! -e "$sprite_out"
mkdir -p "$sprite_out"
for sprite_game in sprite_catch sprite_platform sprite_maze sprite_runner sprite_key sprite_push; do
    for sprite_layout in tiles packed; do
        sprite_flags=()
        if [[ "$sprite_layout" == packed ]]; then
            sprite_flags=(-DSPRITE_PACKED_VIEW)
        fi
        clang -std=c11 -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -O1 -g \
            -fsanitize=address,undefined -fno-omit-frame-pointer \
            -I. -Isrc -Iraylib-5.5_linux_amd64/include "${sprite_flags[@]}" \
            "-DGAME_HEADER=\"ocean/$sprite_game/$sprite_game.h\"" \
            "-DGAME_CONFIG=\"config/$sprite_game.ini\"" \
            ocean/sprite_common/test_attention.c raylib-5.5_linux_amd64/lib/libraylib.a \
            -lGL -lpthread -ldl -lrt -lm -o "$sprite_out/$sprite_game-$sprite_layout"
        "$sprite_out/$sprite_game-$sprite_layout" \
            > "$sprite_out/$sprite_game-$sprite_layout.log" 2>&1
    done
done
sha256sum ocean/sprite_common/{sprite.h,attention.h,test_attention.c} \
    > "$sprite_out/source.sha256"
