#!/usr/bin/env bash
# Scalar environment tests, never CPU neural training.
set -euo pipefail
cd "$(dirname "$0")/../.."
sprite_out="${1:?Supply a fresh output directory}"
[[ ! -e "$sprite_out" ]]
mkdir -p "$sprite_out"
sha256sum src/pufferenv.h src/ini.h config/default.ini \
    ocean/sprite_common/sprite.h ocean/sprite_common/test_games.c \
    ocean/sprite_common/test_render.c ocean/sprite_common/test.sh \
    ocean/sprite_common/attention.h ocean/sprite_common/test_attention.c \
    raylib-5.5_linux_amd64/lib/libraylib.a \
    > "$sprite_out/source.sha256"
for sprite_game in ${SPRITE_TEST_GAMES:-sprite_catch sprite_platform sprite_maze sprite_runner sprite_key sprite_push}; do
    sha256sum "ocean/$sprite_game/$sprite_game.h" "ocean/$sprite_game/sprites.h" \
        "config/$sprite_game.ini" >> "$sprite_out/source.sha256"
    sprite_kind="${sprite_game#sprite_}"
    for sprite_test in games render attention; do
        clang -std=c11 -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -O1 -g \
            -fsanitize=address,undefined -fno-omit-frame-pointer \
            -I. -Isrc -Iraylib-5.5_linux_amd64/include \
            "-DGAME_HEADER=\"ocean/$sprite_game/$sprite_game.h\"" \
            "-DGAME_CONFIG=\"config/$sprite_game.ini\"" "-DSUITE_${sprite_kind^^}" \
            "ocean/sprite_common/test_$sprite_test.c" \
            raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lpthread -ldl -lrt -lm \
            -o "$sprite_out/$sprite_game-$sprite_test"
        "$sprite_out/$sprite_game-$sprite_test" > "$sprite_out/$sprite_game-$sprite_test.log" 2>&1
    done
done
