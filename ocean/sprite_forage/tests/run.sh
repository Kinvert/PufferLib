#!/usr/bin/env bash
# Game, observations, viewer writes and assets only. No GPU or neural execution.
set -euo pipefail
cd "$(dirname "$0")/../../.."
forage_out="${1:?Usage: bash ocean/sprite_forage/tests/run.sh FRESH_OUTPUT_DIR}"
if [[ -e "$forage_out" ]]; then
    echo "Output already exists: $forage_out" >&2
    exit 1
fi
mkdir -p "$forage_out"
forage_raylib="${RAYLIB_ROOT:-raylib-5.5_linux_amd64}"
forage_flags=(-std=c11 -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -O1 -g
    -fsanitize=address,undefined -fno-omit-frame-pointer
    -I. -Isrc -I"$forage_raylib/include")
forage_link=("$forage_raylib/lib/libraylib.a" -lGL -lpthread -ldl -lrt -lm)
for forage_test in environment modes render navigation rng; do
    clang "${forage_flags[@]}" "ocean/sprite_forage/tests/test_$forage_test.c" \
        "${forage_link[@]}" -o "$forage_out/test-$forage_test"
    "$forage_out/test-$forage_test" | tee "$forage_out/$forage_test.log"
done
clang "${forage_flags[@]}" '-DIMPORTED_HEADER="ocean/sprite_forage/imported_sprites.h"' \
    ocean/sprite_forage/tests/test_imported.c "${forage_link[@]}" -o "$forage_out/test-imported"
"$forage_out/test-imported" \
    ocean/sprite_forage/assets/kenney/tile_{0114,0115,0116,0113,0101,0122,0123,0124,0120,0110,0004,0112}.png \
    | tee "$forage_out/imported.log"
sha256sum config/sprite_forage.ini ocean/sprite_forage/sprite_forage.h \
    ocean/sprite_forage/sprite_forage.cu ocean/sprite_forage/sprites.h \
    ocean/sprite_forage/imported_sprites.h src/pufferenv.h src/pufferl.cu \
    src/puffercpu.c src/ocean.cu > "$forage_out/source.sha256"
