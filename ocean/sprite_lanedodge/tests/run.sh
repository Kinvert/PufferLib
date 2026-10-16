#!/usr/bin/env bash
# Game, observations, viewer writes and assets only. No GPU or neural execution.
set -euo pipefail
cd "$(dirname "$0")/../../.."
lanedodge_out="${1:?Usage: bash ocean/sprite_lanedodge/tests/run.sh FRESH_OUTPUT_DIR}"
if [[ -e "$lanedodge_out" ]]; then
    echo "Output already exists: $lanedodge_out" >&2
    exit 1
fi
mkdir -p "$lanedodge_out"
lanedodge_raylib="${RAYLIB_ROOT:-raylib-5.5_linux_amd64}"
lanedodge_flags=(-std=c11 -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -O1 -g
    -fsanitize=address,undefined -fno-omit-frame-pointer
    -I. -Isrc -I"$lanedodge_raylib/include")
lanedodge_link=("$lanedodge_raylib/lib/libraylib.a" -lGL -lpthread -ldl -lrt -lm)
clang "${lanedodge_flags[@]}" ocean/sprite_lanedodge/tests/test_environment.c \
    "${lanedodge_link[@]}" -o "$lanedodge_out/test-environment"
lanedodge_status=0
"$lanedodge_out/test-environment" | tee "$lanedodge_out/environment.log" || lanedodge_status=$?
echo "$lanedodge_status" > "$lanedodge_out/exit-status.txt"
if [[ "$lanedodge_status" -ne 0 ]]; then
    exit "$lanedodge_status"
fi
sha256sum config/sprite_lanedodge.ini ocean/sprite_lanedodge/sprite_lanedodge.h \
    ocean/sprite_lanedodge/sprite_lanedodge.cu ocean/sprite_lanedodge/sprites.h \
    ocean/sprite_lanedodge/tests/test_environment.c \
    ocean/sprite_lanedodge/tests/run.sh \
    src/pufferenv.h src/pufferl.cu src/ocean.cu src/nature.cu \
    research/sprite_encoders.cu > "$lanedodge_out/source.sha256"
