#!/usr/bin/env bash
# CPU world/raster checks only; no neural policy execution.
set -euo pipefail
cd "$(dirname "$0")/../../.."
raylib_dir=raylib-5.5_linux_amd64
out=${1:-build/pongcnn/exact-episode-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$(dirname "$out")"
mkdir "$out"
flags=(-std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra
       -Wno-unused-function -Wno-unused-parameter -O1 -g
       -fsanitize=address,undefined -fno-omit-frame-pointer
       -I. -Isrc -I"$raylib_dir/include")
for variant in pong pongcnn; do
    extra=(); if [[ "$variant" == pongcnn ]]; then extra=(-DTEST_PIXELS); fi
    "${CC:-clang}" "${flags[@]}" "${extra[@]}" \
        "-DENV_HEADER=\"ocean/$variant/$variant.h\"" ocean/pongcnn/tests/test_exact_episode.c \
        "$raylib_dir/lib/libraylib.a" -lGL -lpthread -ldl -lrt -lm -o "$out/test_$variant"
    "$out/test_$variant" > "$out/$variant.trace"
done
cmp "$out/pong.trace" "$out/pongcnn.trace"
"$out/test_pongcnn" > "$out/repeat.trace"
cmp "$out/pongcnn.trace" "$out/repeat.trace"
for rep in {1..6}; do
    "$out/test_pongcnn" "$rep" > "$out/r$rep.trace"
    cmp "$out/pong.trace" "$out/r$rep.trace"
done
sha256sum "$out"/*.trace > "$out/traces.sha256"
echo 'PASS: episode/RNG/dirty-state starts, both scoring sides and match terminals, whole-match versus rally counters, zero/nonzero-point caps.'
echo 'PASS: first-to-1..21 final-fraction bounds independently enumerated; 96 stock-reference trajectories, all drawings, repeat, ASan/UBSan.'
echo "Evidence: $out"
