#!/usr/bin/env bash
# Native CPU environment/raster/receipt checks only, no neural model.
set -euo pipefail
cd "$(dirname "$0")/../../.."
out=${1:?Use a fresh output directory}
mkdir -p "$(dirname "$out")"
mkdir "$out"
raylib_dir=raylib-5.5_linux_amd64
flags=(-std=c11 -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L -Wall -Wextra
       -Wno-unused-function -Wno-unused-parameter -O1 -g
       -fsanitize=address,undefined -fno-omit-frame-pointer
       -I. -Isrc -I"$raylib_dir/include")
for variant in snakebench snakecnn; do
    "${CC:-clang}" "${flags[@]}" "-DENV_HEADER=\"ocean/$variant/$variant.h\"" \
        ocean/snakecnn/tests/test_exact_episode.c "$raylib_dir/lib/libraylib.a" \
        -lGL -lpthread -ldl -lrt -lm -o "$out/test_$variant"
    "$out/test_$variant" > "$out/$variant.trace"
done
cmp "$out/snakebench.trace" "$out/snakecnn.trace"
"$out/test_snakecnn" > "$out/repeat.trace"
cmp "$out/snakecnn.trace" "$out/repeat.trace"
for representation in {1..5}; do
    "$out/test_snakecnn" "$representation" > "$out/r$representation.trace"
    cmp "$out/snakebench.trace" "$out/r$representation.trace"
done
sha256sum "$out/"*.trace > "$out/traces.sha256"
echo 'PASS: identity/reset/spawn/RNG/actor-reset flag, death/horizon/reward/rounding accounting, independent 96 game trajectories.'
echo 'PASS: all six pixel drawings match state worlds; rejected corrupt counters/actions, repeat, ASan/UBSan.'
