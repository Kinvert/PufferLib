#!/usr/bin/env bash
# CPU environment/raster/identity counters only. No policy or GPU.
set -euo pipefail
cd "$(dirname "$0")/../../.."
out=${1:?Choose a fresh output directory}
mkdir -p "$(dirname "$out")"; mkdir "$out"
raylib_dir=raylib-5.5_linux_amd64
flags=(-std=c11 -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L -Wall -Wextra
       -Wno-unused-function -Wno-unused-parameter -O1 -g
       -fsanitize=address,undefined -fno-omit-frame-pointer
       -I. -Isrc -I"$raylib_dir/include")
for variant in maze mazecnn; do
    extra=()
    if [ "$variant" = mazecnn ]; then extra=(-DTEST_PIXELS -DPUFFER_MAZECNN); fi
    "${CC:-clang}" "${flags[@]}" "${extra[@]}" \
        "-DENV_HEADER=\"ocean/$variant/$variant.h\"" \
        ocean/mazecnn/tests/test_exact_episode.c "$raylib_dir/lib/libraylib.a" \
        -lGL -lpthread -ldl -lrt -lm -o "$out/test_$variant"
    "$out/test_$variant" > "$out/$variant.trace"
done
cmp "$out/maze.trace" "$out/mazecnn.trace"
"$out/test_mazecnn" > "$out/repeat.trace"; cmp "$out/maze.trace" "$out/repeat.trace"
for representation in {1..5}; do
    "$out/test_mazecnn" "$representation" > "$out/r$representation.trace"
    cmp "$out/maze.trace" "$out/r$representation.trace"
done
sha256sum "$out/"*.trace > "$out/traces.sha256"
echo 'PASS: 96 independent native game trajectories; success/timeout and duplicate native log accounted once.'
echo 'PASS: identity/level/dirty/reset/prefix/cyclic allocation, all six drawings, invalid actions, repeat, ASan/UBSan.'
