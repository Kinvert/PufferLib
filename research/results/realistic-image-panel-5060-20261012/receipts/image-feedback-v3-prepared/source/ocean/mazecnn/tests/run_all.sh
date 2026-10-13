#!/usr/bin/env bash
# CPU game/raster checks, no neural model or GPU access.
set -euo pipefail
cd "$(dirname "$0")/../../.."
out=${1:?Choose a fresh output directory}
mkdir -p "$(dirname "$out")"
mkdir "$out"
raylib_dir=raylib-5.5_linux_amd64
test -f "$raylib_dir/lib/libraylib.a"
flags=(-std=c11 -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L -Wall -Wextra
       -Wno-unused-function -Wno-unused-parameter -O1 -g
       -fsanitize=address,undefined -fno-omit-frame-pointer
       -I. -Isrc -I"$raylib_dir/include")
for variant in maze mazecnn; do
    extra=()
    if [ "$variant" = mazecnn ]; then extra=(-DTEST_PIXELS); fi
    "${CC:-clang}" "${flags[@]}" "${extra[@]}" \
        "-DENV_HEADER=\"ocean/$variant/$variant.h\"" \
        ocean/mazecnn/tests/test_environment.c "$raylib_dir/lib/libraylib.a" \
        -lGL -lpthread -ldl -lrt -lm -o "$out/test_$variant"
    "$out/test_$variant" > "$out/$variant.trace"
done
cmp "$out/maze.trace" "$out/mazecnn.trace"
"$out/test_mazecnn" > "$out/repeat.trace"
cmp "$out/maze.trace" "$out/repeat.trace"
for representation in {1..5}; do
    "$out/test_mazecnn" "$representation" > "$out/r$representation.trace"
    cmp "$out/maze.trace" "$out/r$representation.trace"
done
for seed in 0 12345 4294967295; do
    "$out/test_mazecnn" 0 1 "$seed" > "$out/mixed-$seed.trace"
    cmp "$out/maze.trace" "$out/mixed-$seed.trace"
done
for option in 'num_maps 0' 'num_maps 8193' 'num_maps 1.5' 'num_maps nan' \
              'map_size 4' 'map_size 48' 'map_size 11.5' 'map_size nan' 'map_size 1e100' 'map_size inf' \
              'representation -1' 'representation 6' 'representation 1.5' \
              'representation_mode 2' 'representation_seed -1' 'representation_seed 4294967296'; do
    if "$out/test_mazecnn" --invalid $option > /dev/null 2> "$out/invalid-${option// /-}.txt"; then
        echo "Invalid configuration accepted: $option" >&2; exit 1
    fi
    rg -q 'MazeCNN|CNN appearance:' "$out/invalid-${option// /-}.txt"
done
for option in 'total_agents 0' 'total_agents nan' 'total_agents 1e100' \
              'total_agents 17' 'num_buffers 0' 'num_buffers 3' 'num_buffers 4.5' \
              'num_maps 1.5' 'map_size 1e100' 'map_size nan'; do
    if "$out/test_mazecnn" --invalid-vec $option > /dev/null 2> "$out/vector-invalid-${option// /-}.txt"; then
        echo "Invalid vector configuration accepted: $option" >&2; exit 1
    fi
    rg -q 'MazeCNN|CNN appearance:' "$out/vector-invalid-${option// /-}.txt"
done
sha256sum "$out/"*.trace > "$out/traces.sha256"
echo "PASS: 49,152 decisions per panel match independent game reference and original Maze; all six/three mixed appearances agree."
echo "PASS: BFS levels, bounds/local pixels, goal-at-cap/double-reset, RNG/worker-order/shared levels, repeat, ASan/UBSan."
