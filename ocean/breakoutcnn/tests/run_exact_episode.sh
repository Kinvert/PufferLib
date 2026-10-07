#!/usr/bin/env bash
# Host world/raster checks; no GPU or policy execution.
set -euo pipefail
cd "$(dirname "$0")/../../.."
raylib_dir=raylib-5.5_linux_amd64
out=${1:-build/breakoutcnn/exact-episode-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$(dirname "$out")"
mkdir "$out"
flags=(-std=c11 -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L -Wall -Wextra
       -Wno-unused-function -Wno-unused-parameter -O1 -g
       -fsanitize=address,undefined -fno-omit-frame-pointer
       -I. -Isrc -I"$raylib_dir/include")
for variant in breakout breakoutcnn; do
    extra=()
    if [[ "$variant" == breakoutcnn ]]; then extra=(-DTEST_PIXELS); fi
    "${CC:-clang}" "${flags[@]}" "${extra[@]}" \
        "-DENV_HEADER=\"ocean/$variant/$variant.h\"" \
        ocean/breakoutcnn/tests/test_exact_episode.c \
        "$raylib_dir/lib/libraylib.a" -lGL -lpthread -ldl -lrt -lm \
        -o "$out/test_$variant"
    "$out/test_$variant" > "$out/$variant.trace"
done
cmp "$out/breakout.trace" "$out/breakoutcnn.trace"
"$out/test_breakoutcnn" > "$out/repeat.trace"
cmp "$out/breakoutcnn.trace" "$out/repeat.trace"
for representation in {1..4}; do
    "$out/test_breakoutcnn" "$representation" > "$out/r$representation.trace"
    cmp "$out/breakout.trace" "$out/r$representation.trace"
done
sha256sum "$out"/*.trace > "$out/traces.sha256"
echo 'PASS: independent episode starts/launch RNG, terminal frame-skip capture, cap without fake terminal, all five rasters.'
echo 'PASS: 96 action trajectories against original puf_step; partial final action, winning reward, final-loss reset, repeat, ASan/UBSan.'
echo "Evidence: $out"
