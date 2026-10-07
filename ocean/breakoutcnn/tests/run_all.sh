#!/usr/bin/env bash
# CPU simulation/pixel safety only. No neural model executes.
set -euo pipefail
cd "$(dirname "$0")/../../.."
raylib_dir=raylib-5.5_linux_amd64
test -f "$raylib_dir/lib/libraylib.a"
mkdir -p build/breakoutcnn
flags=(-std=c11 -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L -Wall -Wextra
       -Wno-unused-function -Wno-unused-parameter -O1 -g
       -fsanitize=address,undefined -fno-omit-frame-pointer
       -I. -Isrc -I"$raylib_dir/include")
for variant in breakout breakoutcnn; do
    extra=()
    if [[ "$variant" == breakoutcnn ]]; then extra=(-DTEST_PIXELS); fi
    "${CC:-clang}" "${flags[@]}" "${extra[@]}" \
        "-DENV_HEADER=\"ocean/$variant/$variant.h\"" \
        ocean/breakoutcnn/tests/test_environment.c \
        "$raylib_dir/lib/libraylib.a" -lGL -lpthread -ldl -lrt -lm \
        -o "build/breakoutcnn/test_$variant"
    "build/breakoutcnn/test_$variant" > "build/breakoutcnn/$variant.trace"
done
cmp build/breakoutcnn/breakout.trace build/breakoutcnn/breakoutcnn.trace
build/breakoutcnn/test_breakoutcnn > build/breakoutcnn/repeat.trace
cmp build/breakoutcnn/breakoutcnn.trace build/breakoutcnn/repeat.trace
for representation in {1..4}; do
    build/breakoutcnn/test_breakoutcnn "$representation" > "build/breakoutcnn/representation-$representation.trace"
    cmp build/breakoutcnn/breakout.trace "build/breakoutcnn/representation-$representation.trace"
done
for seed in 0 12345 4294967295; do
    build/breakoutcnn/test_breakoutcnn 0 1 "$seed" > "build/breakoutcnn/mixed-$seed.trace"
    cmp build/breakoutcnn/breakout.trace "build/breakoutcnn/mixed-$seed.trace"
done
for option in '-1' '5' 'nan' '1.5' '0 2' '0 1 -1' '0 1 4294967296'; do
    if build/breakoutcnn/test_breakoutcnn $option >/dev/null 2>build/breakoutcnn/appearance-invalid.txt; then
        echo "Invalid appearance options accepted: $option" >&2; exit 1
    fi
    rg -q 'must be an integer' build/breakoutcnn/appearance-invalid.txt
done
echo 'PASS: 18,432 transitions per panel match original Breakout, all five fixed/three mixed appearances.'
echo 'PASS: launch/brick/life/terminal fixtures; literal raster, clipping, removal, hidden-state isolation, overwrite, repeat, ASan/UBSan.'
