#!/usr/bin/env bash
# CPU environment/observation safety checks only; no neural model executes.
set -euo pipefail
cd "$(dirname "$0")/../../.."
raylib_dir=raylib-5.5_linux_amd64
test -f "$raylib_dir/lib/libraylib.a"
mkdir -p build/flappycnn
flags=(-std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra
       -Wno-unused-function -Wno-unused-parameter -O1 -g
       -fsanitize=address,undefined -fno-omit-frame-pointer
       -I. -Isrc -I"$raylib_dir/include")
for variant in flappy flappycnn; do
    extra=()
    if [[ "$variant" == flappycnn ]]; then extra=(-DTEST_PIXELS); fi
    "${CC:-clang}" "${flags[@]}" "${extra[@]}" \
        "-DENV_HEADER=\"ocean/$variant/$variant.h\"" \
        ocean/flappycnn/tests/test_environment.c \
        "$raylib_dir/lib/libraylib.a" -lGL -lpthread -ldl -lrt -lm \
        -o "build/flappycnn/test_$variant"
    "build/flappycnn/test_$variant" > "build/flappycnn/$variant.trace"
done
cmp build/flappycnn/flappy.trace build/flappycnn/flappycnn.trace
build/flappycnn/test_flappycnn > build/flappycnn/flappycnn.repeat.trace
cmp build/flappycnn/flappycnn.trace build/flappycnn/flappycnn.repeat.trace
for representation in {1..3}; do
    build/flappycnn/test_flappycnn "$representation" > "build/flappycnn/representation-$representation.trace"
    cmp build/flappycnn/flappy.trace "build/flappycnn/representation-$representation.trace"
done
for seed in 0 12345 4294967295; do
    build/flappycnn/test_flappycnn 0 1 "$seed" > "build/flappycnn/mixed-$seed.trace"
    cmp build/flappycnn/flappy.trace "build/flappycnn/mixed-$seed.trace"
done
for option in '-1' '4' 'nan' '1.5' '0 2' '0 1 -1' '0 1 4294967296'; do
    if build/flappycnn/test_flappycnn $option >/dev/null 2>build/flappycnn/appearance-invalid.txt; then
        echo "Invalid appearance options accepted: $option" >&2
        exit 1
    fi
    rg -q 'must be an integer' build/flappycnn/appearance-invalid.txt
done
echo 'PASS: 49,152 transitions plus pass/crash/cap/respawn fixtures match native Flappy.'
echo 'PASS: fixed/mixed appearances, clipping, overwrite, velocity isolation, repeatability, ASan/UBSan.'
