#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
raylib_dir=raylib-5.5_linux_amd64
test -f "$raylib_dir/lib/libraylib.a"
out=${1:-build/pongcnn/environment-$(date +%Y%m%d-%H%M%S)}
test ! -e "$out"
mkdir -p "$out"
flags=(-std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra
       -Wno-unused-function -Wno-unused-parameter -O1 -g
       -fsanitize=address,undefined -fno-omit-frame-pointer
       -I. -Isrc -I"$raylib_dir/include")
for variant in pong pongcnn; do
    extra=()
    if [ "$variant" = pongcnn ]; then extra=(-DTEST_PIXELS); fi
    "${CC:-clang}" "${flags[@]}" "${extra[@]}" \
        "-DENV_HEADER=\"ocean/$variant/$variant.h\"" \
        ocean/pongcnn/tests/test_environment.c \
        "$raylib_dir/lib/libraylib.a" -lGL -lpthread -ldl -lrt -lm \
        -o "$out/test_$variant"
    "$out/test_$variant" > "$out/$variant.trace"
done
cmp "$out/pong.trace" "$out/pongcnn.trace"
"$out/test_pongcnn" > "$out/pongcnn.repeat.trace"
cmp "$out/pongcnn.trace" "$out/pongcnn.repeat.trace"
for representation in {1..6}; do
    "$out/test_pongcnn" "$representation" > "$out/representation-$representation.trace"
    cmp "$out/pong.trace" "$out/representation-$representation.trace"
done
for seed in 0 12345 4294967295; do
    for catalog in 0 1; do
        "$out/test_pongcnn" 0 1 "$seed" "$catalog" > "$out/mixed-$seed-$catalog.trace"
        cmp "$out/pong.trace" "$out/mixed-$seed-$catalog.trace"
    done
done
for option in '-1' '7' 'nan' '1.5' '0 2' '0 1 -1' '0 1 4294967296' '0 1 0 -1' '0 1 0 2' '0 1 0 nan' '0 1 0 0.5' '5 1 0 0'; do
    if "$out/test_pongcnn" $option >/dev/null 2>"$out/appearance-invalid.txt"; then
        echo "Invalid appearance options accepted: $option" >&2
        exit 1
    fi
    rg -q 'must be an integer' "$out/appearance-invalid.txt"
done
echo "PASS: 49,152 transitions match native Pong; repeat, pixel fixtures, rewards/reset, ASan/UBSan."
echo "PASS: seven fixed appearances, both mixed catalogs and three seeds preserve Pong physics/RNG; texture decoding/invalid settings pass."
