#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
raylib_dir=raylib-5.5_linux_amd64
test -f "$raylib_dir/lib/libraylib.a"
mkdir -p build/pongcnn
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
        -o "build/pongcnn/test_$variant"
    "build/pongcnn/test_$variant" > "build/pongcnn/$variant.trace"
done
cmp build/pongcnn/pong.trace build/pongcnn/pongcnn.trace
build/pongcnn/test_pongcnn > build/pongcnn/pongcnn.repeat.trace
cmp build/pongcnn/pongcnn.trace build/pongcnn/pongcnn.repeat.trace
echo "PASS: 49,152 transitions match native Pong; repeat, pixel fixtures, rewards/reset, ASan/UBSan."
