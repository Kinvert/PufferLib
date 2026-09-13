#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

# Same shared Raylib directory used by build.sh and other Ocean environment tests.
case "$(uname -s)" in
    Linux) raylib_dir=raylib-5.5_linux_amd64; platform_libs=(-lGL -lpthread -ldl -lrt) ;;
    Darwin) raylib_dir=raylib-5.5_macos; platform_libs=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL) ;;
    *) echo "Unsupported platform" >&2; exit 1 ;;
esac
if [ ! -f "$raylib_dir/lib/libraylib.a" ]; then
    echo "Shared Raylib is missing. Use PufferLib's normal CPU build first:" >&2
    echo "bash build.sh connect4cnn --cpu" >&2
    exit 1
fi
mkdir -p build/connect4cnn
flags=(-std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra
       -Wno-unused-function -Wno-unused-parameter -O1 -g
       -fsanitize=address,undefined -fno-omit-frame-pointer
       -I. -Isrc -I"$raylib_dir/include")
for variant in connect4 connect4cnn; do
    extra=()
    if [ "$variant" = connect4cnn ]; then extra=(-DTEST_PIXELS); fi
    "${CC:-clang}" "${flags[@]}" "${extra[@]}" \
        "-DENV_HEADER=\"ocean/$variant/$variant.h\"" \
        ocean/connect4cnn/tests/test_environment.c \
        "$raylib_dir/lib/libraylib.a" "${platform_libs[@]}" -lm \
        -o "build/connect4cnn/test_$variant"
    "build/connect4cnn/test_$variant" > "build/connect4cnn/$variant.trace"
done
cmp build/connect4cnn/connect4.trace build/connect4cnn/connect4cnn.trace
build/connect4cnn/test_connect4cnn > build/connect4cnn/connect4cnn.repeat.trace
cmp build/connect4cnn/connect4cnn.trace build/connect4cnn/connect4cnn.repeat.trace
echo "PASS: pixel fixtures, reset/terminal checks, 4096-step upstream parity, repeated trace (ASan/UBSan)."
