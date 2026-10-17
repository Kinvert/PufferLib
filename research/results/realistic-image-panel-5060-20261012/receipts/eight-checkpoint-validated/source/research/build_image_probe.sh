#!/usr/bin/env bash
# Native headless raster safety/physics fixture, never a CPU CNN.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# == 5 && ! -e "$5" ]] || { echo 'Usage: build_image_probe.sh GAME HEIGHT WIDTH CHANNELS FRESH_OUT' >&2; exit 1; }
image_game=$1
case "$image_game" in
    connect4cnn) image_reps=10 ;;
    pongcnn|flappycnn) image_reps=7 ;;
    breakoutcnn) image_reps=5 ;;
    snakecnn|mazecnn) image_reps=6 ;;
    *) exit 1 ;;
esac
for image_size in "$2" "$3"; do [[ "$image_size" =~ ^[0-9]+$ ]] && ((image_size >= 36 && image_size <= 1024)) || exit 1; done
[[ "$4" == 1 || "$4" == 3 ]] || exit 1
mkdir -p "$(dirname "$5")"
image_flags=(-DCNN_IMAGE_V1 -DCNN_IMAGE_HEIGHT="$2" -DCNN_IMAGE_WIDTH="$3" -DCNN_IMAGE_CHANNELS="$4")
[[ "${IMAGE_PROBE_LEGACY:-0}" == 0 ]] || image_flags=()
image_command=("${CC:-clang}" -std=c11 -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -O1 -g
    -fsanitize=address,undefined -fno-omit-frame-pointer -I. -Isrc -Iraylib-5.5_linux_amd64/include
    "-DENV_HEADER=\"ocean/$image_game/$image_game.h\"" "-DPUFFER_${image_game^^}"
    -DIMAGE_REPRESENTATIONS="$image_reps" "${image_flags[@]}"
    research/image_raster_probe.c raylib-5.5_linux_amd64/lib/libraylib.a
    -lGL -lpthread -ldl -lrt -lm -o "$5")
printf '%q ' "${image_command[@]}" > "$5.command.txt"
printf '\n' >> "$5.command.txt"
"${image_command[@]}" > "$5.build.txt" 2>&1
