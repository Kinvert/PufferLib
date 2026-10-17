#!/usr/bin/env bash
# Compile an explicit image contract into both the native trainer and metadata.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# == 6 && ! -e "$6" ]] || { echo 'Usage: build_image_policy.sh GAME HEIGHT WIDTH CHANNELS FAMILY FRESH_DIR' >&2; exit 1; }
image_game=$1
case "$image_game" in connect4cnn|pongcnn|flappycnn|breakoutcnn|snakecnn|mazecnn) ;; *) exit 1 ;; esac
for image_size in "$2" "$3"; do [[ "$image_size" =~ ^[0-9]+$ ]] && ((image_size >= 36 && image_size <= 1024)) || exit 1; done
[[ "$4" == 1 || "$4" == 3 ]] || exit 1
[[ "${NVCC_ARCH:-}" =~ ^sm_[0-9]+[a-z]?$ ]] || { echo 'Explicit NVCC_ARCH required.' >&2; exit 1; }
[[ -z "${NVCC_EXTRA:-}" && -z "${NVCC_APPEND_FLAGS:-}" && -z "${NVCC_PREPEND_FLAGS:-}" ]] || { echo 'Unrecorded compiler overrides refused.' >&2; exit 1; }
image_defs=(-DCNN_IMAGE_V1 -DCNN_IMAGE_HEIGHT="$2" -DCNN_IMAGE_WIDTH="$3" -DCNN_IMAGE_CHANNELS="$4")
case "$5" in default) ;; impala) image_defs+=(-DC4_IMPALA_CNN) ;; impoola) image_defs+=(-DC4_IMPOOLA_CNN) ;; *) exit 1 ;; esac
source ocean/connect4cnn/runtime_env.sh
mkdir -p "$6"
image_out=$(realpath "$6")
image_mode=${CNN_IMAGE_BUILD_MODE:-full}
[[ "$image_mode" == full || "$image_mode" == metadata ]] || { echo 'Invalid CNN_IMAGE_BUILD_MODE' >&2; exit 1; }
git rev-parse HEAD > "$image_out/revision.txt"
git status --short > "$image_out/worktree.txt"
{ rg --files src "ocean/$image_game" ocean/connect4cnn -g '*.h' -g '*.cu' -g '*.c' -g '!**/tests/**'; printf '%s\n' build.sh research/policy_metadata.cu research/build_image_policy.sh ocean/connect4cnn/runtime_env.sh; } | sort -u > "$image_out/source-files.txt"
while IFS= read -r image_source; do sha256sum "$image_source"; done < "$image_out/source-files.txt" > "$image_out/source.sha256"
while IFS= read -r image_source; do mkdir -p "$image_out/source/$(dirname "$image_source")"; cp "$image_source" "$image_out/source/$image_source"; done < "$image_out/source-files.txt"
(cd "$image_out/source" && sha256sum -c ../source.sha256) > "$image_out/copied-source-check.txt"
printf '%s\n' "NVCC_ARCH=$NVCC_ARCH" "CUDA_HOME=$CUDA_HOME" "NCCL_ROOT=$NCCL_ROOT" > "$image_out/environment.txt"
printf '%s\n' "${image_defs[*]}" > "$image_out/image-flags.txt"
"$CUDA_HOME/bin/nvcc" --version > "$image_out/compiler.txt"
if [[ "$image_mode" == full ]]; then
    NVCC_EXTRA="${image_defs[*]}" bash build.sh "$image_game" "$image_out/puffer" --float > "$image_out/train-build.txt" 2>&1
    "$image_out/puffer" eval_exact_info > "$image_out/trainer-info.json"
fi
image_command=("$CUDA_HOME/bin/nvcc" -O1 -std=c++17 -arch="$NVCC_ARCH" --threads 1
    -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128
    -I. -Isrc -Ivendor -Iraylib-5.5_linux_amd64/include -I"$CUDA_HOME/include/cccl" -I"$NCCL_ROOT/include"
    "${image_defs[@]}" "-DPUFFER_${image_game^^}" "-DENV_HEADER=\"ocean/$image_game/$image_game.h\""
    "-DPUFFER_ENV_NAME=\"$image_game\"" research/policy_metadata.cu raylib-5.5_linux_amd64/lib/libraylib.a
    -L"$CUDA_HOME/lib64" -L"$NCCL_ROOT/lib" -L/usr/lib/wsl/lib
    -lcudart -lcublas -lcurand -lcusolver -lnccl -l:libnvidia-ml.so.1 -lGL -lpthread -o "$image_out/metadata")
printf '%q ' "${image_command[@]}" > "$image_out/metadata-command.txt"
printf '\n' >> "$image_out/metadata-command.txt"
"${image_command[@]}" > "$image_out/metadata-build.txt" 2>&1
sha256sum -c "$image_out/source.sha256" > "$image_out/source-check.txt"
"$image_out/metadata" --image-info > "$image_out/metadata-info.json"
if [[ "$image_mode" == full ]]; then sha256sum "$image_out/puffer" "$image_out/metadata" > "$image_out/binaries.sha256"
else sha256sum "$image_out/metadata" > "$image_out/binaries.sha256"; fi
echo "Built $image_mode image targets without GPU execution: $image_out"
