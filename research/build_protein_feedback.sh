#!/usr/bin/env bash
# Temporary native core research mode, never a production or PR build.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 1 || -e "$1" ]]; then
    echo 'Usage: bash research/build_protein_feedback.sh FRESH_OUTPUT_DIR' >&2
    exit 1
fi
source ocean/connect4cnn/runtime_env.sh
feedback_arch="${NVCC_ARCH:-sm_120}"
[[ "$feedback_arch" =~ ^sm_[0-9]+[a-z]?$ ]] || exit 1
[[ -z "${NVCC_APPEND_FLAGS:-}" && -z "${NVCC_EXTRA:-}" && -z "${NVCC_FLAGS:-}" ]] || exit 1
[[ -z "${NVCC_PREPEND_FLAGS:-}" || "$NVCC_PREPEND_FLAGS" == '--threads 1' ]] || exit 1
mkdir -p "$1"
feedback_out=$(realpath "$1")
mkdir "$feedback_out/source"
rg --files src ocean/connect4cnn | sort -u > "$feedback_out/source-files.txt"
printf '%s\n' build.sh research/build_protein_feedback.sh >> "$feedback_out/source-files.txt"
while IFS= read -r feedback_source; do
    sha256sum "$feedback_source"
    cp --parents "$feedback_source" "$feedback_out/source/"
done < "$feedback_out/source-files.txt" > "$feedback_out/source.sha256"
git rev-parse HEAD > "$feedback_out/revision.txt"
git status --short > "$feedback_out/worktree.txt"
"$CUDA_HOME/bin/nvcc" --version > "$feedback_out/compiler.txt"
printf '%s\n' "CUDA_HOME=$CUDA_HOME" "NCCL_ROOT=$NCCL_ROOT" "NVCC_ARCH=$feedback_arch" \
    'NVCC_PREPEND_FLAGS=--threads 1 -DPUFFER_RESEARCH_PROTEIN_FEEDBACK' > "$feedback_out/environment.txt"
printf '%q ' bash build.sh connect4cnn "$feedback_out/worker" --float > "$feedback_out/command.txt"
printf '\n' >> "$feedback_out/command.txt"
NVCC_ARCH="$feedback_arch" NVCC_PREPEND_FLAGS='--threads 1 -DPUFFER_RESEARCH_PROTEIN_FEEDBACK' \
    bash build.sh connect4cnn "$feedback_out/worker" --float > "$feedback_out/build.txt" 2>&1
sha256sum -c "$feedback_out/source.sha256" > "$feedback_out/source-check.txt"
sha256sum "$feedback_out/worker" > "$feedback_out/binary.sha256"
echo "Built temporary native PROTEIN feedback worker: $feedback_out"
