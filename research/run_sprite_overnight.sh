#!/usr/bin/env bash
# Build/prepare without GPU work; run validation then the bounded native search.
set -euo pipefail
cd "$(dirname "$0")/.."
sprite_command="${1:-}"
sprite_dir="${2:-}"
if [[ $# != 2 || ! "$sprite_command" =~ ^(prepare|prepare-smoke|run|run-smoke|report|export)$ ]]; then
    echo 'Usage: bash research/run_sprite_overnight.sh prepare FRESH_DIR' >&2
    echo '       bash research/run_sprite_overnight.sh run PREPARED_DIR' >&2
    echo '       prepare-smoke / run-smoke use the bounded local5060 route.' >&2
    echo '       report PREPARED_DIR rebuilds stopped-run exports offline.' >&2
    echo '       export PREPARED_DIR creates compact evidence, keeping all raw weights.' >&2
    exit 2
fi
if [[ -n "${SPRITE_RUNTIME_ENV:-}" ]]; then
    source "$SPRITE_RUNTIME_ENV"
elif [[ -f build/connect4cnn/runtime-5090.sh ]]; then
    source build/connect4cnn/runtime-5090.sh
fi
source ocean/connect4cnn/runtime_env.sh
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use an existing uv Python3.12 venv"'
export NVCC_ARCH=sm_120
export NVCC_PREPEND_FLAGS='--threads 1'
if [[ "$sprite_command" == prepare* ]]; then
    [[ ! -e "$sprite_dir" ]] || { echo 'Use a fresh directory; no evidence overwrite.' >&2; exit 1; }
    mkdir -p "$sprite_dir"
    sprite_dir=$(realpath "$sprite_dir")
    git rev-parse HEAD > "$sprite_dir/revision.txt"
    git status --short > "$sprite_dir/worktree.txt"
    "$CUDA_HOME/bin/nvcc" --version > "$sprite_dir/compiler.txt"
    printf 'CUDA_HOME=%s\nNCCL_ROOT=%s\nNVCC_ARCH=%s\n' \
        "$CUDA_HOME" "$NCCL_ROOT" "$NVCC_ARCH" > "$sprite_dir/toolchain.txt"
    .venv/bin/python -m unittest research.tests.test_sprite_overnight \
        > "$sprite_dir/host-tests.txt" 2>&1
    sprite_mode=overnight
    if [[ "$sprite_command" == prepare-smoke ]]; then sprite_mode=smoke; fi
    bash research/test_sprite_packed.sh "$sprite_dir/raster" > "$sprite_dir/raster.txt" 2>&1
    bash research/build_sprite_benchmarks.sh "$sprite_dir/build" > "$sprite_dir/build.txt" 2>&1
    bash research/build_sprite_packed_checks.sh "$sprite_dir/build" > "$sprite_dir/checks-build.txt" 2>&1
    .venv/bin/python research/sprite_overnight.py prepare --build "$sprite_dir/build" \
        --out "$sprite_dir/campaign" --mode "$sprite_mode" > "$sprite_dir/preparation.txt" 2>&1
    printf 'Prepared without GPU execution: %s\n' "$sprite_dir"
    printf 'When the GPU is idle: bash research/run_sprite_overnight.sh %s %q\n' \
        "${sprite_command/prepare/run}" "$sprite_dir"
    exit 0
fi
sprite_dir=$(realpath "$sprite_dir")
[[ -f "$sprite_dir/campaign/plan.json" ]] || { echo 'Prepare this directory first.' >&2; exit 1; }
if [[ "$sprite_command" == export ]]; then
    sprite_archive="build/hardware-artifacts/sprite-overnight.$(basename "$sprite_dir").$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
    .venv/bin/python research/export_sprite_overnight.py --campaign "$sprite_dir/campaign" --out "$sprite_archive"
    exit 0
fi
if [[ "$sprite_command" == report ]]; then
    .venv/bin/python research/sprite_overnight.py report --out "$sprite_dir/campaign"
    exit 0
fi
sprite_gpu=5090
if [[ "$sprite_command" == run-smoke ]]; then sprite_gpu=5060; fi
sprite_status=0
.venv/bin/python research/sprite_overnight.py run --out "$sprite_dir/campaign" \
    --expected-gpu "$sprite_gpu" > "$sprite_dir/run.txt" 2>&1 || sprite_status=$?
printf 'Exit status: %s\nResults: %s/campaign/result.json\nCurves: %s/campaign/curves.html\n' \
    "$sprite_status" "$sprite_dir" "$sprite_dir"
exit "$sprite_status"
