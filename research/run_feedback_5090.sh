#!/usr/bin/env bash
# Build/preparation is GPU-free. Only the explicit run command uses the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

feedback_command="${1:-}"
if [[ "$feedback_command" != prepare && "$feedback_command" != run ]] || [[ $# -gt 2 ]]; then
    echo 'Usage: bash research/run_feedback_5090.sh prepare [FRESH_DIR]' >&2
    echo '       bash research/run_feedback_5090.sh run PREPARED_DIR' >&2
    exit 2
fi

# Reuse the established 5090 environment when present. No CUDA installations.
if [[ -f build/connect4cnn/runtime-5090.sh ]]; then
    source build/connect4cnn/runtime-5090.sh
fi
source ocean/connect4cnn/runtime_env.sh
export NVCC_ARCH=sm_120
export NVCC_PREPEND_FLAGS='--threads 1'
if [[ ! -x .venv/bin/python ]]; then
    [[ "$feedback_command" == prepare ]] || { echo 'Run prepare first.' >&2; exit 1; }
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python 'numpy==2.5.3'
fi
.venv/bin/python -c 'import numpy; import sys; assert sys.version_info[:2] == (3, 12), "Use uv venv --python 3.12"'

if [[ "$feedback_command" == prepare ]]; then
    [[ -x "$CUDA_HOME/bin/nvcc" && -f "$NCCL_ROOT/include/nccl.h" ]] || {
        echo 'Point CUDA_HOME/NCCL_ROOT at the existing toolkit/NCCL. Do not install or change CUDA.' >&2
        exit 1
    }
    feedback_dir="${2:-build/cross-game-feedback/5090-$(date -u +%Y%m%dT%H%M%SZ)}"
    [[ ! -e "$feedback_dir" ]] || { echo "Refusing existing directory: $feedback_dir" >&2; exit 1; }
    mkdir -p "$feedback_dir"
    feedback_dir=$(realpath "$feedback_dir")
    git rev-parse HEAD > "$feedback_dir/revision.txt"
    git status --short > "$feedback_dir/worktree.txt"
    .venv/bin/python -m unittest research.tests.test_cross_game_feedback \
        research.tests.test_candidate_mini research.tests.test_candidate_panel \
        research.tests.test_candidate_frontiers research.tests.test_candidate_viewer \
        > "$feedback_dir/host-tests.txt" 2>&1
    bash research/build_protein_feedback.sh "$feedback_dir/optimizer" > "$feedback_dir/optimizer-build.txt" 2>&1
    PROTEIN_FEEDBACK_HOST_BINARY="$feedback_dir/optimizer/worker" \
        .venv/bin/python -m unittest research.tests.test_native_feedback_host \
        > "$feedback_dir/native-host-tests.txt" 2>&1
    .venv/bin/python research/candidate_panel.py build --out "$feedback_dir/games" \
        > "$feedback_dir/game-builds.txt" 2>&1
    bash research/build_policy_metadata.sh "$feedback_dir/metadata" > "$feedback_dir/metadata-builds.txt" 2>&1
    .venv/bin/python research/cross_game_feedback.py prepare \
        --recipe research/recipes/cross_game_feedback_smoke.ini \
        --registry "$feedback_dir/games/registry.json" --optimizer-build "$feedback_dir/optimizer" \
        --policy-metadata "$feedback_dir/metadata" --out "$feedback_dir/prepared" \
        > "$feedback_dir/preparation.txt" 2>&1
    .venv/bin/python research/cross_game_feedback.py inspect --out "$feedback_dir/prepared" \
        > "$feedback_dir/inspection.txt" 2>&1
    printf 'GPU-free preparation passed: %s\nRun when the 5090 is idle:\n  bash research/run_feedback_5090.sh run %q\n' \
        "$feedback_dir" "$feedback_dir"
    exit 0
fi

[[ $# == 2 && -f "$2/prepared/plan.json" ]] || { echo 'Specify the directory printed by prepare.' >&2; exit 2; }
feedback_dir=$(realpath "$2")
[[ ! -e "$feedback_dir/prepared/execution" && ! -e "$feedback_dir/review" ]] || {
    echo 'Allocation already started. Preserve it; no automatic restart.' >&2
    exit 1
}
feedback_status=0
.venv/bin/python research/cross_game_feedback.py run --plan "$feedback_dir/prepared/plan.json" \
    --mode canary5090 --allow-gpu > "$feedback_dir/run.txt" 2>&1 || feedback_status=$?
if [[ "$feedback_status" == 0 ]]; then
    .venv/bin/python research/cross_game_feedback.py audit --plan "$feedback_dir/prepared/plan.json" \
        --out "$feedback_dir/review" > "$feedback_dir/audit.txt" 2>&1 || feedback_status=$?
fi
# Keep failed and partial allocations too. Archives are evidence, not a pass.
mkdir -p build/hardware-artifacts
feedback_archive=$(mktemp "build/hardware-artifacts/feedback-5090.$(basename "$feedback_dir").XXXXXXXX.tar.gz")
tar -czf "$feedback_archive" -C "$(dirname "$feedback_dir")" "$(basename "$feedback_dir")"
sha256sum "$feedback_archive" | tee "$feedback_archive.sha256"
printf 'Exit status: %s\nEvidence: %s\nLogs: %s\n' "$feedback_status" "$feedback_archive" "$feedback_dir"
exit "$feedback_status"
