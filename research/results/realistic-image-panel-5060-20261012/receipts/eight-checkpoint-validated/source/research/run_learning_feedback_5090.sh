#!/usr/bin/env bash
# Two commands: build/prepare without GPU, then explicit staged learning run.
set -euo pipefail
cd "$(dirname "$0")/.."
learning_command="${1:-}"
[[ $# -le 2 && ( "$learning_command" == prepare || "$learning_command" == run ) ]] || {
    echo 'Usage: bash research/run_learning_feedback_5090.sh prepare [FRESH_DIR]' >&2
    echo '       bash research/run_learning_feedback_5090.sh run PREPARED_DIR' >&2
    exit 2
}
if [[ -f build/connect4cnn/runtime-5090.sh ]]; then source build/connect4cnn/runtime-5090.sh; fi
source ocean/connect4cnn/runtime_env.sh
export NVCC_ARCH=sm_120
unset NVCC_PREPEND_FLAGS
if [[ ! -x .venv/bin/python ]]; then
    [[ "$learning_command" == prepare ]] || exit 1
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python 'numpy==2.5.3'
fi
.venv/bin/python -c 'import numpy, sys; assert sys.version_info[:2] == (3,12), "Use uv venv --python 3.12"'
if [[ "$learning_command" == prepare ]]; then
    learning_root="${2:-build/learning-feedback/5090-$(date -u +%Y%m%dT%H%M%SZ)}"
    [[ ! -e "$learning_root" ]] || { echo 'Use a fresh allocation.' >&2; exit 1; }
    mkdir -p "$learning_root"; learning_root=$(realpath "$learning_root")
    git rev-parse HEAD > "$learning_root/revision.txt"
    git status --short > "$learning_root/worktree.txt"
    .venv/bin/python -m unittest research.tests.test_feedback_campaign research.tests.test_cnn_graph_acceptance \
        research.tests.test_cross_game_feedback research.tests.test_candidate_panel research.tests.test_candidate_mini \
        research.tests.test_candidate_frontiers research.tests.test_candidate_viewer > "$learning_root/host-tests.txt" 2>&1
    bash research/build_protein_feedback.sh "$learning_root/optimizer" > "$learning_root/optimizer-build.txt" 2>&1
    PROTEIN_FEEDBACK_HOST_BINARY="$learning_root/optimizer/worker" \
        .venv/bin/python -m unittest research.tests.test_native_feedback_host > "$learning_root/native-host-tests.txt" 2>&1
    .venv/bin/python research/candidate_panel.py build --out "$learning_root/games" > "$learning_root/game-builds.txt" 2>&1
    bash research/build_policy_metadata.sh "$learning_root/metadata" > "$learning_root/metadata-builds.txt" 2>&1
    bash ocean/connect4cnn/tests/build_encoder_test.sh test_nature "$learning_root/native.so"
    bash research/build_native_flex_reference.sh "$learning_root/reference.so"
    .venv/bin/python research/cnn_graph_acceptance.py bundle --native "$learning_root/native.so" \
        --reference "$learning_root/reference.so" --out "$learning_root/validation" > "$learning_root/validation-preparation.txt"
    .venv/bin/python research/feedback_campaign.py prepare --study research/recipes/feedback_learning.ini \
        --registry "$learning_root/games/registry.json" --metadata "$learning_root/metadata" \
        --optimizer "$learning_root/optimizer" --validation "$learning_root/validation" \
        --out "$learning_root/prepared" > "$learning_root/preparation.txt" 2>&1
    .venv/bin/python research/feedback_campaign.py inspect --out "$learning_root/prepared" > "$learning_root/inspection.txt"
    printf 'GPU-free preparation passed: %s\nExplicit RTX 5090 launch:\n  bash research/run_learning_feedback_5090.sh run %q\n' "$learning_root" "$learning_root"
    exit 0
fi
[[ $# == 2 && -f "$2/prepared/campaign.json" ]] || { echo 'Supply the directory printed by prepare.' >&2; exit 2; }
learning_root=$(realpath "$2")
[[ ! -e "$learning_root/prepared/execution" ]] || { echo 'Already started; no automatic retry.' >&2; exit 1; }
learning_status=0
.venv/bin/python research/feedback_campaign.py run --out "$learning_root/prepared" --allow-gpu \
    > "$learning_root/run.txt" 2>&1 || learning_status=$?
# Successful, failed and blocked-learning packets all retain evidence.
mkdir -p build/hardware-artifacts
learning_archive=$(mktemp "build/hardware-artifacts/learning-feedback-5090.$(basename "$learning_root").XXXXXXXX.tar.gz")
tar -czf "$learning_archive" -C "$(dirname "$learning_root")" "$(basename "$learning_root")"
sha256sum "$learning_archive" | tee "$learning_archive.sha256"
printf 'Exit status: %s\nEvidence: %s\nLogs: %s\n' "$learning_status" "$learning_archive" "$learning_root"
exit "$learning_status"
