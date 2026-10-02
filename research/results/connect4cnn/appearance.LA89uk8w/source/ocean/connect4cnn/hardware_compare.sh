#!/usr/bin/env bash
# Fixed four-model hardware check, no architecture or hyperparameter search.
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:---full}
if (( $# > 1 )) || [[ "$mode" != --full && "$mode" != --canary && "$mode" != --print-command ]]; then
    echo "Usage: bash ocean/connect4cnn/hardware_compare.sh [--canary|--full|--print-command]" >&2
    exit 2
fi
args=(--variants flex_quality nature_cnn impala_cnn impoola_cnn
    --steps 13312000 --checkpoints 13 --seeds 173 --eval-seed 20173 --eval-games 1024 --timeout 2400
    --wandb disabled --require-idle-gpu --note "Fixed four-model hardware comparison; seed 173; same-revision cross-host timing, not a new sweep")
if [[ "$mode" == --canary ]]; then
    args=(--variants flex_quality nature_cnn impala_cnn impoola_cnn
        --steps 65536 --checkpoints 4 --seeds 9173 --eval-seed 29173 --eval-games 128 --timeout 120
        --wandb disabled --require-idle-gpu --note "Hardware CANARY: four-model build/train/reload plumbing only; exclude from speed claims")
fi
if [[ "$mode" == --print-command ]]; then
    printf '%q ' .venv/bin/python ocean/connect4cnn/compare.py "${args[@]}"
    printf '\n'
    exit 0
fi
test -x .venv/bin/python || { echo 'Create this checkout’s Python 3.12 uv venv as documented in research/HARDWARE_COMPARISON.md.' >&2; exit 1; }
source ocean/connect4cnn/runtime_env.sh
export OPENBLAS_NUM_THREADS=1
export NVCC_ARCH="${NVCC_ARCH:-sm_120}"
exec .venv/bin/python ocean/connect4cnn/compare.py "${args[@]}"
