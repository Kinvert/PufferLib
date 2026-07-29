#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${DOGFIGHT_TRAIN_BIN:-$repo_root/puffer}"
puffer_venv="${PUFFER_VENV:-/home/claude/PufferLib/.venv}"
nccl_lib="$puffer_venv/lib/python3.12/site-packages/nvidia/nccl/lib"
if [[ -d "$nccl_lib" ]]; then
    export LD_LIBRARY_PATH="$nccl_lib:${LD_LIBRARY_PATH:-}"
fi
dry_run=0

usage() {
    cat >&2 <<'EOF'
usage: train_vanilla_selfplay.sh [--dry-run] RUN_ID [TIMESTEPS] [TOTAL_AGENTS] [SEED]

Runs the config/dogfight.ini Robocode-derived native self-play profile.
Only run-specific values are command-line overrides. When TIMESTEPS is omitted,
the checked-in config/dogfight.ini duration is used. Defaults:
TOTAL_AGENTS=4096, SEED=42.
EOF
    exit 2
}

if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi

run_id="${1:-}"
timesteps="${2:-}"
total_agents="${3:-4096}"
seed="${4:-42}"
[[ "$#" -ge 1 && "$#" -le 4 ]] || usage

[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "run id may contain only letters, digits, dot, underscore, and dash" >&2
    exit 2
}
[[ -z "$timesteps" || "$timesteps" =~ ^[1-9][0-9]*$ ]] || {
    echo "timesteps must be a positive integer" >&2
    exit 2
}
[[ "$total_agents" =~ ^[1-9][0-9]*$ ]] || {
    echo "total agents must be a positive integer" >&2
    exit 2
}
[[ "$seed" =~ ^[0-9]+$ ]] || {
    echo "seed must be a nonnegative integer" >&2
    exit 2
}
((total_agents >= 256 && total_agents % 8 == 0)) || {
    echo "total agents must be at least 256 and divisible by 8" >&2
    exit 2
}

command=(
    "$binary"
    train
    dogfight
    "--wandb"
    "--wandb-project=df42"
    "base.run_id=$run_id"
    "base.seed=$seed"
    "train.seed=$seed"
    "selfplay.seed=$seed"
    "vec.total_agents=$total_agents"
    "env.global_step_stride=$total_agents"
)
if [[ -n "$timesteps" ]]; then
    command+=(
        "train.total_timesteps=$timesteps"
        "env.curriculum_total_steps=$timesteps"
    )
fi

if ((dry_run)); then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

[[ -x "$binary" ]] || {
    echo "missing native binary: $binary" >&2
    echo "build it with: CUDA_HOME=/usr/local/cuda-12.8 ./build.sh dogfight" >&2
    exit 2
}

exec "${command[@]}"
