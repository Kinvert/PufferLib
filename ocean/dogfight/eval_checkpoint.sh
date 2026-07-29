#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${DOGFIGHT_EVAL_BIN:-$repo_root/build/puffer-dogfight-eval}"
if [[ -n "${PUFFER_VENV:-}" ]]; then
    puffer_venv="$PUFFER_VENV"
elif [[ -d "$repo_root/.venv" ]]; then
    puffer_venv="$repo_root/.venv"
else
    puffer_venv="/home/claude/PufferLib/.venv"
fi
nccl_lib=""
for candidate in "$puffer_venv"/lib/python*/site-packages/nvidia/nccl/lib; do
    if [[ -d "$candidate" ]]; then
        nccl_lib="$candidate"
        break
    fi
done
if [[ -n "$nccl_lib" ]]; then
    export LD_LIBRARY_PATH="$nccl_lib:${LD_LIBRARY_PATH:-}"
fi
dry_run=0

usage() {
    cat >&2 <<'EOF'
usage:
  eval_checkpoint.sh [--dry-run] headless CHECKPOINT STAGE [EPISODES] [SEED] [MIRROR]
  eval_checkpoint.sh [--dry-run] visible  CHECKPOINT STAGE [EPISODES] [SEED] [MIRROR]

MIRROR is 0 or 1. Identical STAGE and SEED with both values form an exact pair.
The legacy order CHECKPOINT STAGE EPISODES SEED MODE [MIRROR] is also accepted.
Visible evaluation requires DISPLAY (normally DISPLAY=:0).
EOF
    exit 2
}

if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi

if [[ "${1:-}" == "headless" || "${1:-}" == "visible" ]]; then
    mode="$1"
    checkpoint="${2:-}"
    stage="${3:-}"
    episodes="${4:-128}"
    seed="${5:-42}"
    mirror="${6:-0}"
    [[ "$#" -ge 3 && "$#" -le 6 ]] || usage
else
    checkpoint="${1:-}"
    stage="${2:-}"
    episodes="${3:-128}"
    seed="${4:-42}"
    mode="${5:-}"
    mirror="${6:-0}"
    [[ "$#" -ge 2 && "$#" -le 6 ]] || usage
fi

[[ "$mode" == "headless" || "$mode" == "visible" ]] || usage
[[ -f "$checkpoint" ]] || {
    echo "checkpoint does not exist: $checkpoint" >&2
    exit 2
}
[[ "$stage" =~ ^[0-9]+$ ]] && ((stage <= 20)) || {
    echo "stage must be an integer from 0 through 20" >&2
    exit 2
}
[[ "$episodes" =~ ^[1-9][0-9]*$ ]] || {
    echo "episodes must be a positive integer" >&2
    exit 2
}
[[ "$seed" =~ ^[0-9]+$ ]] || {
    echo "seed must be a nonnegative integer" >&2
    exit 2
}
[[ "$mirror" == "0" || "$mirror" == "1" ]] || {
    echo "mirror must be 0 or 1" >&2
    exit 2
}

common=(
    dogfight
    "base.load_model_path=$checkpoint"
    "base.seed=$seed"
    "base.async=0"
    "base.num_games=$episodes"
    "base.burnin_games=0"
    "base.eval_agents=2"
    "vec.num_frozen_banks=0"
    "vec.frozen_bank_pct=0"
    "selfplay.enabled=0"
    "env.num_agents=1"
    "env.curriculum_enabled=1"
    "env.curriculum_randomize=0"
    "env.curriculum_target=$stage"
    "env.fixed_stage=$stage"
    "env.rehearsal_stage=-1"
    "env.rehearsal_stride=0"
    "env.role_randomization=0"
    "env.domain_randomization=0"
    "env.vertical_spawn_prob=0"
    "env.eval_spawn_mode=0"
    "env.eval_lateral_mirror=$mirror"
)

if [[ "$mode" == "headless" ]]; then
    command=("$binary" eval_bot "${common[@]}")
else
    command=("$binary" render "${common[@]}")
fi

if ((dry_run)); then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

[[ -x "$binary" ]] || {
    echo "missing evaluator: $binary" >&2
    echo "build it with: CUDA_HOME=/usr/local/cuda bash ocean/dogfight/build_eval.sh" >&2
    exit 2
}
if [[ "$mode" == "visible" && -z "${DISPLAY:-}" ]]; then
    echo "visible evaluation requires DISPLAY (normally DISPLAY=:0)" >&2
    exit 2
fi

exec "${command[@]}"
