#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${PUFFER_BIN:-$repo_root/puffer}"
dry_run=0

usage() {
    cat >&2 <<'EOF'
usage: train_reproduction.sh [--dry-run] RUN_ID [TIMESTEPS] [FIXED_STAGE]

Reproduces the stock-core Phase 2 training profile. FIXED_STAGE defaults to 0;
use -1 for the environment-managed curriculum. The 0.02 entropy coefficient
is a reproduction value, not an accepted long-run production default.
EOF
    exit 2
}

if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi

run_id="${1:-}"
timesteps="${2:-33554432}"
fixed_stage="${3:-0}"
[[ "$#" -ge 1 && "$#" -le 3 ]] || usage

[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "run id may contain only letters, digits, dot, underscore, and dash" >&2
    exit 2
}
[[ "$timesteps" =~ ^[1-9][0-9]*$ ]] || {
    echo "timesteps must be a positive integer" >&2
    exit 2
}
[[ "$fixed_stage" =~ ^-?[0-9]+$ ]] &&
    ((fixed_stage >= -1 && fixed_stage <= 20)) || {
    echo "fixed stage must be -1 or an integer from 0 through 20" >&2
    exit 2
}

command=(
    "$binary"
    train
    dogfight
    "base.run_id=$run_id"
    "base.seed=42"
    "base.async=0"
    "vec.total_agents=4096"
    "vec.num_buffers=4"
    "vec.num_frozen_banks=0"
    "vec.frozen_bank_pct=0"
    "selfplay.enabled=0"
    "policy.hidden_size=64"
    "policy.num_layers=3"
    "train.total_timesteps=$timesteps"
    "train.gpus=1"
    "train.learning_rate=0.0025"
    "train.minibatch_size=4096"
    "train.gamma=0.996"
    "train.gae_lambda=0.999"
    "train.ent_coef=0.02"
    "train.clip_coef=0.06"
    "train.vf_coef=4.6"
    "train.vf_clip_coef=1.5"
    "train.max_grad_norm=3.4"
    "train.momentum=0.9896"
    "train.prio_alpha=0.4"
    "train.prio_beta0=0.82"
    "train.replay_ratio=0.95"
    "train.vtrace_rho_clip=0.1"
    "train.vtrace_c_clip=2.5"
    "env.num_agents=1"
    "env.curriculum_enabled=1"
    "env.curriculum_randomize=0"
    "env.curriculum_target=0.9"
    "env.fixed_stage=$fixed_stage"
    "env.max_stage=2"
    "env.global_step_stride=4096"
    "env.role_randomization=0"
    "env.domain_randomization=0.05"
    "env.vertical_spawn_prob=0.02"
    "env.warmup_steps=1000000"
    "env.eval_interval=60000"
    "env.min_eval_episodes=50"
)

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
