#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${DOGFIGHT_TRAIN_BIN:-$repo_root/puffer}"
dry_run=0

usage() {
    cat >&2 <<'EOF'
usage: train_vanilla_selfplay.sh [--dry-run] RUN_ID [TIMESTEPS] [TOTAL_AGENTS] [SEED] [BOOTSTRAP_STEPS]

Runs the bounded Phase 2 Dogfight profile using PufferLib's official native
self-play pool. Defaults: TIMESTEPS=8388608, TOTAL_AGENTS=4096.

This is a source-derived experimental profile, not permanent tuning gospel.
It deliberately uses synchronous collection, fixed stage-0 spawns, one frozen
bank, and no Dogfight coordinator or environment-managed curriculum.
Set DOGFIGHT_FROZEN_BANK_PCT=0 to isolate current-current training from stock
5c's known frozen-row loss contamination.
EOF
    exit 2
}

if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi

run_id="${1:-}"
timesteps="${2:-8388608}"
total_agents="${3:-4096}"
seed="${4:-42}"
bootstrap_steps="${5:-0}"
frozen_bank_pct="${DOGFIGHT_FROZEN_BANK_PCT:-0.1}"
control_rate_penalty="${DOGFIGHT_CONTROL_RATE_PENALTY:-0.002}"
aileron_magnitude_penalty="${DOGFIGHT_AILERON_MAGNITUDE_PENALTY:-0}"
bootstrap_imitation_scale="${DOGFIGHT_BOOTSTRAP_IMITATION_SCALE:-0}"
[[ "$#" -ge 1 && "$#" -le 5 ]] || usage

[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "run id may contain only letters, digits, dot, underscore, and dash" >&2
    exit 2
}
[[ "$timesteps" =~ ^[1-9][0-9]*$ ]] || {
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
[[ "$bootstrap_steps" =~ ^[0-9]+$ ]] || {
    echo "bootstrap steps must be a nonnegative integer" >&2
    exit 2
}
[[ "$frozen_bank_pct" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || {
    echo "DOGFIGHT_FROZEN_BANK_PCT must be between 0 and 1" >&2
    exit 2
}
[[ "$control_rate_penalty" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "DOGFIGHT_CONTROL_RATE_PENALTY must be nonnegative" >&2
    exit 2
}
[[ "$aileron_magnitude_penalty" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "DOGFIGHT_AILERON_MAGNITUDE_PENALTY must be nonnegative" >&2
    exit 2
}
[[ "$bootstrap_imitation_scale" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "DOGFIGHT_BOOTSTRAP_IMITATION_SCALE must be nonnegative" >&2
    exit 2
}
frozen_banks=1
selfplay_enabled=1
if [[ "$frozen_bank_pct" =~ ^0([.]0+)?$ ]]; then
    frozen_banks=0
    selfplay_enabled=0
fi
((total_agents >= 256 && total_agents % 8 == 0)) || {
    echo "total agents must be at least 256 and divisible by 8" >&2
    exit 2
}

command=(
    "$binary"
    train
    dogfight
    "base.run_id=$run_id"
    "base.seed=$seed"
    "base.async=0"
    "base.checkpoint_interval=8"
    "vec.total_agents=$total_agents"
    "vec.num_buffers=4"
    "vec.num_threads=8"
    "vec.num_frozen_banks=$frozen_banks"
    "vec.frozen_bank_pct=$frozen_bank_pct"
    "vec.frozen_bank_hidden_size=64"
    "vec.frozen_bank_num_layers=3"
    "selfplay.mode=vanilla"
    "selfplay.enabled=$selfplay_enabled"
    "selfplay.max_size=8"
    "selfplay.seed=$seed"
    "selfplay.opp_timeout_steps=2097152"
    "selfplay.eval_games=0"
    "selfplay.eval_pool_size=0"
    "policy.hidden_size=64"
    "policy.num_layers=3"
    "train.total_timesteps=$timesteps"
    "train.gpus=1"
    "train.learning_rate=0.00045"
    "train.gamma=0.99"
    "train.gae_lambda=0.999"
    "train.clip_coef=0.11"
    "train.vf_coef=2.9"
    "train.vf_clip_coef=1.5"
    "train.max_grad_norm=1.8"
    "train.ent_coef=0.0024"
    "train.minibatch_size=65536"
    "train.horizon=64"
    "train.prio_alpha=0.99"
    "train.prio_beta0=0.99"
    "env.num_agents=2"
    "env.max_steps=300"
    "env.obs_scheme=1"
    "env.curriculum_enabled=1"
    "env.curriculum_randomize=0"
    "env.curriculum_target=0"
    "env.fixed_stage=0"
    "env.rehearsal_stage=-1"
    "env.rehearsal_stride=0"
    "env.max_stage=0"
    "env.reward_version=1"
    "env.control_rate_penalty=$control_rate_penalty"
    "env.aileron_magnitude_penalty=$aileron_magnitude_penalty"
    "env.role_randomization=1"
    "env.selfplay_bootstrap_steps=$bootstrap_steps"
    "env.selfplay_bootstrap_imitation_scale=$bootstrap_imitation_scale"
    "env.domain_randomization=0"
    "env.vertical_spawn_prob=0"
    "env.recovery_enabled=0"
    "env.global_step_stride=$total_agents"
    "env.eval_spawn_mode=0"
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
