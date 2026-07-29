#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base_script="$script_dir/train_vanilla_selfplay.sh"

usage() {
    echo "usage: $0 [--dry-run] RUN_ID [TIMESTEPS] [TOTAL_AGENTS]" >&2
}

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi

if (( $# < 1 || $# > 3 )); then
    usage
    exit 2
fi

run_id="$1"
timesteps="${2:-134217728}"
total_agents="${3:-16384}"

# Reuse the tested fixed-stage launcher as the single source of truth for
# native Puffer self-play, PPO, topology, and validation settings.
base_command="$(
    bash "$base_script" --dry-run "$run_id" "$timesteps" "$total_agents"
)"
read -r -a command <<< "$base_command"

replace_exact() {
    local from="$1"
    local to="$2"
    local found=0
    local i

    for i in "${!command[@]}"; do
        if [[ "${command[$i]}" == "$from" ]]; then
            command[$i]="$to"
            found=1
        fi
    done

    if (( found == 0 )); then
        echo "expected base profile token not found: $from" >&2
        exit 1
    fi
}

# Dogfight's existing stages progressively widen the starting geometry. This
# schedule stays inside the environment and runs continuously with Puffer's
# official current-policy and historical-policy routing.
replace_exact "env.curriculum_target=0" "env.curriculum_target=0.9"
replace_exact "env.fixed_stage=0" "env.fixed_stage=-1"
replace_exact "env.max_stage=0" "env.max_stage=20"
command+=("env.curriculum_total_steps=$timesteps")

printf '%q ' "${command[@]}"
printf '\n'

if (( dry_run == 1 )); then
    exit 0
fi

exec "${command[@]}"
