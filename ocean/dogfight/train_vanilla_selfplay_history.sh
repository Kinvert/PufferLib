#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base_script="$script_dir/train_vanilla_selfplay.sh"

usage() {
    echo "usage: $0 [--dry-run] RUN_ID [TIMESTEPS] [TOTAL_AGENTS] [FROZEN_FRACTION] [POOL_SIZE]" >&2
}

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi

if (( $# < 1 || $# > 5 )); then
    usage
    exit 2
fi

run_id="$1"
timesteps="${2:-134217728}"
total_agents="${3:-16384}"
frozen_fraction="${4:-0.5}"
pool_size="${5:-100}"

case "$frozen_fraction" in
    0.1|0.25|0.5|0.75) ;;
    *)
        echo "frozen fraction must be one of: 0.1, 0.25, 0.5, 0.75" >&2
        exit 2
        ;;
esac

if [[ ! "$pool_size" =~ ^[0-9]+$ ]] ||
        (( pool_size < 1 || pool_size > 1000 )); then
    echo "pool size must be an integer from 1 through 1000" >&2
    exit 2
fi

base_command="$(
    bash "$base_script" --dry-run "$run_id" "$timesteps" "$total_agents"
)"
read -r -a command <<< "$base_command"

command+=(
    "vec.frozen_bank_pct=$frozen_fraction"
    "selfplay.max_size=$pool_size"
)

printf '%q ' "${command[@]}"
printf '\n'

if (( dry_run == 1 )); then
    exit 0
fi

exec "${command[@]}"
