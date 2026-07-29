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

if (( dry_run == 1 )); then
    exec bash "$base_script" --dry-run \
        "$run_id" "$timesteps" "$total_agents"
fi

exec bash "$base_script" "$run_id" "$timesteps" "$total_agents"
