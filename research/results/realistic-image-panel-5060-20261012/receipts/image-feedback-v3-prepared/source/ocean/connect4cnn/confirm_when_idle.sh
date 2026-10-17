#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Wait outside the timed experiment; do not interrupt another GPU user.
deadline=$((SECONDS + 3600))
while true; do
    active=$(/usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    if [[ -z "${active//[[:space:]]/}" ]]; then
        break
    fi
    if (( SECONDS >= deadline )); then
        echo "GPU remained busy for one hour; confirmation was not started." >&2
        exit 1
    fi
    echo "Waiting for GPU compute processes: $active"
    sleep 30
done
exec bash ocean/connect4cnn/confirm.sh "$@"
