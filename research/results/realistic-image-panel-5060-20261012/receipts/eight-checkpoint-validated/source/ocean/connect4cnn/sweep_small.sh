#!/usr/bin/env bash
# Serial campaigns: both use the same native trainer, budget range, and sidecar.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p build/connect4cnn
campaign_dir=$(mktemp -d build/connect4cnn/fast-search.XXXXXXXX)
printf 'Campaign: %s\n' "$campaign_dir"
for family in nature compact; do
    bash ocean/connect4cnn/sweep.sh --recipe "ocean/connect4cnn/sweep_${family}.ini" \
        --timeout 14400 --wandb online --project puffer-cnn --entity kinvert-k "$@" \
        > "$campaign_dir/$family.log" 2>&1
done
printf 'Completed both sweeps: %s\n' "$campaign_dir"
