#!/usr/bin/env bash
# Explicitly scheduled hardware pilot: canary, then four serial matched runs.
set -euo pipefail
cd "$(dirname "$0")/../.."
gpu=${1:?Usage: benchmark_local.sh 5060|5090}
[[ "$gpu" == 5060 || "$gpu" == 5090 ]] || exit 2
source ocean/connect4cnn/runtime_env.sh
export OPENBLAS_NUM_THREADS=1 NVCC_ARCH=sm_120
smi=$(puffer_find_nvidia_smi)
"$smi" --query-gpu=name --format=csv,noheader | rg -q "RTX $gpu"
puffer_require_idle_gpu "$smi"
mkdir -p build/connect4cnn
exec 9>build/connect4cnn/hardware-benchmark.lock
flock -n 9 || { echo 'Another hardware benchmark owns this checkout.' >&2; exit 1; }
campaign=$(mktemp -d "build/connect4cnn/hardware-$gpu.XXXXXXXX")
receipts="research/results/connect4cnn/rtx-$gpu/$(basename "$campaign")"
mkdir -p "$receipts"
printf 'Campaign: %s\nReceipts: %s\n' "$campaign" "$receipts"
git rev-parse HEAD > "$receipts/revision.txt"
git diff HEAD > "$receipts/working.diff"
printf 'running\n' > "$receipts/status.txt"
finish() {
    code=$?
    trap - EXIT
    printf 'exit_code=%s\n' "$code" > "$receipts/status.txt"
    cp "$campaign"/*.log "$receipts/" 2>/dev/null || true
    exit "$code"
}
trap finish EXIT
bash ocean/connect4cnn/hardware_compare.sh --canary 2>&1 | tee "$campaign/canary.log"
# hardware_compare returns nonzero on any failed job. It writes 16 evaluations.
canary=$(sed -n 's/^Comparison: //p' "$campaign/canary.log" | head -n 1)
.venv/bin/python -c 'import json,sys; from pathlib import Path; x=json.loads((Path(sys.argv[1])/"finished.json").read_text()); assert x == dict(status="ok",jobs=4,evaluations=16,logging_failures=0),x' "$canary"
printf '%s\n' "$canary" > "$receipts/canary-path.txt"
bash ocean/connect4cnn/hardware_compare.sh --full 2>&1 | tee "$campaign/full.log"
full=$(sed -n 's/^Comparison: //p' "$campaign/full.log" | head -n 1)
printf '%s\n' "$full" > "$receipts/full-path.txt"
.venv/bin/python research/report_hardware_run.py "$full" --gpu "$gpu" --output "$receipts/full" \
    2>&1 | tee "$campaign/archive.log"
