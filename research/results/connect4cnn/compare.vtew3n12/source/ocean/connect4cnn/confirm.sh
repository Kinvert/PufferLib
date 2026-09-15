#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ocean/connect4cnn/runtime_env.sh
export OPENBLAS_NUM_THREADS=1
exec .venv/bin/python ocean/connect4cnn/compare.py --confirmation --wandb online --project cnn2 --entity kinvert-k "$@"
