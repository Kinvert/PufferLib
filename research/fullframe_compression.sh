#!/usr/bin/env bash
# Native model implementation; Python supervises captured processes and receipts.
set -euo pipefail
cd "$(dirname "$0")/.."
source ocean/connect4cnn/runtime_env.sh
exec .venv/bin/python research/fullframe_compression.py "$@"
