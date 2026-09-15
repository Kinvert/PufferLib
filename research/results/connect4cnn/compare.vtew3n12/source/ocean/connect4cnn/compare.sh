#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ocean/connect4cnn/runtime_env.sh
exec .venv/bin/python ocean/connect4cnn/compare.py "$@"
