#!/usr/bin/env bash
# Detached native continuation. Preserve failure evidence; never auto-retry.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKET="${1:?Provide the fresh prepared continuation packet}"
SOURCE="/home/keith/Git/ml/cnn-5090-learning"
source "$SOURCE/build/connect4cnn/runtime-5090.sh"
set +e
"$SOURCE/.venv/bin/python" "$ROOT/research/continue_learning_feedback.py" run \
    --source "$SOURCE" --out "$PACKET" --allow-gpu
RUN_STATUS=$?
printf '%s\n' "$RUN_STATUS" > "$PACKET/launcher-exit-code.txt"
"$SOURCE/.venv/bin/python" "$ROOT/research/report_learning_progress.py" "$PACKET" \
    --out "$PACKET/sps-report"
REPORT_STATUS=$?
printf '%s\n' "$REPORT_STATUS" > "$PACKET/sps-report-exit-code.txt"
set -e
if [[ "$RUN_STATUS" -ne 0 ]]; then exit "$RUN_STATUS"; fi
exit "$REPORT_STATUS"
