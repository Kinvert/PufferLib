#!/usr/bin/env bash
set -euo pipefail
cd /home/keith/Git/ml/cnn-5090
trap 'code=$?; printf "%s\n" "$code" > build/connect4cnn/hardware-audit-exit.txt; tmux wait-for -S cnn-hardware-audited' EXIT
tmux wait-for cnn-hardware-finished
test "$(cat build/connect4cnn/hardware-exit.txt)" = 0
run_dir=$(sed -n 's/^Comparison: //p' build/connect4cnn/hardware-launch.log | head -n 1)
cp build/connect4cnn/audit-5090.py build/connect4cnn/finalize-5090.sh "$run_dir/validation-5090/"
cp -a ocean/connect4cnn/tests "$run_dir/validation-5090/test-source"
find "$run_dir/validation-5090/test-source" -type f -not -path '*/__pycache__/*' -exec sha256sum {} + > "$run_dir/validation-5090/test-source.sha256"
.venv/bin/python build/connect4cnn/audit-5090.py "$run_dir"
bash ocean/connect4cnn/package_results.sh "$run_dir"
