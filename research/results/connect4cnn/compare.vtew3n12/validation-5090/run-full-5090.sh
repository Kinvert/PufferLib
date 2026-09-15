#!/usr/bin/env bash
set -euo pipefail
cd /home/keith/Git/ml/cnn-5090
trap 'code=$?; printf "%s\n" "$code" > build/connect4cnn/hardware-exit.txt; tmux wait-for -S cnn-hardware-finished' EXIT
source build/connect4cnn/runtime-5090.sh
test -f build/connect4cnn/validation-5090/finished.txt
git status --short > build/connect4cnn/validation-5090/pre-full-git-status.txt
bash ocean/connect4cnn/hardware_compare.sh --full
run_dir=$(sed -n 's/^Comparison: //p' build/connect4cnn/hardware-launch.log | head -n 1)
cp -a build/connect4cnn/validation-5090 "$run_dir/validation-5090"
cp build/connect4cnn/runtime-5090.sh build/connect4cnn/run-full-5090.sh build/connect4cnn/validate-5090.sh "$run_dir/validation-5090/"
bash ocean/connect4cnn/package_results.sh "$run_dir"
