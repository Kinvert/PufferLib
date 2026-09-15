#!/usr/bin/env bash
set -euo pipefail
cd /home/keith/Git/ml/cnn-5090
source build/connect4cnn/runtime-5090.sh
validation=build/connect4cnn/validation-5090
run_check() {
    local name=$1
    shift
    printf '%q ' "$@" >> "$validation/commands.txt"
    printf '\n' >> "$validation/commands.txt"
    "$@" > "$validation/$name.log" 2>&1
    printf 'PASS: %s\n' "$name"
}
run_check build-nature bash ocean/connect4cnn/tests/build_encoder_test.sh test_nature
run_check nature .venv/bin/python ocean/connect4cnn/tests/test_nature.py --library build/connect4cnn/test_nature.so
run_check flex .venv/bin/python ocean/connect4cnn/tests/test_flex.py --library build/connect4cnn/test_nature.so
run_check build-impala bash ocean/connect4cnn/tests/build_encoder_test.sh test_impala
run_check impala .venv/bin/python ocean/connect4cnn/tests/test_impala.py --library build/connect4cnn/test_impala.so
printf 'All GPU numerical validation checks passed.\n' > "$validation/finished.txt"
