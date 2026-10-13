#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ocean/connect4cnn/runtime_env.sh
mkdir -p build/connect4cnn
run_dir=$(mktemp -d "$PWD/build/connect4cnn/smoke.XXXXXX")
binary="$PWD/build/connect4cnn/train"
test -x "$binary"
git rev-parse HEAD > "$run_dir/revision.txt"
git diff -- src/ocean.cu > "$run_dir/src.patch"
sha256sum "$binary" src/ocean.cu ocean/connect4cnn/connect4cnn.h \
    ocean/connect4cnn/connect4cnn.cu config/connect4cnn.ini \
    ocean/connect4cnn/smoke.sh > "$run_dir/checksums.txt"

common=(--headless --train.gpus=1 --base.gpu_offset=0
    --base.async=0 --base.cudagraphs=1
    --vec.total_agents=64 --vec.num_buffers=1 --vec.num_threads=2
    --policy.hidden_size=128 --policy.num_layers=1
    --train.horizon=32 --train.minibatch_size=2048 --train.replay_ratio=1
    --train.learning_rate=0.001 --base.run_id=smoke
    "--base.checkpoint_dir=$run_dir/checkpoints" "--base.log_dir=$run_dir/metrics")
if ! timeout 120 "$binary" train "${common[@]}" --base.seed=73 \
        --base.eval_episodes=0 --base.checkpoint_interval=16 \
        --train.total_timesteps=65536 > "$run_dir/train.log" 2>&1; then
    tail -n 30 "$run_dir/train.log" >&2
    echo "Training failed: $run_dir" >&2
    exit 1
fi
checkpoint="$run_dir/checkpoints/connect4cnn/smoke/0000000000065536.bin"
test -s "$checkpoint"
if ! timeout 60 "$binary" eval "${common[@]}" --base.seed=10073 \
        --base.eval_episodes=128 "--base.load_model_path=$checkpoint" \
        > "$run_dir/eval.log" 2>&1; then
    tail -n 30 "$run_dir/eval.log" >&2
    echo "Checkpoint evaluation failed: $run_dir" >&2
    exit 1
fi
echo "Completed training and checkpoint reload/evaluation: $run_dir"
