#!/usr/bin/env bash
# One stock-learner pilot: quality pixels or original stock state. No search.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
environment=flappycnn
prepare_only=0
for argument in "$@"; do
    case "$argument" in
        --state) environment=flappy ;;
        --prepare-only) prepare_only=1 ;;
        *) echo 'Usage: bash ocean/flappycnn/stock_pilot.sh [--state] [--prepare-only]' >&2; exit 2 ;;
    esac
done
mkdir -p build/flappycnn
out=$(mktemp -d "$root/build/flappycnn/stock-pilot.XXXXXXXX")
echo "Flappy $environment stock pilot: $out"
printf '%s\n' "$environment" > "$out/environment.txt"
mkdir -p "$out/source/ocean" "$out/source/config" "$out/config"
cp -r src "$out/source/"
cp -r ocean/flappy ocean/flappycnn ocean/connect4cnn "$out/source/ocean/"
cp build.sh "$out/source/"
cp config/default.ini config/flappy.ini config/flappycnn.ini "$out/source/config/"
git rev-parse HEAD > "$out/revision.txt"
git diff -- src/ocean.cu src/pufferl.cu > "$out/integration.patch"
(cd "$out/source" && find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum) > "$out/source.sha256"
cp "$out/source/config/default.ini" "$out/config/default.ini"
if [[ "$environment" == flappy ]]; then
    # Original six-float observations and stock H64/L2 network, unchanged.
    cp "$out/source/config/flappy.ini" "$out/config/flappy.ini"
else
    # Keep the complete established quality network, including its H128/L1 core.
    # The stock Flappy core is H64/L2; learner and environment keys stay stock.
    awk '
        /^\[/ { section=$0 }
        section == "[policy]" && /^hidden_size[[:space:]]*=/ { print "hidden_size = 128"; next }
        section == "[policy]" && /^num_layers[[:space:]]*=/ { print "num_layers = 1"; next }
        { print }
    ' "$out/source/config/flappycnn.ini" > "$out/config/flappycnn.ini"
fi
echo prepared > "$out/status.txt"
if (( prepare_only )); then exit 0; fi

source ocean/connect4cnn/runtime_env.sh
[[ -z "${NVCC_PREPEND_FLAGS:-}" ]]
trap 'echo failed > "$out/status.txt"' EXIT
printf '%q ' bash build.sh "$environment" "$out/train" --float > "$out/build-command.txt"
timeout -k 10 300 bash build.sh "$environment" "$out/train" --float > "$out/build.log" 2>&1
(cd "$root" && sha256sum --check "$out/source.sha256") > "$out/source-check.txt"
sha256sum "$out/train" > "$out/binary.sha256"
echo built > "$out/status.txt"
nvidia_smi=$(puffer_find_nvidia_smi)
if ! puffer_require_idle_gpu "$nvidia_smi" > "$out/gpu-check.txt" 2>&1; then
    echo blocked_gpu > "$out/status.txt"
    trap - EXIT
    cat "$out/gpu-check.txt" >&2
    echo "GPU unavailable or busy; no training started. Receipts: $out" >&2
    exit 3
fi
"$nvidia_smi" > "$out/gpu.txt"
echo running > "$out/status.txt"
(
    cd "$out"
    train_args=(--headless --train.gpus=1 --base.run_id=bright-bird-1
        --base.checkpoint_dir="$out/checkpoints" --base.log_dir="$out/metrics"
        --base.checkpoint_interval=8 --base.eval_episodes=0)
    printf '%q ' "$out/train" train "${train_args[@]}" > train-command.txt
    /usr/bin/time -f %e -o train-wall.txt timeout -k 10 3600 "$out/train" train "${train_args[@]}" > train.txt 2>&1
    # Native training rounds the stock requested 20M down to whole rollouts.
    steps=$((20000000 / (2048 * 64) * (2048 * 64)))
    printf '%s\n' "$steps" > actual-steps.txt
    printf -v checkpoint '%s/checkpoints/%s/bright-bird-1/%016d.bin' "$out" "$environment" "$steps"
    test -s "$checkpoint"
    sha256sum checkpoints/"$environment"/bright-bird-1/*.bin > checkpoints.sha256
    echo trained > status.txt
    puffer_require_idle_gpu "$nvidia_smi"
    "$nvidia_smi" > gpu-before-eval.txt
    eval_args=(--headless --train.gpus=1 --base.seed=10073 --base.eval_episodes=256
        --base.eval_agents=64 --base.load_model_path="$checkpoint")
    printf '%q ' "$out/train" eval "${eval_args[@]}" > eval-command.txt
    /usr/bin/time -f %e -o eval-wall.txt timeout -k 10 300 "$out/train" eval "${eval_args[@]}" > eval.txt 2>&1
    rg "^CUDA_EVAL env=$environment " eval.txt > evaluation.txt
)
echo completed_pending_audit > "$out/status.txt"
trap - EXIT
echo "Training/evaluation exited successfully; audit metrics/checkpoints before conclusions: $out"
