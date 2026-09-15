#!/usr/bin/env bash
# Resume this campaign's unstarted families; never rerun completed training.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
out="$root/build/pongcnn/hypers.jwyApeXS"
recovery="$out/recovery-eval-timeout"
test ! -e "$recovery"
test "$(cat "$out/flex_quality/status.txt")" = completed
test "$(cat "$out/nature_cnn/status.txt")" = completed
for variant in impala_cnn impoola_cnn; do
    test ! -e "$out/$variant/sweep.log"
    test -x "$out/$variant/train"
done
source ocean/connect4cnn/runtime_env.sh
source ocean/pongcnn/gpu_env.sh
PONG_SMI=$(pong_find_smi)
export OPENBLAS_NUM_THREADS=1
gpu_idle() {
    local active
    active=$("$PONG_SMI" --query-compute-apps=pid --format=csv,noheader) || return
    [[ -z "${active//[[:space:]]/}" ]] || { echo 'GPU busy; stopping without interrupting other work.' >&2; return 1; }
}
gpu_idle
mkdir -p "$recovery/source/ocean/pongcnn" "$recovery/source/ocean/connect4cnn"
cp ocean/pongcnn/report.py ocean/pongcnn/compare.ini ocean/pongcnn/resume_hypers.sh "$recovery/source/ocean/pongcnn/"
cp ocean/connect4cnn/wandb_sidecar.py "$recovery/source/ocean/connect4cnn/"
cp "$out/status.txt" "$recovery/original-status.txt"
cp "$out/nature_cnn/summary.json" "$out/nature_cnn/REPORT.md" "$recovery/"
cp build/pongcnn/cnn3-hypers-launch.log "$recovery/original-launch.txt"
(cd "$recovery/source" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$recovery/source.sha256"
sha256sum "$out"/*/train > "$recovery/binaries.sha256"
date -u '+%Y-%m-%d %H:%M:%S UTC' > "$recovery/started.txt"
report="$recovery/source/ocean/pongcnn/report.py"
sidecar="$recovery/source/ocean/connect4cnn/wandb_sidecar.py"
trap 'echo failed > "$out/status.txt"' EXIT
echo running_with_recorded_eval_failure > "$out/status.txt"
# Keep Nature's original 180-second timeout; no selective longer retry.
"$root/.venv/bin/python" "$report" "$out/nature_cnn" --require-evals --allow-eval-failures
timeout -k 10 900 "$root/.venv/bin/python" "$sidecar" "$out/nature_cnn" \
    --mode online --project cnn3 --entity kinvert-k > "$recovery/nature-sidecar.log" 2>&1
for variant in impala_cnn impoola_cnn; do
    job="$out/$variant"
    gpu_idle
    sha256sum "$job/train" "$job/config/"*.ini > "$job/inputs.sha256"
    "$PONG_SMI" > "$job/gpu-before.txt"
    echo running > "$job/status.txt"
    printf 'timeout -k 10 3600 %q sweep --headless\n' "$job/train" > "$job/command.txt"
    code=0
    (cd "$job" && /usr/bin/time -f %e -o sweep-wall.txt timeout -k 10 3600 \
        stdbuf -oL -eL "$job/train" sweep --headless > sweep.log 2>&1) || code=$?
    echo "$code" > "$job/exit-code.txt"
    case "$code" in
        0) echo completed > "$job/status.txt" ;;
        124) echo resource_cap > "$job/status.txt" ;;
        *) echo failed > "$job/status.txt"; exit "$code" ;;
    esac
    "$root/.venv/bin/python" "$report" "$job"
    while IFS=$'\t' read -r run_id checkpoint; do
        gpu_idle
        eval_dir="$job/evaluations/$run_id"
        mkdir -p "$eval_dir/config"
        cp "$job/metrics/pongcnn/$run_id.ini" "$eval_dir/config/default.ini"
        printf '# Effective checkpoint config is in default.ini.\n' > "$eval_dir/config/pongcnn.ini"
        echo 256 > "$eval_dir/requested-games.txt"
        args=(--headless --base.seed=29173 --base.eval_episodes=256 --base.result_fd=0
            --sweep.metric=score --base.load_model_path="$job/$checkpoint")
        printf '%q ' "$job/train" eval "${args[@]}" > "$eval_dir/command.txt"
        code=0
        (cd "$eval_dir" && /usr/bin/time -f %e -o wall.txt timeout -k 10 180 "$job/train" eval "${args[@]}" > eval.txt 2>&1) || code=$?
        echo "$code" > "$eval_dir/exit-code.txt"
    done < "$job/trials.tsv"
    "$root/.venv/bin/python" "$report" "$job" --require-evals --allow-eval-failures
    timeout -k 10 900 "$root/.venv/bin/python" "$sidecar" "$job" \
        --mode online --project cnn3 --entity kinvert-k > "$job/sidecar.log" 2>&1
    echo "Finished $variant; retained evaluation failures, if any, are in evaluation-failures.txt"
done
echo completed_with_eval_failures > "$out/status.txt"
date -u '+%Y-%m-%d %H:%M:%S UTC' > "$recovery/finished.txt"
trap - EXIT
