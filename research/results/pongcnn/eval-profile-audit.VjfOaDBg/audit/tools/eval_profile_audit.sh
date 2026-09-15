#!/usr/bin/env bash
# Bounded diagnostic executor; native binaries only. No retries or quality sweep.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
out=$(realpath "$1")
phase=$2
source build/connect4cnn/runtime-5090.sh
source ocean/pongcnn/gpu_env.sh
PONG_SMI=$(pong_find_smi)
child= monitor=
cleanup() {
    if [[ -n "$child" ]]; then kill -TERM -- "-$child" 2>/dev/null || true; fi
    if [[ -n "$monitor" ]]; then kill "$monitor" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'echo interrupted > "$out/status.txt"; exit 124' TERM INT
activity() {
    while true; do
        date +%s.%N
        "$PONG_SMI" --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader || true
        ps -eo pid,ppid,pcpu,time,rss,stat,comm
        sleep 5
    done
}
if [[ "$phase" == validate ]]; then
    pong_gpu_idle
    date +%s > "$out/gpu-started.txt"
    "$PONG_SMI" > "$out/gpu.txt"
    for model in flex_quality nature_cnn impala_cnn impoola_cnn; do
        flags=()
        case "$model" in
            impala_cnn) flags=(env NVCC_PREPEND_FLAGS=-DC4_IMPALA_CNN);;
            impoola_cnn) flags=(env NVCC_PREPEND_FLAGS=-DC4_IMPOOLA_CNN);;
        esac
        printf '%q ' "${flags[@]}" bash build.sh pongcnn "$out/$model.bin" --float > "$out/build-$model-command.txt"
        timeout -k 2 180 "${flags[@]}" bash build.sh pongcnn "$out/$model.bin" --float > "$out/build-$model.log" 2>&1
    done
    sha256sum "$out"/*.bin > "$out/binaries.sha256"
else
    test "$phase" == campaign
    test -f "$out/validation.json"
    "$root/.venv/bin/python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' "$out/validation.json"
    sha256sum "$out/plan.json" "$out/attempts.tsv" "$out/validation.json" > "$out/frozen-panel.sha256"
fi
while IFS=$'\t' read -r group name version binary cap games; do
    if [[ "$phase" == validate && "$group" != validation ]]; then continue; fi
    if [[ "$phase" == campaign && "$group" == validation ]]; then continue; fi
    dir="$out/$name"
    test ! -e "$dir/started.txt"
    remaining=$(( $(cat "$out/gpu-started.txt") + 2700 - $(date +%s) ))
    if (( remaining <= cap + 2 )); then echo wall_cap > "$out/status.txt"; break; fi
    pong_gpu_idle
    args=("$binary" eval --headless --base.eval_episodes="$games")
    if [[ "$group" == performance ]]; then
        args=("$binary" train --headless --base.seed=52001 --train.total_timesteps=131072
              --base.run_id=trial --base.checkpoint_interval=64
              --base.checkpoint_dir="$dir/checkpoints" --base.log_dir="$dir/metrics")
        if [[ "$version" == instrumented ]]; then
            args=("$CUDA_HOME/bin/nsys" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
                  --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop
                  --kill=none --export=sqlite --output="$dir/profile" "${args[@]}" --base.profile=1)
        fi
    fi
    printf 'cd %q\n' "$dir" > "$dir/command.sh"
    printf 'timeout -k 2 %q ' "$cap" >> "$dir/command.sh"
    printf '%q ' "${args[@]}" >> "$dir/command.sh"
    printf '\n' >> "$dir/command.sh"
    sha256sum "$binary" "$dir/config/"*.ini > "$dir/inputs.sha256"
    date +%s.%N > "$dir/started.txt"
    activity > "$dir/activity.log" 2>&1 & monitor=$!
    # timeout owns the child's process group; the outer trap also cleans that group.
    (cd "$dir"; exec timeout -k 2 "$cap" "${args[@]}") > "$dir/output.log" 2>&1 & child=$!
    code=0
    wait "$child" || code=$?
    child=
    kill "$monitor" 2>/dev/null || true
    wait "$monitor" 2>/dev/null || true
    monitor=
    date +%s.%N > "$dir/ended.txt"
    awk 'NR==1{s=$1} NR==2{printf "%.6f\n",$1-s}' "$dir/started.txt" "$dir/ended.txt" > "$dir/wall.txt"
    echo "$code" > "$dir/exit-code.txt"
done < "$out/attempts.tsv"
if [[ "$phase" == validate ]]; then
    "$root/.venv/bin/python" "$out/tools/eval_profile_audit.py" "$out" validate
else
    date +%s.%N > "$out/gpu-ended.txt"
    echo completed > "$out/status.txt"
    "$root/.venv/bin/python" "$out/tools/eval_profile_audit.py" "$out" report
fi
