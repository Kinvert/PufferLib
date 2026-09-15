#!/usr/bin/env bash
# Locked native Pong train/eval matrix. Bash owns configuration and execution.
set -euo pipefail
mode=full
prepare=0
out=
while (( $# )); do
    case "$1" in
        --prepare-only) prepare=1; shift ;;
        --canary) mode=canary; shift ;;
        --full) mode=full; shift ;;
        --resume) out=$(realpath "$2"); shift 2 ;;
        *) echo 'Usage: confirm.sh [--full|--canary] [--prepare-only] [--resume RUN_DIR]' >&2; exit 2 ;;
    esac
done
if [[ -n "$out" ]]; then
    root=$(cat "$out/checkout.txt")
    mode=$(cat "$out/mode.txt")
    frozen="$out/source/ocean/pongcnn/confirm.sh"
    if [[ "$(realpath "$0")" != "$frozen" ]]; then exec bash "$frozen" --resume "$out"; fi
else
    root=$(cd "$(dirname "$0")/../.." && pwd)
fi
cd "$root"
python="$root/.venv/bin/python"
test -x "$python"
merge_ini() {
    awk '
      /^[[:space:]]*[#;]/ || /^[[:space:]]*$/ { next }
      /^\[/ { section=$0; sub(/\].*$/, "", section); sub(/^\[/, "", section)
        if(section !~ /^sweep\./ && section != "metrics" && !(section in seen)) { seen[section]=1; sections[++ns]=section }
        next }
      index($0,"=") {
        if(section ~ /^sweep\./ || section == "metrics") next
        key=$0; sub(/=.*/, "", key); gsub(/^[ \t]+|[ \t]+$/, "", key)
        value=substr($0,index($0,"=")+1); sub(/[ \t]+[#;].*$/, "", value); gsub(/^[ \t]+|[ \t]+$/, "", value)
        if(!(section in seen)) { seen[section]=1; sections[++ns]=section }
        id=section SUBSEP key
        if(!(id in values)) keys[section,++nk[section]]=key
        values[id]=value
      }
      END { for(i=1;i<=ns;i++) { s=sections[i]; print "[" s "]"; for(j=1;j<=nk[s];j++) { k=keys[s,j]; print k " = " values[s,k] }; print "" } }
    ' "$@"
}
if [[ -z "$out" ]]; then
    mkdir -p build/pongcnn
    out=$(mktemp -d "$root/build/pongcnn/confirm-5090.$mode.XXXXXXXX")
    printf '%s\n' "$root" > "$out/checkout.txt"
    printf '%s\n' "$mode" > "$out/mode.txt"
    date +%s.%N > "$out/prepared-at.txt"
    git rev-parse HEAD > "$out/revision.txt"
    git status --short > "$out/git-status.txt"
    git diff --binary HEAD > "$out/working.patch"
    mkdir -p "$out/source/ocean" "$out/source/config" "$out/source/research"
    cp -a src vendor "$out/source/"
    cp -a ocean/pongcnn ocean/connect4cnn ocean/pong "$out/source/ocean/"
    cp build.sh "$out/source/"
    cp config/default.ini config/pongcnn.ini "$out/source/config/"
    cp research/PONG_5090_HANDOFF.md "$out/source/research/"
    cp build/connect4cnn/runtime-5090.sh "$out/runtime-5090.sh"
    (cd "$out/source" && find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum) > "$out/source.sha256"
    if [[ "$mode" == canary ]]; then
        seeds=(51001); steps=65536; checkpoints=4; interval=8; games=32; train_cap=120; eval_cap=120
    else
        seeds=(31001 31002 31003 31004 31005); steps=4194304; checkpoints=8; interval=256; games=512; train_cap=1800; eval_cap=300
        # Scan actual local Pong configuration receipts, not prose or this plan.
        paths=(research/results/pongcnn)
        while IFS= read -r -d '' prior; do paths+=("$prior"); done < <(find build/pongcnn -mindepth 1 -maxdepth 1 -type d ! -path "$out" -print0)
        seed_code=0
        rg -n --glob '**/metrics/**/*.ini' '^seed[[:space:]]*=[[:space:]]*(3100[1-5]|4100[1-5])([[:space:]#;]|$)' "${paths[@]}" > "$out/seed-history.txt" || seed_code=$?
        if (( seed_code != 1 )); then echo 'Requested seed block was used or could not be checked; inspect seed-history.txt.' >&2; exit 1; fi
        echo 'No matching prior Pong seed configuration found.' >> "$out/seed-history.txt"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$steps" "$checkpoints" "$interval" "$games" "$train_cap" "$eval_cap" > "$out/budget.tsv"
    families=(flex_quality nature_cnn impala_cnn impoola_cnn)
    # Five cyclic rows of an eight-condition Latin square: every condition
    # occupies five distinct positions; every seed contains four A and four B.
    latin=(0 1 7 2 6 3 5 4)
    : > "$out/jobs.tsv"
    block=0
    for seed in "${seeds[@]}"; do
        for ((position=0; position<8; position++)); do
            condition=${latin[$(((position+block)%8))]}
            variant=${families[$((condition/2))]}
            recipe=a; if (( condition%2 )); then recipe=b; fi
            eval_seed=$((seed+10000))
            id="$recipe-$variant-s$seed"
            job="$out/$id"
            mkdir -p "$job/config"
            encoder=0
            if [[ "$variant" == flex_quality ]]; then encoder=4; fi
            if [[ "$variant" == nature_cnn ]]; then encoder=2; fi
            printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$recipe" "$variant" "$seed" "$eval_seed" >> "$out/jobs.tsv"
            {
                printf '[policy]\nencoder = %s\n[train]\ntotal_timesteps = %s\n' "$encoder" "$steps"
                printf '[base]\nseed = %s\nrun_id = trial\ncheckpoint_interval = %s\ncheckpoint_dir = %s/checkpoints\nlog_dir = %s/metrics\nresult_fd = 0\n' "$seed" "$interval" "$job" "$job"
            } > "$job/overrides.ini"
            merge_ini "$out/source/config/default.ini" "$out/source/ocean/pongcnn/compare.ini" "$out/source/ocean/pongcnn/recipe_$recipe.ini" "$job/overrides.ini" > "$job/config/default.ini"
            printf '# Frozen effective training configuration is in default.ini.\n' > "$job/config/pongcnn.ini"
            for ((c=1; c<=checkpoints; c++)); do
                printf -v checkpoint '%016d' "$((c*steps/checkpoints))"
                evaluation="$job/evaluations/$checkpoint"
                mkdir -p "$evaluation/config"
                cp "$job/config/default.ini" "$evaluation/config/default.ini"
                printf '[base]\nseed = %s\neval_episodes = %s\nload_model_path = %s/checkpoints/pongcnn/trial/%s.bin\n' "$eval_seed" "$games" "$job" "$checkpoint" > "$evaluation/config/pongcnn.ini"
            done
        done
        block=$((block+1))
    done
    "$python" "$out/source/ocean/pongcnn/confirm_report.py" "$out" prepare
fi
echo "Pong confirmation: $out"
if (( prepare )); then exit 0; fi
exec 9> "$out/runner.lock"
flock -n 9 || { echo 'This campaign already has an active runner.' >&2; exit 1; }
test ! -e "$out/fatal.txt" || { cat "$out/fatal.txt" >&2; exit 1; }
reporter="$out/source/ocean/pongcnn/confirm_report.py"
source "$out/runtime-5090.sh"
source "$out/source/ocean/pongcnn/gpu_env.sh"
PONG_SMI=$(pong_find_smi)
[[ -z "${NVCC_PREPEND_FLAGS:-}" && -z "${NVCC_EXTRA:-}" ]]
export NVCC_ARCH=sm_120 OPENBLAS_NUM_THREADS=1
read -r steps checkpoints interval games train_cap eval_cap < "$out/budget.tsv"
timed() {
    local directory=$1 cap=$2
    shift 2
    mkdir -p "$directory"
    printf 'cwd=%q ' "$PWD" > "$directory/command.txt"
    printf '%q ' "$@" >> "$directory/command.txt"
    printf '\n' >> "$directory/command.txt"
    date +%s.%N > "$directory/started.txt"
    local code=0
    /usr/bin/time -f %e -o "$directory/wall.txt" timeout -k 10 "$cap" "$@" > "$directory/output.log" 2>&1 || code=$?
    date +%s.%N > "$directory/ended.txt"
    printf '%s\n' "$code" > "$directory/exit-code.txt"
}
if [[ ! -e "$out/protocol.sha256" ]]; then
    pong_gpu_idle
    date +%s.%N > "$out/campaign-started.txt"
    "$PONG_SMI" > "$out/gpu.txt"
    lscpu > "$out/cpu.txt"
    uname -a > "$out/host.txt"
    "$CUDA_HOME/bin/nvcc" --version > "$out/cuda-compiler.txt"
    { clang --version; gcc --version; ccache --version; "$python" --version; } > "$out/compilers.txt"
    env | sort | rg '^(CUDA_HOME|NCCL_ROOT|NVCC_ARCH|OPENBLAS_NUM_THREADS|CPATH|LIBRARY_PATH|LD_LIBRARY_PATH|CCACHE_DIR)=' > "$out/runtime.txt"
    sha256sum "$NCCL_ROOT/include/nccl.h" "$NCCL_ROOT/lib/libnccl.so.2" "$CUDA_HOME/bin/nvcc" raylib-5.5_linux_amd64/lib/libraylib.a > "$out/dependencies.sha256"
    (cd "$root" && sha256sum --check "$out/source.sha256") > "$out/source-build-check.txt"
    for variant in flex_quality nature_cnn impala_cnn impoola_cnn; do
        flags=()
        case "$variant" in
            impala_cnn) flags=(env NVCC_PREPEND_FLAGS=-DC4_IMPALA_CNN) ;;
            impoola_cnn) flags=(env NVCC_PREPEND_FLAGS=-DC4_IMPOOLA_CNN) ;;
        esac
        build="$out/builds/$variant"
        if [[ -e "$build/started.txt" ]]; then
            test -e "$build/exit-code.txt" && test "$(cat "$build/exit-code.txt")" = 0
        else
            timed "$build" 300 "${flags[@]}" bash build.sh pongcnn "$out/$variant.bin" --float
            test "$(cat "$build/exit-code.txt")" = 0
        fi
    done
    (cd "$out" && sha256sum ./*.bin) > "$out/binaries.sha256"
    (cd "$root" && sha256sum --check "$out/source.sha256") >> "$out/source-build-check.txt"
    "$python" "$reporter" "$out" freeze
fi
(cd "$out" && sha256sum --check protocol.sha256 binaries.sha256) > "$out/resume-check.txt"
sha256sum --check "$out/dependencies.sha256" >> "$out/resume-check.txt"
"$python" "$reporter" "$out" verify
if [[ -e "$out/finished.json" ]]; then
    echo "Campaign already finished; preserving all completed and censored attempts: $out"
    exit 0
fi
segment=$(mktemp -d "$out/segment.XXXXXXXX")
date +%s.%N > "$segment/started.txt"
trap 'date +%s.%N > "$segment/ended.txt"' EXIT
while IFS=$'\t' read -r id recipe variant seed eval_seed; do
    job="$out/$id"
    train="$job/training"
    if [[ ! -e "$train/started.txt" ]]; then
        pong_gpu_idle
        "$PONG_SMI" > "$job/gpu-before.txt"
        (cd "$job" && timed "$train" "$train_cap" "$out/$variant.bin" train --headless)
    fi
    # A started attempt without an exit receipt is censored, never retrained.
    code=interrupted
    if [[ -f "$train/exit-code.txt" ]]; then code=$(cat "$train/exit-code.txt"); fi
    if [[ "$code" == 0 ]]; then
        if ! "$python" "$reporter" "$out" audit-job --job "$id"; then
            echo "Numerical/configuration audit failed: $id" > "$out/fatal.txt"; exit 1
        fi
        for ((c=1; c<=checkpoints; c++)); do
            printf -v checkpoint '%016d' "$((c*steps/checkpoints))"
            evaluation="$job/evaluations/$checkpoint"
            if [[ ! -e "$evaluation/started.txt" ]]; then
                pong_gpu_idle
                (cd "$evaluation" && timed "$evaluation" "$eval_cap" "$out/$variant.bin" eval --headless)
            fi
            if ! "$python" "$reporter" "$out" audit-eval --job "$id" --checkpoint "$checkpoint"; then
                echo "Evaluation numerical/configuration audit failed: $id/$checkpoint" > "$out/fatal.txt"; exit 1
            fi
        done
    elif [[ "$code" != 124 && "$code" != 137 && "$code" != interrupted ]]; then
        echo "Native training failed: $id exit=$code; investigate preserved logs." > "$out/fatal.txt"; exit 1
    fi
    "$python" "$reporter" "$out" collect
done < "$out/jobs.tsv"
date +%s.%N > "$segment/ended.txt"
date +%s.%N > "$out/campaign-ended.txt"
"$python" "$reporter" "$out" analyze
if [[ "$mode" == canary ]]; then "$python" "$reporter" "$out" require-success; fi
bash "$root/ocean/connect4cnn/package_results.sh" "$out" > "$out/package.log"
cat "$out/package.log"
echo "Finished: $out/REPORT.md"
