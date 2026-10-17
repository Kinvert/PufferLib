#!/usr/bin/env bash
# Bash preparation/launch; native C/CUDA PROTEIN and training; external reports/W&B.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
mode=sweep
variants=(flex_quality nature_cnn impala_cnn impoola_cnn)
wall_cap=3600
max_runs=24
wandb=online
project=cnn3
while (( $# )); do
    case "$1" in
        --prepare-only) mode=prepare; shift ;;
        --canary) mode=canary; shift ;;
        --variant) variants=("$2"); shift 2 ;;
        --wall-cap) wall_cap=$2; shift 2 ;;
        --max-runs) max_runs=$2; shift 2 ;;
        --wandb) wandb=$2; shift 2 ;;
        --project) project=$2; shift 2 ;;
        *) echo "Usage: $0 [--prepare-only|--canary] [--variant flex_quality|nature_cnn|impala_cnn|impoola_cnn] [--wall-cap SECONDS] [--max-runs N] [--wandb disabled|offline|online] [--project NAME]" >&2; exit 2 ;;
    esac
done
[[ "$wall_cap" =~ ^[1-9][0-9]*$ && "$max_runs" =~ ^[1-9][0-9]*$ ]]
[[ "$wandb" == disabled || "$wandb" == offline || "$wandb" == online ]]
if [[ "$mode" == canary ]]; then max_runs=2; wall_cap=180; fi
for variant in "${variants[@]}"; do
    case "$variant" in flex_quality|nature_cnn|impala_cnn|impoola_cnn) ;; *) exit 2 ;; esac
done
mkdir -p build/pongcnn
out=$(mktemp -d "$root/build/pongcnn/hypers.XXXXXXXX")
echo "Pong hyperparameter campaign: $out"
echo "$mode" > "$out/mode.txt"
git rev-parse HEAD > "$out/revision.txt"
git diff --binary > "$out/working.patch"
mkdir -p "$out/source/ocean" "$out/source/config"
cp -r src "$out/source/"
cp -r ocean/pongcnn ocean/connect4cnn "$out/source/ocean/"
cp build.sh "$out/source/"
cp config/default.ini config/pongcnn.ini "$out/source/config/"
(cd "$out/source" && find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum) > "$out/source.sha256"

# Merge known INI inputs in order, omitting all inherited search dimensions.
# Native puf_ini_load_file has the same last-value-wins merge semantics.
merge_ini() {
    awk '
        /^[[:space:]]*[#;]/ || /^[[:space:]]*$/ { next }
        /^\[/ { section=$0; sub(/\].*$/, "", section); sub(/^\[/, "", section); next }
        index($0, "=") {
            if (section ~ /^sweep\./ && FILENAME != ARGV[3]) next
            key=$0; sub(/=.*/, "", key); gsub(/^[ \t]+|[ \t]+$/, "", key)
            value=substr($0,index($0,"=")+1); sub(/[ \t]+[#;].*$/, "", value); gsub(/^[ \t]+|[ \t]+$/, "", value)
            if (!(section in seen_s)) { seen_s[section]=1; sections[++ns]=section }
            id=section SUBSEP key
            if (!(id in values)) keys[section,++nk[section]]=key
            values[id]=value
        }
        END { for (i=1;i<=ns;i++) { s=sections[i]; print "[" s "]"; for(j=1;j<=nk[s];j++) { k=keys[s,j]; print k " = " values[s,k] }; print "" } }
    ' "$@"
}
for variant in "${variants[@]}"; do
    job="$out/$variant"
    mkdir -p "$job/config" "$job/evaluations"
    echo pongcnn > "$job/environment.txt"
    echo "$variant" > "$job/variant.txt"
    encoder=0
    if [[ "$variant" == flex_quality ]]; then encoder=4; fi
    if [[ "$variant" == nature_cnn ]]; then encoder=2; fi
    {
        printf '[policy]\nencoder = %s\n[base]\ncheckpoint_dir = %s/checkpoints\nlog_dir = %s/metrics\n[sweep]\nmax_runs = %s\n' "$encoder" "$job" "$job" "$max_runs"
        if [[ "$mode" == canary ]]; then
            printf '[train]\ntotal_timesteps = 32768\n[sweep.train.total_timesteps]\nmin = 32768\nmax = 65536\n'
        fi
    } > "$job/overrides.ini"
    merge_ini "$out/source/config/default.ini" "$out/source/ocean/pongcnn/compare.ini" \
        "$out/source/ocean/pongcnn/sweep.ini" > "$job/config/default.ini"
    cp "$job/overrides.ini" "$job/config/pongcnn.ini"
    # No architecture/environment keys can enter the native search unnoticed.
    "$root/.venv/bin/python" "$out/source/ocean/pongcnn/report.py" "$job" --validate
done
echo prepared > "$out/status.txt"
if [[ "$mode" == prepare ]]; then exit 0; fi
source ocean/connect4cnn/runtime_env.sh
nvidia_smi=$(puffer_find_nvidia_smi)
export OPENBLAS_NUM_THREADS=1
[[ -z "${NVCC_PREPEND_FLAGS:-}" ]]
gpu_idle() {
    puffer_require_idle_gpu "$nvidia_smi"
}
trap 'echo failed > "$out/status.txt"' EXIT
gpu_idle
"$nvidia_smi" > "$out/gpu.txt"
for variant in "${variants[@]}"; do
    flags=()
    case "$variant" in
        impala_cnn) flags=(env NVCC_PREPEND_FLAGS=-DC4_IMPALA_CNN) ;;
        impoola_cnn) flags=(env NVCC_PREPEND_FLAGS=-DC4_IMPOOLA_CNN) ;;
    esac
    printf '%q ' "${flags[@]}" bash build.sh pongcnn "$out/$variant/train" --float > "$out/$variant/build-command.txt"
    timeout -k 10 300 "${flags[@]}" bash build.sh pongcnn "$out/$variant/train" --float > "$out/$variant/build.log" 2>&1
done
(cd "$root" && sha256sum --check "$out/source.sha256") > "$out/source-check.txt"
overall=completed
echo running > "$out/status.txt"
for variant in "${variants[@]}"; do
    job="$out/$variant"
    sha256sum "$job/train" "$job/config/"*.ini > "$job/inputs.sha256"
    gpu_idle
    "$nvidia_smi" > "$job/gpu-before.txt"
    echo running > "$job/status.txt"
    printf 'timeout -k 10 %q %q sweep --headless\n' "$wall_cap" "$job/train" > "$job/command.txt"
    code=0
    (cd "$job" && /usr/bin/time -f %e -o sweep-wall.txt timeout -k 10 "$wall_cap" \
        stdbuf -oL -eL "$job/train" sweep --headless > sweep.log 2>&1) || code=$?
    echo "$code" > "$job/exit-code.txt"
    case "$code" in
        0) echo completed > "$job/status.txt" ;;
        124) echo resource_cap > "$job/status.txt" ;;
        *) echo failed > "$job/status.txt"; exit "$code" ;;
    esac
    "$root/.venv/bin/python" "$out/source/ocean/pongcnn/report.py" "$job"
    # Reload every completed final checkpoint on a separate development seed.
    while IFS=$'\t' read -r run_id checkpoint; do
        gpu_idle
        eval_dir="$job/evaluations/$run_id"
        mkdir -p "$eval_dir/config"
        cp "$job/metrics/pongcnn/$run_id.ini" "$eval_dir/config/default.ini"
        printf '# Effective checkpoint config is in default.ini.\n' > "$eval_dir/config/pongcnn.ini"
        games=256
        if [[ "$mode" == canary ]]; then games=64; fi
        echo "$games" > "$eval_dir/requested-games.txt"
        eval_args=(--headless --base.seed=29173 --base.eval_episodes="$games" --base.result_fd=0
            --sweep.metric=score --base.load_model_path="$job/$checkpoint")
        printf '%q ' "$job/train" eval "${eval_args[@]}" > "$eval_dir/command.txt"
        eval_code=0
        (cd "$eval_dir" && /usr/bin/time -f %e -o wall.txt timeout -k 10 180 "$job/train" eval "${eval_args[@]}" > eval.txt 2>&1) || eval_code=$?
        echo "$eval_code" > "$eval_dir/exit-code.txt"
    done < "$job/trials.tsv"
    audit_args=(--require-evals)
    if [[ "$mode" == sweep ]]; then audit_args+=(--allow-eval-failures); fi
    "$root/.venv/bin/python" "$out/source/ocean/pongcnn/report.py" "$job" "${audit_args[@]}"
    if [[ -s "$job/evaluation-failures.txt" ]]; then overall=completed_with_eval_failures; fi
    # Upload between families, outside native sweep timing.
    timeout -k 10 900 "$root/.venv/bin/python" "$out/source/ocean/connect4cnn/wandb_sidecar.py" \
        "$job" --mode "$wandb" --project "$project" --entity kinvert-k > "$job/sidecar.log" 2>&1
    echo "Finished $variant: $(cat "$job/status.txt"); $job/REPORT.md"
done
echo "$overall" > "$out/status.txt"
trap - EXIT
echo "Pong campaign finished: $out"
