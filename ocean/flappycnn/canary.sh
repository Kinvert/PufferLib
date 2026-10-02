#!/usr/bin/env bash
# Matched, bounded native train/reload plumbing. No search or W&B upload.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
mode=run
case "${1:-}" in
    --prepare-only) mode=prepare ;;
    --build-only) mode=build ;;
    '') ;;
    *) echo 'Usage: bash ocean/flappycnn/canary.sh [--prepare-only|--build-only]' >&2; exit 2 ;;
esac
(( $# <= 1 ))
mkdir -p build/flappycnn
out=$(mktemp -d "$root/build/flappycnn/canary.XXXXXXXX")
echo "Flappy canary: $out"
variants=(state flex_quality nature_cnn impala_cnn impoola_cnn)
mkdir -p "$out/source/ocean" "$out/source/config"
cp -r src "$out/source/"
cp -r ocean/flappy ocean/flappycnn ocean/connect4cnn "$out/source/ocean/"
cp build.sh "$out/source/"
cp config/default.ini config/flappy.ini config/flappycnn.ini "$out/source/config/"
git rev-parse HEAD > "$out/revision.txt"
git diff -- src/ocean.cu src/pufferl.cu > "$out/integration.patch"
(cd "$out/source" && find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum) > "$out/source.sha256"

# Native last-value-wins INI merge; all inherited search sections omitted.
merge_ini() {
    awk '
        /^[[:space:]]*[#;]/ || /^[[:space:]]*$/ { next }
        /^\[/ { section=$0; sub(/\].*$/, "", section); sub(/^\[/, "", section); next }
        index($0,"=") && section !~ /^sweep\./ {
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
    env=flappycnn; encoder=0
    case "$variant" in state) env=flappy ;; flex_quality) encoder=4 ;; nature_cnn) encoder=2 ;; esac
    job="$out/$variant"
    mkdir -p "$job/config"
    merge_ini "$out/source/config/default.ini" "$out/source/ocean/flappycnn/compare.ini" > "$job/config/default.ini"
    printf '[base]\nenv_name = %s\n[policy]\nencoder = %s\n' "$env" "$encoder" > "$job/config/$env.ini"
done
"$root/.venv/bin/python" "$out/source/ocean/flappycnn/report.py" "$out" --validate-only
echo prepared > "$out/status.txt"
if [[ "$mode" == prepare ]]; then exit 0; fi

source ocean/connect4cnn/runtime_env.sh
[[ -z "${NVCC_PREPEND_FLAGS:-}" ]]
trap 'echo failed > "$out/status.txt"' EXIT
for variant in "${variants[@]}"; do
    env=flappycnn; flags=()
    case "$variant" in
        state) env=flappy ;;
        impala_cnn) flags=(env NVCC_PREPEND_FLAGS=-DC4_IMPALA_CNN) ;;
        impoola_cnn) flags=(env NVCC_PREPEND_FLAGS=-DC4_IMPOOLA_CNN) ;;
    esac
    if [[ "$variant" == nature_cnn ]]; then
        # The common native binary dispatches encoder 2 and 4 from the INI.
        cp "$out/flex_quality.bin" "$out/nature_cnn.bin"
        printf 'cp flex_quality.bin nature_cnn.bin; Nature selected by policy.encoder=2\n' > "$out/build-nature_cnn-command.txt"
        continue
    fi
    printf '%q ' "${flags[@]}" bash build.sh "$env" "$out/$variant.bin" --float > "$out/build-$variant-command.txt"
    timeout -k 10 300 "${flags[@]}" bash build.sh "$env" "$out/$variant.bin" --float > "$out/build-$variant.log" 2>&1
done
(cd "$root" && sha256sum --check "$out/source.sha256") > "$out/source-check.txt"
sha256sum "$out"/*.bin > "$out/binaries.sha256"
echo built > "$out/status.txt"
if [[ "$mode" == build ]]; then trap - EXIT; echo "Built only: $out"; exit 0; fi

nvidia_smi=$(puffer_find_nvidia_smi)
puffer_require_idle_gpu "$nvidia_smi"
"$nvidia_smi" > "$out/gpu.txt"
echo running > "$out/status.txt"
for variant in "${variants[@]}"; do
    env=flappycnn
    if [[ "$variant" == state ]]; then env=flappy; fi
    job="$out/$variant"
    puffer_require_idle_gpu "$nvidia_smi"
    "$nvidia_smi" > "$job/gpu-before.txt"
    common=(--headless --base.run_id=trial --base.eval_episodes=0
        --base.checkpoint_dir="$job/checkpoints" --base.log_dir="$job/metrics")
    (
        cd "$job"
        printf '%q ' "$out/$variant.bin" train "${common[@]}" > train-command.txt
        /usr/bin/time -f %e -o train-wall.txt timeout -k 10 120 "$out/$variant.bin" train "${common[@]}" > train.txt 2>&1
        checkpoint="$job/checkpoints/$env/trial/0000000000065536.bin"
        test -s "$checkpoint"
        sha256sum checkpoints/"$env"/trial/*.bin > checkpoints.sha256
        eval_args=(--headless --base.seed=49173 --base.eval_episodes=64
            --base.load_model_path="$checkpoint")
        printf '%q ' "$out/$variant.bin" eval "${eval_args[@]}" > eval-command.txt
        /usr/bin/time -f %e -o eval-wall.txt timeout -k 10 120 "$out/$variant.bin" eval "${eval_args[@]}" > eval.txt 2>&1
    )
done
"$root/.venv/bin/python" "$out/source/ocean/flappycnn/report.py" "$out"
echo ok > "$out/status.txt"
trap - EXIT
echo "PASS: five native train/reload jobs; receipts: $out"
