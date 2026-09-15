#!/usr/bin/env bash
# Fixed plumbing check. No search, long training or automatic W&B upload.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
if (( $# > 1 )) || [[ "${1:-}" != "" && "${1:-}" != --prepare-only ]]; then
    echo "Usage: bash ocean/pongcnn/canary.sh [--prepare-only]" >&2
    exit 2
fi
mkdir -p build/pongcnn
out=$(mktemp -d "$root/build/pongcnn/canary.XXXXXXXX")
echo "Pong canary: $out"
variants=(state flex_quality nature_cnn impala_cnn impoola_cnn)
mkdir -p "$out/source/ocean" "$out/source/config"
cp -r src "$out/source/"
cp -r ocean/pong ocean/pongcnn ocean/connect4cnn "$out/source/ocean/"
cp build.sh "$out/source/"
cp config/default.ini config/pong.ini config/pongcnn.ini "$out/source/config/"
git rev-parse HEAD > "$out/revision.txt"
git diff -- src/ocean.cu > "$out/integration.patch"
(cd "$out/source" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$out/source.sha256"
for variant in "${variants[@]}"; do
    env=pongcnn
    if [[ "$variant" == state ]]; then env=pong; fi
    mkdir -p "$out/$variant/config"
    cp "$out/source/config/default.ini" "$out/$variant/config/default.ini"
    cp "$out/source/ocean/pongcnn/compare.ini" "$out/$variant/config/$env.ini"
done
echo prepared > "$out/status.txt"
if [[ "${1:-}" == --prepare-only ]]; then exit 0; fi

source ocean/connect4cnn/runtime_env.sh
source ocean/pongcnn/gpu_env.sh
PONG_SMI=$(pong_find_smi)
export OPENBLAS_NUM_THREADS=1
if [[ -n "${NVCC_PREPEND_FLAGS:-}" ]]; then
    echo "Unset NVCC_PREPEND_FLAGS; the runner selects reference builds." >&2
    exit 2
fi
active=$("$PONG_SMI" --query-compute-apps=pid --format=csv,noheader)
if [[ -n "${active//[[:space:]]/}" ]]; then
    echo "GPU is busy; leave confirmation undisturbed and run this canary later." >&2
    exit 1
fi
trap 'echo failed > "$out/status.txt"' EXIT
run() {
    printf 'cwd=%q ' "$PWD" >> "$out/commands.txt"
    printf '%q ' "$@" >> "$out/commands.txt"
    printf '\n' >> "$out/commands.txt"
    "$@"
}
"$PONG_SMI" > "$out/gpu.txt"
for variant in "${variants[@]}"; do
    env=pongcnn
    flags=()
    case "$variant" in
        state) env=pong ;;
        nature_cnn) flags=(env NVCC_PREPEND_FLAGS=-DC4_NATURE_CNN) ;;
        impala_cnn) flags=(env NVCC_PREPEND_FLAGS=-DC4_IMPALA_CNN) ;;
        impoola_cnn) flags=(env NVCC_PREPEND_FLAGS=-DC4_IMPOOLA_CNN) ;;
    esac
    run timeout 300 "${flags[@]}" bash build.sh "$env" "$out/$variant.bin" --float \
        > "$out/build-$variant.txt" 2>&1
done
# Current source must match the archived build inputs before any training starts.
(cd "$root" && sha256sum --check "$out/source.sha256") > "$out/source-check.txt"
sha256sum "$out"/*.bin > "$out/binaries.sha256"
for variant in "${variants[@]}"; do
    env=pongcnn
    encoder=0
    if [[ "$variant" == state ]]; then env=pong; fi
    if [[ "$variant" == flex_quality ]]; then encoder=4; fi
    trial="$out/$variant"
    active=$("$PONG_SMI" --query-compute-apps=pid --format=csv,noheader)
    if [[ -n "${active//[[:space:]]/}" ]]; then
        echo "GPU became busy before $variant; stopping without interrupting it." >&2
        exit 1
    fi
    "$PONG_SMI" > "$trial/gpu-before.txt"
    common=(--headless --policy.encoder="$encoder" --base.run_id=trial
        --base.checkpoint_dir="$trial/checkpoints" --base.log_dir="$trial/metrics")
    (
        cd "$trial"
        run /usr/bin/time -f %e -o train-wall.txt timeout 120 "$out/$variant.bin" train \
            "${common[@]}" > train.txt 2>&1
        checkpoint="$trial/checkpoints/$env/trial/0000000000065536.bin"
        test -s "$checkpoint"
        sha256sum checkpoints/"$env"/trial/*.bin > checkpoints.sha256
        run timeout 120 "$out/$variant.bin" eval "${common[@]}" \
            --base.seed=29173 --base.eval_episodes=64 --base.load_model_path="$checkpoint" > eval.txt 2>&1
        printf '%s ' "$variant"
        rg '^CUDA_EVAL ' eval.txt
    ) >> "$out/results.txt"
done
echo ok > "$out/status.txt"
trap - EXIT
echo "PASS: five native train/reload jobs; receipts: $out"
