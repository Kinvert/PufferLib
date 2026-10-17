#!/usr/bin/env bash
# Bounded reproducibility checks, not a quality comparison or final study.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
if (( $# > 1 )) || [[ "${1:-}" != "" && "${1:-}" != --prepare-only ]]; then
    echo "Usage: bash ocean/connect4cnn/appearance_canary.sh [--prepare-only]" >&2
    exit 2
fi
mkdir -p build/connect4cnn
out=$(mktemp -d "$root/build/connect4cnn/appearance.XXXXXXXX")
echo "Appearance canary: $out"
mkdir -p "$out/source/ocean" "$out/source/config"
cp -r src "$out/source/"
cp -r ocean/connect4 ocean/connect4cnn ocean/pong ocean/pongcnn "$out/source/ocean/"
cp build.sh "$out/source/"
cp config/default.ini config/connect4cnn.ini config/pongcnn.ini "$out/source/config/"
git rev-parse HEAD > "$out/revision.txt"
git diff --binary > "$out/working.patch"
(cd "$out/source" && find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum) > "$out/source.sha256"
variants=(flex_quality nature_cnn impala_cnn impoola_cnn)
for game in connect4cnn pongcnn; do
    for variant in "${variants[@]}"; do
        for repeat in 1 2; do
            trial="$out/$game/$variant/repeat-$repeat"
            mkdir -p "$trial/config"
            cp config/default.ini "$trial/config/default.ini"
            cp "ocean/$game/compare.ini" "$trial/config/$game.ini"
            if [[ "$game" == connect4cnn ]]; then
                sed -i '/^\[policy\]/a encoder = 4\ncnn_depth = 1\ncnn_projection = 64\ncnn_global_pool = 0\ncnn_channels_1 = 16\ncnn_kernel_1 = 7\ncnn_stride_1 = 4\ncnn_pool_1 = 0\ncnn_residual_1 = 0' "$trial/config/$game.ini"
            fi
            sed -i 's/^representation_mode = .*/representation_mode = 1/; s/^representation_seed = .*/representation_seed = 12345/' "$trial/config/$game.ini"
        done
    done
done
echo prepared > "$out/status.txt"
if [[ "${1:-}" == --prepare-only ]]; then exit 0; fi
trap 'echo failed > "$out/status.txt"' EXIT
source ocean/connect4cnn/runtime_env.sh
export OPENBLAS_NUM_THREADS=1
if [[ -n "${NVCC_PREPEND_FLAGS:-}" ]]; then
    echo 'Unset NVCC_PREPEND_FLAGS; this runner chooses reference builds.' >&2
    exit 2
fi
nvidia_smi=$(puffer_find_nvidia_smi)
puffer_require_idle_gpu "$nvidia_smi"
"$nvidia_smi" > "$out/gpu.txt"
"$CUDA_HOME/bin/nvcc" --version > "$out/compiler.txt"
uname -a > "$out/host.txt"
bash ocean/connect4cnn/tests/run_all.sh > "$out/connect4-cpu-tests.txt" 2>&1
bash ocean/pongcnn/tests/run_all.sh > "$out/pong-cpu-tests.txt" 2>&1
for game in connect4cnn pongcnn; do
    build/connect4cnn/test_appearance "$game" 12345 64 > "$out/$game/appearance.csv"
    for variant in "${variants[@]}"; do
        flags=()
        case "$variant" in
            nature_cnn) flags=(env NVCC_PREPEND_FLAGS=-DC4_NATURE_CNN) ;;
            impala_cnn) flags=(env NVCC_PREPEND_FLAGS=-DC4_IMPALA_CNN) ;;
            impoola_cnn) flags=(env NVCC_PREPEND_FLAGS=-DC4_IMPOOLA_CNN) ;;
        esac
        binary="$out/$game/$variant/train"
        printf '%q ' "${flags[@]}" bash build.sh "$game" "$binary" --float >> "$out/build-commands.txt"
        printf '\n' >> "$out/build-commands.txt"
        timeout --kill-after=5 300 "${flags[@]}" bash build.sh "$game" "$binary" --float \
            > "$out/$game/$variant/build.txt" 2>&1
    done
done
(cd "$root" && sha256sum --check "$out/source.sha256") > "$out/source-check-before.txt"
for game in connect4cnn pongcnn; do
    for variant in "${variants[@]}"; do
        encoder=0
        if [[ "$variant" == flex_quality ]]; then encoder=4; fi
        binary="$out/$game/$variant/train"
        sha256sum "$binary" >> "$out/binaries.sha256"
        for repeat in 1 2; do
            trial="$out/$game/$variant/repeat-$repeat"
            puffer_require_idle_gpu "$nvidia_smi"
            # Change only worker count between repeats to test schedule independence.
            common=(--headless --train.gpus=1 --base.seed=56173
                --policy.encoder="$encoder" --vec.num_threads="$repeat"
                --base.run_id=trial --base.checkpoint_interval=4
                --train.total_timesteps=16384
                --base.checkpoint_dir="$trial/checkpoints" --base.log_dir="$trial/metrics")
            (
                cd "$trial"
                printf '%q ' "$binary" train "${common[@]}" > train-command.txt
                /usr/bin/time -f %e -o train-wall.txt timeout --kill-after=5 60 \
                    "$binary" train "${common[@]}" > train.txt 2>&1
                checkpoint="$trial/checkpoints/$game/trial/0000000000016384.bin"
                test -s "$checkpoint"
                sha256sum checkpoints/"$game"/trial/*.bin > checkpoints.sha256
                printf '%q ' "$binary" eval "${common[@]}" --base.seed=66173 \
                    --base.eval_episodes=64 --base.load_model_path="$checkpoint" > eval-command.txt
                /usr/bin/time -f %e -o eval-wall.txt timeout --kill-after=5 30 \
                    "$binary" eval "${common[@]}" --base.seed=66173 \
                    --base.eval_episodes=64 --base.load_model_path="$checkpoint" > eval.txt 2>&1
                rg '^CUDA_EVAL ' eval.txt > eval-result.txt
                sha256sum --check checkpoints.sha256 > checkpoint-check-after-eval.txt
            )
        done
        for steps in 0000000000008192 0000000000016384; do
            cmp "$out/$game/$variant/repeat-1/checkpoints/$game/trial/$steps.bin" \
                "$out/$game/$variant/repeat-2/checkpoints/$game/trial/$steps.bin"
        done
        cmp "$out/$game/$variant/repeat-1/eval-result.txt" "$out/$game/$variant/repeat-2/eval-result.txt"
    done
done
(cd "$root" && sha256sum --check "$out/source.sha256") > "$out/source-check-after.txt"
.venv/bin/python ocean/connect4cnn/appearance_report.py "$out"
echo ok > "$out/status.txt"
trap - EXIT
echo "PASS: eight models, paired mixed-appearance training and reload; $out/REPORT.md"
