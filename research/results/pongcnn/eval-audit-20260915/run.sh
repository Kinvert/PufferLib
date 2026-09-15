#!/usr/bin/env bash
set -euo pipefail
cd /home/claude/cnn
source ocean/connect4cnn/runtime_env.sh
export OPENBLAS_NUM_THREADS=1
gpu=$(puffer_find_nvidia_smi)
out=$PWD/build/pongcnn/eval-audit-20260915/verified
old=$PWD/build/pongcnn/hypers.jwyApeXS/flex_quality
sha256sum "$old/train" "$out/train" > "$out/binaries.sha256"
git diff -- src/pufferl.cu > "$out/evaluation.patch"
for run_id in sweep_1789435894587_0004 sweep_1789435919496_0006; do
    checkpoint=$(awk -F '\t' -v id="$run_id" '$1 == id { print $2 }' "$old/trials.tsv")
    test -n "$checkpoint"
    checkpoint="$old/$checkpoint"
    sha256sum "$checkpoint" >> "$out/checkpoints.sha256"
    for version in old new; do
        job="$out/$run_id/$version"
        mkdir -p "$job/config"
        cp "$old/evaluations/$run_id/config/"*.ini "$job/config/"
        binary="$old/train"
        if [[ "$version" == new ]]; then binary="$out/train"; fi
        puffer_require_idle_gpu "$gpu"
        (
            cd "$job"
            /usr/bin/time -f %e -o wall.txt timeout -k 5 30 "$binary" eval --headless \
                --base.seed=29173 --base.eval_episodes=256 --base.result_fd=0 \
                --sweep.metric=score --base.load_model_path="$checkpoint" > eval.txt 2>&1
        )
        rg '^CUDA_EVAL ' "$job/eval.txt" > "$job/result.txt"
    done
    cmp "$out/$run_id/old/result.txt" "$out/$run_id/new/result.txt"
done
sha256sum --check "$out/checkpoints.sha256"
# Deliberately unfinished diagnostic verifies progress survives external timeout.
job="$out/progress"
mkdir -p "$job/config"
cp "$old/evaluations/$run_id/config/"*.ini "$job/config/"
puffer_require_idle_gpu "$gpu"
code=0
(
    cd "$job"
    timeout -k 5 7 "$out/train" eval --headless --base.seed=29173 \
        --base.eval_episodes=100000000 --base.result_fd=0 --sweep.metric=score \
        --base.load_model_path="$checkpoint" > eval.txt 2>&1
) || code=$?
echo "$code" > "$job/exit-code.txt"
test "$code" = 124
rg '^CUDA_EVAL_PROGRESS ' "$job/eval.txt"
if rg -q '^CUDA_EVAL ' "$job/eval.txt"; then exit 1; fi
echo 'PASS: two original/patched saved-checkpoint evaluations match exactly; timeout progress retained.'
