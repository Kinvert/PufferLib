# Connect4CNN

Native Connect4 copied at upstream revision `89414204`, with direct grayscale pixel observations. The standard 6×7 game, seven actions, opponent, rewards, and reset semantics are preserved. Upstream `connect4` is left intact for differential checks.

## Observation contract

- Float32, single channel, top-to-bottom row-major `[1,36,44]`, flattened to 1,584 values by the native environment API.
- Every board cell is a solid 6×6 block: empty=0, player=1, opponent=0.5.
- One black pixel column on each side. Nature's valid convolutions need width 44 to cover the seventh board column; width 42 would fit the network but discard that column.
- Pixels are written directly from the bitboard in C. No graphics capture, resizing library, window, or Raylib drawing is used to generate observations.
- The optional human viewer retains the original decorated board and reads the bitboard directly. It is not the policy's input image.

## Build and verify

Use PufferLib's normal build and its shared root-level Raylib dependency:

```bash
mkdir -p build/connect4cnn
bash build.sh connect4cnn build/connect4cnn/play --cpu
bash ocean/connect4cnn/tests/run_all.sh
build/connect4cnn/play --headless --base.eval_episodes=10
```

`build.sh` obtains the shared Raylib package if it is missing; the test runner reuses it and installs nothing. Tests use the real native environment API, ASan/UBSan, known-board image checks, and 4,096 matched steps against the original Connect4, followed by a repeated seeded trace. They do not require a display or GPU. Generated binaries/traces stay in `build/`.

The CPU runner without a model is only an environment smoke test, not a trained policy or a random-policy benchmark. Its CPU inference implementation has not been extended for the new CNN checkpoint format; use the native GPU trainer's `eval --headless` for CNN checkpoints.

## Tiny CNN and GPU training

The native trainer now selects `connect4cnn.cu` through the existing custom-encoder registration. Only five lines were added to `src/ocean.cu`; the trainer, optimizer, and build script are unchanged.

Architecture: `Conv(1→8,4×4,stride=4,valid) → ReLU → flatten NHWC (792) → Linear(hidden)`. No biases, no pixel rescaling, no activation after the projection. The existing MinGRU/core and decoder follow it. This is the first custom baseline, not Nature, IMPALA, or Impoola. At hidden=128 and one MinGRU layer it has 151,680 total parameters, of which 101,504 belong to the encoder.

The convolution uses direct patch extraction plus existing `puf_mm` GEMMs. Activation/gradient buffers are registered before execution; the encoder's forward/backward paths allocate nothing and use the supplied stream. No new floating-point atomics are introduced. Runtime repeatability still depends on the full execution setup, not just this code.

On G240, run GPU commands with tool escalation outside the sandbox. WSL's GPU utility is `/usr/lib/wsl/lib/nvidia-smi`; missing `/dev/nvidia*` is not evidence of an unavailable GPU. The helper below sets process-local paths for the existing CUDA toolkit and NCCL package. It first searches this checkout's venv, then reads the existing NCCL package in the main PufferLib clone; `NCCL_ROOT` can explicitly select another existing installation. It installs nothing and does not activate or modify another venv.

```bash
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash build.sh connect4cnn build/connect4cnn/train --float
NVCC_ARCH=sm_120 bash ocean/connect4cnn/tests/build_encoder_test.sh
.venv/bin/python ocean/connect4cnn/tests/test_encoder.py --library build/connect4cnn/test_encoder.so
bash ocean/connect4cnn/smoke.sh
```

The tested precision is float32 (`--float`). BF16 learning/performance has not been validated. `--cu` selects a GPU **environment**, not the CNN, and is not used here. The environment remains native CPU code while CNN/core learning runs on one GPU (`train.gpus=1`).

`smoke.sh` caps training at 120 seconds and evaluation at 60 seconds; it performs 65,536 training steps followed by checkpoint reload and evaluation with a separate seed. Each invocation creates a fresh `build/connect4cnn/smoke.*` directory containing source/binary hashes, resolved INI/metrics, logs, and checkpoints. The original INI recipe remains a starting point; the smoke runner overrides the budget, learning rate, batch settings, core width, and async mode explicitly.

## State versus CNN comparison

Result labels: **`state` means the original PufferLib Connect4/default network with our modified common configuration**, including hidden size 128. Its 18.51% win rate is not an untouched stock-config result. **`tiny_cnn` is our custom Conv4×4/stride4, eight-channel → ReLU → flatten → linear encoder**, feeding the same recurrent core and heads. **`stock_state` preserves the stock training configuration**, using float32 and the common evaluation setup. **`nature_cnn` is the adapted Nature encoder** described below.

The completed 13,279,232-step comparison used identical learner/core settings and paired seed lists for both policies. Relative to stock Connect4, both used hidden size 128 instead of 256, learning rate 0.001 instead of 0.00847027, replay ratio 1 instead of 3.16619, 64 agents instead of 4,096, one buffer instead of eight, minibatch 2,048 instead of 8,192, and synchronous execution. See [policy identities and the full hyperparameter comparison](../../research/EXPERIMENT_LOG.md#policy-identities-and-hyperparameters-for-comparewe7qgdcg) for the saved settings, evaluation protocol, and interpretation. The CNN has more parameters and nonlinear feature extraction; the observed learning advantage does not isolate which difference helped. The subsequent stock-config measurement below establishes a much stronger state baseline.

One entry point builds **separate native binaries** for original state-based Connect4 and the tiny pixel CNN, trains both, evaluates saved checkpoints, and writes a Markdown report plus CSV:

```bash
# Short end-to-end comparison: three seeds, four checkpoints per policy/seed.
NVCC_ARCH=sm_120 bash ocean/connect4cnn/compare.sh

# Example longer comparison; explicit budget and per-process timeout.
NVCC_ARCH=sm_120 bash ocean/connect4cnn/compare.sh \
    --steps 1048576 --seeds 73 74 75 --checkpoints 4 --eval-games 1024 --timeout 300
```

Run outside the sandbox with tool escalation. The script sources the same process-local GPU setup automatically. No packages are installed. Change common learner/core/vector settings in [compare.ini](compare.ini); the runner passes them to both binaries and checks the resulting effective INIs agree. Budget must be divisible by rollout batch × checkpoint count. Current rollout batch is 64 agents × 32 horizon = 2,048 decisions.

Each invocation prints a fresh `build/connect4cnn/compare.*` directory containing:

- `REPORT.md`: final results by seed and checkpoint learning-curve tables.
- `results.csv`: win rate, return, actual evaluation games, parameter count, steps, training/checkpoint wall time, evaluation time, whole-process SPS, native average/last SPS, and last logged VRAM.
- `recipe.ini`, `protocol.json`, `source/`, and `commands.jsonl`: common recipe, seeds, source snapshots, exact commands, revision, and source/binary hashes.
- Separate binaries, resolved native INIs, training/evaluation logs, and checkpoints. `jobs.json` records failed trials instead of silently dropping them.

Trials run serially on one GPU, alternating which policy goes first across seeds. Both use float32, the same game/opponent, learner, core width/depth, training budget, and evaluation seed lists. Evaluation uses fresh processes and happens after training at a fixed checkpoint schedule. It may overshoot the requested episode count because the native evaluator is batched; the report retains actual counts.

This compares the **state-input/default-encoder pipeline** with the **pixel-input/CNN pipeline**, with identical core/learner settings. Total parameter counts and compute are deliberately not matched: at hidden=128/core layers=1, state has 55,552 parameters versus the tiny CNN's 151,680. The report measures the combined representation/encoder cost; it cannot attribute every difference solely to pixel perception. A separately tuned comparison would answer a different question.

Training wall time includes startup and checkpoint writing, but excludes compilation and subsequent evaluation. Checkpoint wall time is approximated from file timestamps relative to training-process launch. Very short timings are dominated by startup and should not be treated as throughput rankings. Native training metrics remain available in each resolved INI.

Completed comparisons also append to [the persistent experiment log](../../research/EXPERIMENT_LOG.md). Use `--note 'what changed or what this run tests'` so later runs can be compared with their predecessors. Whole-process SPS is steps divided by measured process wall time. Native average SPS uses the trainer's final logged uptime; native last SPS is only the last sample/bin. CSV rows retain these distinct timing boundaries. VRAM is the last logged reading, not peak memory.

First stock-sized budget (rounded for the four-checkpoint schedule):

```bash
NVCC_ARCH=sm_120 bash ocean/connect4cnn/compare.sh \
    --steps 13279232 --seeds 73 74 75 --checkpoints 4 --eval-games 1024 --timeout 600 \
    --note 'First full-budget comparison; unchanged common recipe and architectures'
```

Use `--variants state tiny_cnn nature_cnn` to select the common-recipe candidates; the default remains `state tiny_cnn`. IMPALA and Impoola are not implemented yet. Selection builds separate binaries; Nature uses the process-local compiler definition `C4_NATURE_CNN`. The tiny encoder source and standard build system remain unchanged.

Verified comparison: `build/connect4cnn/compare.16ohu06z/REPORT.md`, two training seeds (73,74), 65,536 steps each, two checkpoints each, all four training jobs and eight checkpoint evaluations successful. Both policies had zero final wins at this smoke budget; no gameplay ranking follows from that check. Keep `sweep.metric=score` in the recipe: the native evaluation's score field follows that selection, while `perf` independently reports win rate.

See [the implementation plan](../../research/CONNECT4CNN_PLAN.md) for the staged learning and architecture-comparison milestones.

## Stock training configuration baseline

```bash
NVCC_ARCH=sm_120 bash ocean/connect4cnn/compare.sh --stock \
    --seeds 73 74 75 --checkpoints 4 --eval-games 1024 --timeout 600 \
    --note 'Stock Connect4 training configuration, float32; common evaluation'
```

`--stock` reads `config/default.ini` plus `config/connect4.ini`, preserves training/vec/policy/environment/selfplay settings, and checks their resolved values. It rejects custom recipes, step budgets, and variant lists. Only seed, output paths, and checkpoint/evaluation controls are overridden. The build is explicitly float32, consistent with the other tests; this is not a default-BF16 measurement. Separate evaluations use 64 agents, one buffer, two threads, synchronous execution, the same held-out seeds, and 1,024 requested games. The hidden size remains 256 to match the stock checkpoints.

Stock requests 13,272,299 decisions and completes 101 whole batches of 131,072: **13,238,272 actual decisions**. Four checkpoints occur at 3,407,872 / 6,815,744 / 10,223,616 / 13,238,272 decisions. SPS uses the actual count. These checkpoint locations differ slightly from the small-batch comparison.

Completed results: [compare.9egh6y44](../../research/results/connect4cnn/compare.9egh6y44/REPORT.md), three seeds and 12 checkpoint evaluations passed. Mean final win rate **98.87%**, mean process SPS **306,403**, mean training wall **43.414 seconds**, and **209,408 parameters**. The shared experimental recipe substantially weakened the earlier state baseline. Tiny CNN's 64.04% result does not beat this stock-config baseline; differences in tuning, capacity, and batching prevent attributing the gap solely to observation representation.

## Adapted Nature CNN

[nature.cu](nature.cu) implements valid convolutions `8×8/s4,32 → 4×4/s2,64 → 3×3/s1,64`, then flatten and a linear projection to the configured core width. Every layer has bias and ReLU. Spatial maps are 8×10 → 3×4 → 1×2. At hidden size 128, the encoder has **88,352 parameters**, and the full policy has **138,528**.

This preserves the Nature convolution stack, adapting the original four-channel 84×84 input and 512-unit projection to one-channel 36×44 and the common 128-wide recurrent interface. PufferLib's actor/critic learner, MinGRU, initialization, and optimizer are retained; this is an encoder comparison, not a DQN reproduction. Biases are registered as one-dimensional parameters and use the existing optimizer's vector update path.

The initial CUDA implementation uses preallocated im2col buffers and existing PufferLib GEMMs. Backpropagation gathers overlapping patch gradients in a fixed order and reduces bias gradients with a fixed tree, without atomic accumulation. It is a correctness baseline; throughput measurements include this implementation's costs.

```bash
NVCC_ARCH=sm_120 bash ocean/connect4cnn/tests/build_encoder_test.sh test_nature
source ocean/connect4cnn/runtime_env.sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python ocean/connect4cnn/tests/test_nature.py \
    --library build/connect4cnn/test_nature.so
NVCC_ARCH=sm_120 bash ocean/connect4cnn/compare.sh --variants nature_cnn \
    --steps 13279232 --seeds 73 74 75 --checkpoints 4 --eval-games 1024 --timeout 600 \
    --note 'First Nature comparison with unchanged common recipe'
```

Float32 CPU finite differences for all four weight/bias pairs and all-cell coverage checks passed. Actual CUDA forward/parameter-gradient comparisons passed against independent float64 NumPy results for batches 1/3/32 and hidden sizes 16/32/128, including inactive ReLUs and blank observations. Rollout/train parity and eager/graph repeated execution were exact. BF16 remains unvalidated.

Two 65,536-step Nature smoke runs (`compare.z10zc3xf`, `compare.ltxw6alr`) produced byte-identical checkpoints; all encoder weight/bias arrays updated. Final checkpoint SHA256: `1bbecd40be2327277c1852ce273720e09f026aab072b53a87f02e0d81cbfcb1b`.

Full comparison [compare.2s__8t8l](../../research/results/connect4cnn/compare.2s__8t8l/REPORT.md) completed all three training seeds and 12 checkpoint evaluations at 13,279,232 decisions per seed. Nature win rates were **77.10%, 88.33%, and 70.68%**; mean **78.71%**, mean process SPS **80,840**, mean wall **164.279 seconds**. Resolved Nature settings were checked against the earlier state/tiny run. All checkpoints were finite. Nature exceeded tiny CNN's win rate in each paired seed while running slower; both remain below the separately configured stock state baseline. The [experiment history](../../research/EXPERIMENT_LOG.md#current-measured-baselines-2026-09-12-local-time) contains the four-policy summary.

## Verified 2026-09-12

- Standard `build.sh` CPU build passed without modifications to the build system.
- Pixel fixtures, terminal/reset cases, upstream trajectory parity, and repeated-trace checks passed under ASan/UBSan.
- Standard headless CPU runner completed 10 untrained episodes in 53 steps. This checks the environment/runner path only; no learning claim follows.
- Interactive window rendering has not been exercised.
- Float32 CUDA outputs and both parameter-gradient matrices match an independent NumPy reference for batches 1/3/32 and hidden sizes 16/32/128. The reference passed finite differences. GPU train/rollout output parity and repeated eager/CUDA-graph execution passed, including blank observations and inactive ReLUs.
- Two identical seeded training smoke runs saved byte-identical final checkpoints. All 151,680 weights were finite; both CNN parameter matrices changed between the 32,768- and 65,536-step checkpoints. Their common final SHA256 is `3f009967923e06bc48ea092a8339693cc651d0dff54619eeb09b6e735b4f894e`.
- First run: `build/connect4cnn/smoke.a0cFzy`; repeat: `build/connect4cnn/smoke.EMKokT`. Checkpoint evaluation reported score −0.944056 and zero wins over 286 completed games (128 requested; batched evaluation overshoots). This short check verifies training/checkpoint plumbing.
- Longer matched comparison: `build/connect4cnn/compare.we7qgdcg`, 13,279,232 decisions per run, three seeds per policy, all six runs and 24 checkpoint evaluations successful. Mean evaluation wins: state 18.51%, tiny CNN 64.04%. Mean process SPS: state 108,831, tiny CNN 97,733. This demonstrates learning with the common recipe; parameter counts differ and convergence has not been established. See [the experiment history](../../research/EXPERIMENT_LOG.md) for full measurements and provenance.
