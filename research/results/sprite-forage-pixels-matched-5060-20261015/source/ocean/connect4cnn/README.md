# Connect4CNN

Native Connect4 copied at upstream revision `89414204`, with direct grayscale pixel observations. The standard 6×7 game, seven actions, opponent, rewards, and reset semantics are preserved. Upstream `connect4` is left intact for differential checks.

## Observation contract

- Float32, single channel, top-to-bottom row-major `[1,36,44]`, flattened to 1,584 values by the native environment API.
- With default `env.representation=0`, every board cell is a solid 6×6 block: empty=0, player=1, opponent=0.5. Other presets are documented in [REPRESENTATIONS.md](REPRESENTATIONS.md).
- The default has one black pixel column on each side. Nature's valid convolutions need width 44 to cover the seventh board column; width 42 would fit the network but discard that column.
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

Use `--variants state tiny_cnn nature_cnn impala_cnn impoola_cnn` to select common-recipe candidates; the default remains `state tiny_cnn`. Selection builds separate binaries with process-local encoder definitions. Existing encoder sources and the standard build system remain unchanged when adding a candidate.

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

## IMPALA and Impoola

[impala.cu](impala.cu) implements one shared backbone for both candidates. Each of three stages, with channels **16/32/32**, has a 3×3 convolution, 3×3 stride-two max pooling, and two preactivation residual blocks. Each residual block is `ReLU → Conv3×3 → ReLU → Conv3×3 → add skip`, with no extra activation after the addition. All convolutions have bias. After the final stage, apply ReLU, the selected readout, a linear projection to hidden size, and ReLU.

This follows the [original IMPALA torso](https://github.com/google-deepmind/scalable_agent/blob/master/experiment.py#L129). Convolutions use SAME padding; max pooling uses the original TensorFlow SAME alignment, including trailing-edge-only padding for even dimensions. Negative infinity represents padded pool entries, and equal maxima choose the first valid position in row-major order. For our image, pooled maps are 18×22 → 9×11 → 5×6.

`impala_cnn` flattens the final 5×6×32 map to 960 features. `impoola_cnn` applies global average pooling to the same post-ReLU map, giving 32 features, following the [Impoola paper's readout change](https://arxiv.org/html/2503.05546v2#S4). This is a controlled Impoola variant using the same original-SAME backbone; no separate padding or width changes accompany GAP. The input is grayscale 36×44 and the projection is 128 rather than the reference's 256. These are adapted encoders within PufferLib's existing recurrent actor/critic learner, not reproductions of the papers' complete agents or benchmark scores.

Reference cross-check: the [Impoola author implementation at commit 1d71110451cc](https://github.com/raphajaner/impoola/blob/1d71110451cc03390d8cf6911b570995d17d94cb/impoola/train/nn.py) likewise places activation before GAP, but uses symmetric `MaxPool2d(..., padding=1)`. Our pair deliberately keeps original IMPALA SAME alignment, which differs at even image sizes. Both native variants use PufferLib Kaiming initialization with gain √2 and zero biases, rather than that file's default PyTorch layer initialization. A reproduction of the author's exact implementation would need those differences handled as explicit additional variants.

| Candidate at hidden 128 | Convolution parameters | Projection parameters | Encoder parameters | Full policy parameters |
|---|---:|---:|---:|---:|
| IMPALA | 97,312 | 123,008 | 220,320 | 270,496 |
| Impoola | 97,312 | 4,224 | 101,536 | 151,712 |

Derived forward weight MACs per observation: **11,493,120 IMPALA**, **11,374,336 Impoola**, versus **647,168 Nature** at this input/projection size. One MAC is one multiplication plus accumulation; these counts exclude bias, activation, pooling, backward, optimizer, and core/head operations. GAP removes many projection parameters but only about 1% of this pair's encoder MACs. Neither these counts nor parameter totals alone establish wall-clock speed.

Convolution scratch is shared and regenerated during backward passes. Activations needed for residual gradients are retained. Overlapping convolution/pooling gradients use fixed-order gathers, and bias gradients use fixed reduction trees. This keeps the unchanged 2,048-decision training minibatch within the RTX 5060's memory: smoke runs logged about 3.5 GB total GPU memory. The first implementation is a correctness baseline; timing includes scratch traffic and kernel overhead, so its SPS is not a hardware-independent property of IMPALA or Impoola.

```bash
NVCC_ARCH=sm_120 bash ocean/connect4cnn/tests/build_encoder_test.sh test_impala
source ocean/connect4cnn/runtime_env.sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python ocean/connect4cnn/tests/test_impala.py \
    --library build/connect4cnn/test_impala.so
NVCC_ARCH=sm_120 bash ocean/connect4cnn/compare.sh --variants impala_cnn impoola_cnn \
    --steps 13279232 --seeds 73 74 75 --checkpoints 4 --eval-games 1024 --timeout 2400 \
    --note 'IMPALA/Impoola: original-SAME backbone, flatten versus GAP, common recipe'
```

Float32 finite differences passed for all 16 weight/bias pairs in each variant, as did all-cell coverage checks. GPU forward and every parameter gradient matched independent NumPy results for batches 1/3/32 and hidden sizes 16/32/128. Pooling tests covered even/odd borders, negative inputs, equal maxima, and overlapping backward contributions. Rollout/train outputs and repeated eager/CUDA-graph execution matched exactly. BF16 is not validated.

Training smoke runs `compare.ij5_xpo7` and `compare.sqngvlom` each completed both variants at 65,536 decisions, with independent checkpoint evaluation. Matching seeds produced byte-identical checkpoints, all parameters were finite, and every encoder weight/bias array updated. Final SHA256: IMPALA `0e6e0b2835d0e39dfeb50891d46733d858b61ad67b9d2a60343dc9301832e1a0`; Impoola `b1251292b649a88d1e72307ee1c6c05b80f578f9bd3605eb44794037d69f4015`. Both had zero evaluation wins at the smoke budget and about 9,800–10,000 process SPS.

The [full-budget comparison](../../research/results/connect4cnn/compare.l6d5sbk2/REPORT.md) completed all six trials and 24 evaluations on September 13. At 13,279,232 decisions per seed, IMPALA averaged **99.19% held-out wins and 10,800 process SPS**; Impoola averaged **69.49% and 10,833 SPS**. Mean training times were 1,229.612 and 1,225.833 seconds. All checkpoints were finite, and resolved learner/core settings matched the earlier Nature baseline. See the [experiment history](../../research/EXPERIMENT_LOG.md) for the six-policy table, per-seed variation, and separately configured stock baseline.

## Experimental CNN native sweeps

### Flexible stage controls

New experimental family: `policy.encoder=5` adds four stages, dilation, residual repetitions, adaptive spatial readout, and fixed/learned activation choices. [sweep_flex2.ini](sweep_flex2.ini) describes the broad grammar; the first bounded search panels are [depth one](sweep_flex2_fast.ini) and [stage two](sweep_flex2_stage2.ini). Follow [the 5090 handoff](../../NEXT_5090_FLEX2_SWEEP.md) and [expanded-grammar instructions](../../research/FLEX2_CNN_SWEEP.md). Native float32 builds previously passed; GPU math/training/reload and old-kernel regressions remain pending on the 5090. Prepare without GPU access using `bash ocean/connect4cnn/sweep.sh --recipe ocean/connect4cnn/sweep_flex2_fast.ini --max-runs 3 --canary --prepare-only --wandb disabled`. No new learning result is implied by these recipes. Dataset pretraining remains [design only](../../research/PIXEL_PRETRAINING_DATASET_PLAN.md).

`policy.encoder=4` selects [flex.cu](flex.cu), with independent per-stage channels, kernels, strides, residual switches, and pooling; depth 1–3, flatten/GAP, and projection widths 16–128. All shape settings live in [sweep_flex.ini](sweep_flex.ini). It defaults to 128 trials but is not automatically launched. The runner now accepts any nonempty subset of a family's sweep sections: omit an option's section to fix its `[policy]` value, or omit the timestep sweep to fix training duration. Existing IDs and checkpoint meanings are preserved. Full ranges, operator semantics, and validation scope: [flexible search controls](../../research/FLEX_CNN_SWEEP.md).

For a bounded canary, use `NVCC_ARCH=sm_120 OPENBLAS_NUM_THREADS=1 bash ocean/connect4cnn/sweep.sh --recipe ocean/connect4cnn/sweep_flex.ini --canary --max-runs 12 --wandb disabled`. The one-knob example `tests/flex_kernel.ini` sweeps kernel size while fixing every other option and the budget. New reports retain every active option in `architecture_json`; inactive stages do not change the architecture fingerprint.

### Compact search with a Nature control

Run `NVCC_ARCH=sm_120 OPENBLAS_NUM_THREADS=1 bash ocean/connect4cnn/sweep_small.sh` detached for 12 Nature budget trials followed by 24 compact architecture/budget trials. Both use the same native training/PROTEIN path, online W&B project `kinvert-k/puffer-cnn`, fixed learner/core, and 829,952–6,639,616 decision range. Histories retain 25 points, checkpoints every 500 updates plus final. The 300-second predicted-cost ceiling is not a hard worker limit; each campaign has a four-hour hard deadline. Do not actively monitor it.

Numeric `policy.encoder=2` selects the existing adapted Nature exactly; `3` selects the compact family. Compact supports `cnn_depth=1/2/3`, `cnn_channels=8/16/32`, `cnn_stride=2/4` for the first convolution, and `cnn_projection=32/64/128`. The recurrent core stays at 128 units: smaller projections add a final linear mapping into it. There are 54 configurations. ID 0 retains the compiled default and ID 1 retains the residual family. Each campaign fixes its family; Nature sweeps only budget. Full rationale and limitations: [compact search plan](../../research/COMPACT_CNN_SWEEP.md).

Both paths share the validated kernels in `nature.cu`; Nature's operations and parameter layout are preserved. The Nature-equivalent compact setting (32 channels, 3 layers, stride 4, projection 128) matches its outputs/gradients bit-for-bit. All 54 shapes passed float32/float64 reference and eager/graph checks; Nature's old and new training checkpoints matched exactly. Run `build_encoder_test.sh test_nature`, then `test_nature.py` and `test_compact.py` against `build/connect4cnn/test_nature.so`. Two 12-trial canaries passed (`sweep.9sl0k5ll`, `sweep.lvfycy5l`); compact checkpoint reload passed. Use `sweep.sh --recipe ocean/connect4cnn/sweep_compact.ini --canary --max-runs 12 --wandb disabled` for another bounded plumbing check.

### Original residual family

`policy.encoder=1` selects [cnn.cu](cnn.cu); `0` preserves the existing compiled reference/default encoder. This family has three SAME Conv3/MaxPool3-s2 stages, channels `[c,2c,2c]`, zero to two preactivation residual blocks per stage, and flatten or GAP before Linear/ReLU to the existing core width. `cnn_channels` accepts 8/16/32, `cnn_blocks` accepts 0/1/2, and `cnn_global_pool` accepts 0/1. Input remains the same 36×44 pixel board. Core width/layers are separate settings. All weights start fresh and train jointly.

The actual optimizer, trial selection/launch, CNN, and training are native C/CUDA. `sweep.py` is **temporary configuration/reporting scaffolding**. Replacing that glue with native tooling is a delivery requirement. It does not implement its own search algorithm. The external Python W&B sidecar stays outside training. Nature now has a control campaign; other reference-family sweeps and randomized environments remain future work.

```bash
# Run outside the sandbox, after checking that the GPU is available.
NVCC_ARCH=sm_120 bash ocean/connect4cnn/sweep.sh

# No W&B SDK invocation; still save JSON sidecar payloads and local results.
NVCC_ARCH=sm_120 bash ocean/connect4cnn/sweep.sh --wandb disabled

# Explicit online logging, using your existing W&B authentication.
NVCC_ARCH=sm_120 bash ocean/connect4cnn/sweep.sh --wandb online --project puffer-cnn
```

Defaults: 12 trials, 600-second hard deadline for the entire sweep process group, float32, one GPU, offline W&B. The first candidate is the base INI configuration; PROTEIN then explores before making a model-guided proposal. These short budgets test plumbing, not useful learning or a speed ranking.

For the first learning-scale discovery campaign:

```bash
NVCC_ARCH=sm_120 bash ocean/connect4cnn/sweep.sh \
    --recipe ocean/connect4cnn/sweep_discovery.ini --timeout 43200 \
    --wandb online --project puffer-cnn --entity kinvert-k
```

This recipe uses 24 trials, 3,319,808–13,279,232 requested decisions, a 6,639,616-decision initial candidate, and the same 18 shapes and fixed learner/core. Checkpoints are saved every 1,000 updates plus the final update. PROTEIN's predicted-cost ceiling is 3,600 seconds; this is not a per-trial hard timeout. The command gives the entire sweep a 12-hour hard deadline, not an expected duration. Launch long campaigns detached with their stdout saved, and let them run without active monitoring. Confirm finalists later with held-out evaluation and repeated seeds.

The runner merges the original defaults, Connect4CNN config, common `compare.ini`, and [sweep.ini](sweep.ini). It removes inherited sweep sections and retains only the four ranges declared in `sweep.ini`: channels, blocks, pooling, and timesteps. It writes an isolated `build/connect4cnn/sweep.*/config/` and runs one freshly built binary from that directory. **The checkout's `config/default.ini` is never rewritten.** Use `--recipe PATH`, `--max-runs N`, and `--timeout SECONDS` for a bounded custom campaign. The first version deliberately validates this limited shape space.

Every campaign saves its source snapshots/hashes, binary hash, effective configuration, GPU record, native sweep log, trial INIs, checkpoints, CSV, Markdown report, completion status, and sidecar JSON. The architecture fingerprint includes the construction version, observation shape, channels/blocks/readout, and core dimensions. Sidecar payloads also hash the final checkpoint. Load checkpoints with the corresponding campaign config; the native checkpoint itself is still a raw FP32 array.

Native `sweep.metric=perf` is **training win rate**, not the held-out evaluation used in the reference reports. Final score/cost/steps come from PROTEIN stdout (four score decimals, two cost decimals), while INI histories are downsampled means. W&B preserves native names: `SPS`, `uptime`, `agent_steps`, `env/perf`, `env/score`, `loss/*`, and the other recorded metrics, with `agent_steps` as the plot axis. Native cost excludes trainer setup and graph-capture time. Reported trial average SPS is actual decisions divided by that rounded native cost; the `SPS` curve retains the trainer's recorded values. Whole sweep wall time additionally includes worker startup and search. Log-sampled budgets are truncated by native integer/batch handling, including possible lower-bound floating-point rounding; report actual completed decisions rather than the requested budget.

The current native observation loop overwrites earlier points sharing the same parameter vector; the optimizer effectively retains the final observation. Five-point summaries remain in artifacts. This implementation preserves that existing PROTEIN behavior.

### W&B sidecar

Display names use stable adjective/noun/trial names such as `happy-cat-1`; internal native IDs remain in metadata and filenames for provenance. Schema 2 updates existing W&B runs in place, restores native metric names, hides the legacy `binned/*` metrics from automatic plots, and removes their obsolete summary entries. Successful migration is recorded locally so repeated syncs do not append the same history again. Historical archive payloads retain their original schema.

The sidecar follows the F-Zero/Admiral saved-INI pattern. It joins complete native logs with final PROTEIN output, creates one W&B run per completed trial, and tracks successful ingestion locally so rerunning it does not re-upload completed runs to the same destination. Online mode automatically runs the CPU sidecar concurrently and uploads each completed trial; that CPU/I/O activity may affect timing slightly. Offline/disabled modes ingest after training. The follower exits after `finished.json` appears. It does not stream per-step training metrics that the native trainer does not write. Results appear as grouped runs in the project's Runs view, not as a W&B-managed sweep: PROTEIN owns the search. Online upload was verified on September 13 at https://wandb.ai/kinvert-k/puffer-cnn using the 12 completed canary trials.

```bash
.venv/bin/python ocean/connect4cnn/wandb_sidecar.py build/connect4cnn/sweep.RUN \
    --mode offline --project puffer-cnn
```

`--mode online` and optional `--entity` select an explicit remote destination. Offline mode stores W&B files locally, following the [W&B SDK mode contract](https://docs.wandb.ai/models/ref/python/functions/init). The project Python 3.12 dependency lock now includes W&B 0.21.4; install/sync it with `uv` into this checkout's `.venv`. No Torch or system CUDA changes are needed. W&B requires protobuf 6.x, so the lock uses protobuf 6.33.6; LanceDB and FastEmbed remain at their previous pinned versions.

### Validation

```bash
NVCC_ARCH=sm_120 bash ocean/connect4cnn/tests/build_encoder_test.sh test_cnn
source ocean/connect4cnn/runtime_env.sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python ocean/connect4cnn/tests/test_cnn.py \
    --library build/connect4cnn/test_cnn.so
.venv/bin/python ocean/connect4cnn/tests/test_sweep_tools.py
```

All 18 shapes passed native float32 forward/all-gradient comparisons, rollout/train equality, repeated eager/graph execution, and allocation alignment checks. The smallest-width variants also passed parameter finite differences. CPU tool tests cover inherited-sweep removal, source-config preservation, final versus binned metrics, partial results, missing checkpoints, and repeat sidecar ingestion. BF16 remains unvalidated.

## Verified 2026-09-12

- Standard `build.sh` CPU build passed without modifications to the build system.
- Pixel fixtures, terminal/reset cases, upstream trajectory parity, and repeated-trace checks passed under ASan/UBSan.
- Standard headless CPU runner completed 10 untrained episodes in 53 steps. This checks the environment/runner path only; no learning claim follows.
- Interactive window rendering has not been exercised.
- Float32 CUDA outputs and both parameter-gradient matrices match an independent NumPy reference for batches 1/3/32 and hidden sizes 16/32/128. The reference passed finite differences. GPU train/rollout output parity and repeated eager/CUDA-graph execution passed, including blank observations and inactive ReLUs.
- Two identical seeded training smoke runs saved byte-identical final checkpoints. All 151,680 weights were finite; both CNN parameter matrices changed between the 32,768- and 65,536-step checkpoints. Their common final SHA256 is `3f009967923e06bc48ea092a8339693cc651d0dff54619eeb09b6e735b4f894e`.
- First run: `build/connect4cnn/smoke.a0cFzy`; repeat: `build/connect4cnn/smoke.EMKokT`. Checkpoint evaluation reported score −0.944056 and zero wins over 286 completed games (128 requested; batched evaluation overshoots). This short check verifies training/checkpoint plumbing.
- Longer matched comparison: `build/connect4cnn/compare.we7qgdcg`, 13,279,232 decisions per run, three seeds per policy, all six runs and 24 checkpoint evaluations successful. Mean evaluation wins: state 18.51%, tiny CNN 64.04%. Mean process SPS: state 108,831, tiny CNN 97,733. This demonstrates learning with the common recipe; parameter counts differ and convergence has not been established. See [the experiment history](../../research/EXPERIMENT_LOG.md) for full measurements and provenance.
## Pixel representation selector

`[env] representation = 0` preserves the original image. Integer IDs 0–9 select filled squares, gaps, disks, X/O marks and smaller centered boards down to one pixel per cell, all within the same 36×44 tensor. See [REPRESENTATIONS.md](REPRESENTATIONS.md) for the exact mapping, native PROTEIN recipe and robustness caveats. `sweep_representation.ini` fixes the CNN and sweeps only appearance; adding its `[sweep.env.representation]` section to another recipe enables joint search. Reports retain representation separately from network identity.
