# Encoder memory before learner calibration

October 6, 2026. Native constructor/registration metadata and scalar arithmetic
only. **No CUDA allocation, GPU query, policy, training or dataset executed.**
Both GPU holds remain. No production encoder/trainer/kernel/optimizer changed.

## Stock-batch restriction

At stock Flappy minibatch 16,384, current native IMPALA registers
18,200,133,632 bytes (**16.950 GiB**) of encoder training tensors alone;
Impoola registers 18,139,316,224 bytes (**16.894 GiB**). One necessary
allocation already exceeds G240's nominal 8 GiB. That recipe cannot run all
four current float32 implementations on the 5060 unchanged. This is a static
required-buffer restriction, not an observed OOM or intrinsic architectural
limit of IMPALA under other implementations.

The profiler now accepts host-only `--describe` batches through 65,536,
returning before CUDA initialization/allocation. **Scheduled GPU `--run` still
caps batches at 2,048 and remains on hold/unqualified.** Independent C arithmetic
matches actual registrations at H64/128/256 and six batches: 72 comparisons,
ten native rejection cases and nine scalar artifact-hash checks pass. Current
parameter counts also match 52 retained hardware rows and Flappy's quality anchor.

## Shared smaller-batch candidate

`ocean/flappycnn/learner_stock_small_batch.ini` differs from `learner_stock.ini`
in **only minibatch_size**, changing 16,384 to 2,048 equally for every CNN.
Actor/vector/LR/clipping/horizon settings and fixed H128/L1 graphs stay the same.
A test enforces that single-field difference and paired resolved settings.

At 19,922,944 decisions, both candidates process one complete rollout pass per
epoch. Stock minibatch gives eight updates/rollout, 1,216 total; smaller
minibatch gives 64 updates/rollout, 9,728 total. Equal replay/decisions do not
make optimization trajectories identical. The existing common small-batch
recipe also gives 9,728 updates with different actor/vector/LR/clipping settings.
Retain separate complete frontiers, negative results and tuning allowances.

Registered encoder memory lower bounds at H128:

| Shared learner candidate | Quality GiB | Nature GiB | IMPALA GiB | Impoola GiB |
|---|---:|---:|---:|---:|
| Stock Flappy, minibatch 16,384 | 0.778 | 1.780 | 18.109 | 18.044 |
| Existing common recipe, minibatch 2,048 | 0.092 | 0.212 | 2.157 | 2.148 |
| Stock Flappy except minibatch 2,048 | 0.154 | 0.331 | 3.278 | 3.262 |

This sum includes encoder training payload, every actor buffer's encoder
payload, primary encoder weights/gradients and an extra encoder weight copy in
async mode. Actor batch is agents/buffers: 512 in four buffers for stock actors;
64 in one buffer for the common recipe. The reviewed single-policy native
registration path keeps these buffers live.

It excludes observations/replay, core/head tensors, recurrent states, optimizer
storage, other padding, environment buffers, CUDA graphs, vendor/context/driver
overhead. **Below eight or 32 GiB does not certify full-policy fit.** No free
memory, peak VRAM or SPS was measured. All six native pixel games reuse this
`[1,36,44]` encoder geometry; their remaining policy/game allocations differ.

## Reproduce without a GPU

Use fresh directories and existing dependencies; explicit compile architecture
avoids GPU discovery. The metadata auditor needs exactly one config per model,
so use a single-seed panel or retained canonical copies:

```bash
NVCC_ARCH=sm_120 bash ocean/connect4cnn/build_encoder_profile.sh build/FRESH_PROFILERS
cc -std=c11 -O2 -Wall -Wextra -Werror research/encoder_costs.c -o build/FRESH_COSTS
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/audit_encoder_profile.py \
  --binaries build/FRESH_PROFILERS --cost-binary build/FRESH_COSTS \
  --configs build/FRESH_SINGLE_SEED_PANEL/flappycnn/r0 \
  --out build/FRESH_METADATA --batches 1 64 512 2048 16384 65536 --host-hash

.venv/bin/python research/prepare_pixel_robustness.py \
  --environments flappycnn --appearances default --seeds 53121 53122 53123 \
  --steps 19922944 --checkpoint-steps 1048576 \
  --learner-recipe flappycnn=ocean/flappycnn/learner_stock_small_batch.ini \
  --out build/pixel-learning/FRESH_STOCK_SMALL_BATCH
```

The metadata binary is a Connect4-compiled instrument for shared fixed shapes,
not a Flappy policy run. Default descriptor audit batches remain 1/64/2,048;
invalid/duplicate batch arguments reject before output creation. Large host
descriptors don't qualify large-batch neural math or kernel index ranges.

The new retained candidate is **12 planned jobs, zero trained**: drawing 0,
three paired seeds, four frozen CNNs and 19 checkpoint steps per job. It is
recipe calibration preparation, not all-drawing learning/confirmation. Earlier
48-job all-drawing candidates remain unchanged. 35 learner/config/scalar tests
and 16 host/synthetic profiler-supervisor tests pass. [Receipts and offline
table reproduction](results/learner-memory-20261006/README.md) retain sources,
full configs, compiler/binary ledgers and the initial summary-script failure.

## Next execution gates

Qualify pending exact evaluation using saved checkpoints first. In a scheduled
window, check whole-trainer memory/math/reload for all four families with the
same selected candidate before timing. Stock minibatch is a 5090 candidate,
whose complete fit remains unverified. Stock-small-batch is a 5060/5090
candidate with unverified fit. Don't silently shrink only IMPALA's batch,
omit it after OOM or report a failed run as a successful score.

High explicit-patch memory is a baseline-backend review target, not architecture
superiority evidence. Apply later qualified backend changes fairly, renew math/
repeatability/timing and retain older-source curves. Continue shared per-game
calibration and frozen-architecture transfer across all six native games.
