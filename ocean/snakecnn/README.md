# SnakeCNN and matched SnakeBench

October 5, 2026. Fifth native pixel workload, using
[a separately versioned one-agent episodic protocol](PROTOCOL.md).
`snakecnn` observes direct-memory pixels; `snakebench` observes local one-hot
state. Both share `game.h`, actions, RNG, rewards, horizon and episode starts.
Original stock Snake and its INI remain unchanged. **This is not stock-Snake
performance evidence. No neural policy has run in either new task.**

## Why a separate game protocol

Stock Snake uses global `rand()`, leaves board contents on reset, and respawns
after death without emitting the commented-out terminal. Its default is 256
agents sharing a large world. Copying that behavior would not give independent
episode RNG or reliable recurrent resets. These new tasks instead fix one
agent, local uint32 RNG, clean board/ring resets, and explicit death/horizon
terminals. Corpse and multi-agent dynamics are excluded and rejected.

Movement keeps stock's absolute actions, reversal away from the own neck,
collision with the occupied tail, food reward and maximum live length of ring
capacity minus one. Default board is 26x26 with a five-cell wall margin,
leaving a 16x16 playable interior, four foods, ring capacity 128 and a 2048
decision game horizon. These settings are an untuned starting protocol.
Spawning ranks current empty cells with portable local RNG; finite capacity
constraints guarantee room to replenish food. Initialization allocates grid
and ring once. Decisions/reset/raster use no allocations or global RNG.

Ending reward/terminal and explicit last-episode counters survive automatic
reset. Native score is ending snake length, perf clipped length/120, alongside
food count, return, duration and death/horizon counts. A horizon ending isn't
silently labeled death. It is a declared game boundary, distinct from a future
evaluator's administrative timeout. `perf` is not win rate or proof of solving.

## Pixels and local information

Both interfaces see the same centered 11x11 crop. State is float32 one-hot
11x11x8, retaining stock category IDs for empty/food/wall/own-body. Pixels are
float32 CHW 1x36x44: 3x3 tiles at x=5/y=1, empty 0, food .25, wall .5, body
.75, fixed center head 1. No length, global food direction, RNG, timestep or
whole-board channel enters either observation. Head position is already known
from the crop center. Raylib is only the optional human viewer.

Six [presets](PROTOCOL.md#equal-local-information) vary squares/disks/gaps,
inversion, food/body palette and horizontal reflection. Fixed or per-slot
mixed assignment uses the shared appearance hash; it never advances game RNG
and doesn't change after reset. Action IDs stay fixed under reflection.
All images and CNN FLOPs have the same shape. Choosing the easiest preset in
a sweep is architecture discovery, not robustness evidence.

## Verified environment groundwork

```bash
# CPU game/raster checks only. Fresh immutable output, no neural model.
bash ocean/snakecnn/tests/run_all.sh build/snakecnn/FRESH_ENVIRONMENT_ID

# Normal shared dependency/build paths, compilation only.
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash build.sh snakecnn build/snakecnn/train --float
NVCC_ARCH=sm_120 NVCC_PREPEND_FLAGS=-DC4_IMPALA_CNN \
  bash build.sh snakecnn build/snakecnn/impala --float
NVCC_ARCH=sm_120 NVCC_PREPEND_FLAGS=-DC4_IMPOOLA_CNN \
  bash build.sh snakecnn build/snakecnn/impoola --float
NVCC_ARCH=sm_120 bash build.sh snakebench build/snakecnn/state --float
```

All four sm_120 float32 targets compile; unchanged original Snake also compiles.
Numeric encoder 4 selects the same frozen quality graph used on other games;
2 selects Nature. IMPALA/Impoola use their per-build selectors and encoder 0.
All existing shared CNN kernels are reused. Core dispatch adds only SnakeCNN
to the two existing pixel allowlists. No training/CNN numerical code changed.

An independent array/deque model checks reset RNG/spawns and 49,152 decisions
per panel on three board/capacity regimes, including nearly full food capacity.
State/pixel game traces match for all six fixed and three mixed appearance
panels. Fixtures cover ring wrap, reversal, growth limit, occupied-tail/body/
wall death, death-at-horizon, rewarded horizon completion, dirty resets and
ending counters. Literal inverse-coordinate pixels, guards, overwrite,
offscreen-food isolation, global-RNG/worker-order independence, repeat and
ASan/UBSan pass. These checks don't execute a CPU neural network.

## Sweep and comparison preparation

```bash
# Writes three short trial configurations; runs no GPU or policy.
bash ocean/connect4cnn/sweep.sh --environment snakecnn \
  --recipe ocean/connect4cnn/tests/flex_kernel.ini --max-runs 3 \
  --canary --prepare-only --wandb disabled

# Current six-task panel: 24 fixed-model jobs, prepared only.
.venv/bin/python research/prepare_pixel_robustness.py \
  --out build/pixel-robustness/FRESH_FIVE_TASK_ID

# Current 38 drawing conditions: 152 fixed-model jobs, prepared only.
.venv/bin/python research/prepare_pixel_robustness.py --appearances all \
  --out build/pixel-robustness/FRESH_APPEARANCE_ID
```

Sweep recipe settings override the starting graph, as in other tasks. The
multi-task panel keeps quality frozen at 16-channel 7x7/stride-4 plus projection
64 and H128/L1. All four families receive matching within-task learners,
drawings, budgets, checkpoints and paired seeds. Snake controls stay fixed;
only supported architecture/budget/appearance keys are sweep dimensions.
Default recipes are plumbing, not calibrated learning or paper confirmation.

## Remaining gates

Both GPU holds and encoder-5's separate 5090 numerical gates remain. Build
success doesn't prove decoder/recurrent resets, checkpoint reload, GPU math or
learning. Snake's [dedicated exact evaluator](DETERMINISTIC_EVAL.md) now compiles:
explicit IDs, game/action seeds, death/horizon receipts and all-assigned quota
auditing. Eight host tests and independent game/counter/sanitizer checks pass;
starts match across drawings/families/core sizes and independent tail replay.
The supervised launcher now passes fourteen total host/glue checks, including
GPU-free existing-checkpoint preparation and retained failures. GPU quota/
reset/reload/math/repeatability acceptance remains pending. No pooled-v1 score
may stand in for it.
Then calibrate a common learner/horizon and compare full paired-seed checkpoint
curves across all drawings and tasks, preserving losses and failures.

[Retained preparation receipts](../../research/results/snakecnn/environment-20261005/README.md).
[Full fixed-graph robustness plan](../../research/MULTI_ENV_ROBUSTNESS.md).
