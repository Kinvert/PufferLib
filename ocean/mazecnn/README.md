# MazeCNN

October 6, 2026. Native Maze with direct-memory float32 pixels of its original
local 11x11 categorical view. Adds navigation and partial observability to the
pixel panel without introducing a renderer into policy observations. Original
`ocean/maze/maze.h` and `config/maze.ini` remain unchanged. **No neural policy
has run. Both GPU holds and encoder-5 acceptance gates remain.**

## Game and information

The adaptation copies original Maze's level generator, shared vector level
table, local `rand_r`, movement, five actions, sparse goal reward, area-based
timeout and automatic resets. Original header SHA256:
`c9992ce1536e9abb3b5a72240ce66d67554d017e3fdca979ed46edfa8ffa2900`.
Only observations, appearance assignment and rejection of unsafe configuration
inputs differ. No game RNG draw is used to choose a drawing.

The original local view is centered on the agent with radius five. MazeCNN
encodes precisely those category locations, including original zero padding
outside the 47x47 storage array. No absolute coordinate, goal direction,
map ID, timestep or hidden corridor enters the input. The frame is CHW
`1x36x44`: an 11x11 tile region with 3x3 pixels per tile, x offset 5 and y
offset 1. Empty=0, wall=.5, goal=.75 and agent=1 in preset 0. Every visible
category retains a distinct value in every preset; original state and pixels
observe the same local categories. The optional Raylib viewer shows the game
state and isn't the pixel generator.

`num_maps=8192`, `map_size=-1` preserve the original generated table. Sizes
vary from 5 to 45 with odd dimensions. Fixed sizes 5..47 are accepted; even
values normalize down as in the original. Level-generation RNG starts at 42,
each level's carving seed is its index, and custom vector slot RNGs follow
the original sequence starting at 42. **`base.seed` does not select a new level
table or reseed native worlds.** It controls the learner's separate randomness.
Different policies can consume different reset counts during training. A
dedicated exact adapter now allocates the same level/start IDs independently;
its GPU execution remains unvalidated.

Level storage is allocated once and shared by all native vector slots. Reset
copies a level; decisions/rasterization do not allocate or use global `rand()`.
The pixels add buffer work and the encoder adds compute; no speed measurement
or superiority claim follows from environment parity.

## Fixed and mixed drawings

| ID | Drawing |
|---:|---|
| 0 | Solid 3x3 category tiles |
| 1 | Five-pixel cross/disk approximation within each tile |
| 2 | 2x2 category tiles with one-pixel row/column gaps |
| 3 | Invert the complete image, including padding |
| 4 | Swap wall and goal intensities |
| 5 | Reflect local tile columns horizontally; actions stay unchanged |

`env.representation_mode=0` selects the fixed ID. Mode 1 assigns IDs 0–5 once
per native slot from `representation_seed` and slot index; resets retain it.
Native custom vector initialization uses actual slot indices, not its derived
game RNG values. Different worker/buffer orders preserve worlds and assignments.
Pong's expanded-catalog option does not apply here. All encoders see the same
fixed shape, and reducing tile area does not reduce encoder FLOPs. Keep mixed
assignment seeds separate from training seeds and report each fixed ID.

## Inherited caveats and evaluation gate

These quirks are preserved and covered by fixtures rather than silently fixed:

- Every terminal calls `puf_reset` (one level draw) and then selects a level
  again (second draw). The second level is the returned observation.
- A goal reached exactly on the `2*width*height` timeout decision is logged
  twice. Native `perf` averages the logged terminal reward entries; it is not
  an audited exact assigned-episode success rate.
- `in_bounds` uses inclusive width/height. Generated wall borders prevent
  ordinary trajectories escaping; tests cover the supported generated maps.
- The human viewer can change its `direction` field. Headless action 0 uses
  the stored direction, normally zero/stay. The adaptation preserves this.

The [dedicated exact adapter and host suite/audit](DETERMINISTIC_EVAL.md) now
compile and pass game/counter/host checks. They capture each success/timeout
once while retaining duplicate native logs separately. Preparation uses
`maze-exact-compiled-gpu-validation-pending`. Supervised checkpoint preparation
and failure/audit checks now pass without GPU use. Audited held-out training/
tuning exposure and recurrent-reset/quota/reload/repeat
GPU acceptance remain open. Do not repair training rules as part of evaluation
without declaring a separate protocol and matching state control. The original
training recipes reserve no unseen levels yet; random adjacent frames are not
held-out levels.

## Preparation and checks

```bash
# CPU game/raster/sanitizers only; fresh output, no model.
bash ocean/mazecnn/tests/run_all.sh build/mazecnn/FRESH_ENVIRONMENT_ID

# Normal compilation only, existing process-local dependencies.
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash build.sh mazecnn build/mazecnn/default --float
NVCC_ARCH=sm_120 NVCC_EXTRA=-DC4_IMPALA_CNN bash build.sh mazecnn build/mazecnn/impala --float
NVCC_ARCH=sm_120 NVCC_EXTRA=-DC4_IMPOOLA_CNN bash build.sh mazecnn build/mazecnn/impoola --float
NVCC_ARCH=sm_120 bash build.sh maze build/mazecnn/state --float

# Prepare native PROTEIN configurations only, not execution.
bash ocean/connect4cnn/sweep.sh --environment mazecnn \
  --recipe ocean/connect4cnn/tests/flex_kernel.ini --max-runs 3 \
  --canary --prepare-only --wandb disabled

# Six-task fixed-architecture packet, not training authorization.
.venv/bin/python research/prepare_pixel_robustness.py \
  --out build/pixel-robustness/FRESH_SIX_TASK_ID
```

The common recipe is untuned plumbing. It keeps the frozen quality graph
(16-channel 7x7/stride-4, projection 64, H128/L1) shared across tasks. Nature,
IMPALA and Impoola reuse existing kernels and match within-task learners,
appearances/budgets/seeds. Numeric encoder IDs are the same as other games.
Scratch initialization is independent per training run. Maze's five-action
head is task-specific, shared across encoders. Original stock state tuning is
much larger; this common recipe isn't a stock-performance reproduction.

Game checks compare 49,152 decisions per panel against an independent movement/
reset/log reference and original Maze at 5/11/47/mixed-size maps. BFS verifies
generated levels have a path to the goal. All six fixed and three mixed panels
match game traces. Literal pixels at storage corners, offscreen isolation,
dirty-buffer overwrite, shared vector levels, slot/worker-order RNG, goal-at-cap,
both reset draws, invalid settings, repeats and ASan/UBSan pass. These tests
use supplied actions and don't execute a CPU neural network.

The first test attempt caught a fixture that moved the agent without refreshing
the observation. The fixture was corrected; no assertion/tolerance was removed.
Initial failure and passing reruns are retained separately. Compilation and
configuration checks cannot validate GPU math or successful learning.

[Full robustness plan](../../research/MULTI_ENV_ROBUSTNESS.md).
[Retained environment/build/configuration evidence](../../research/results/mazecnn/environment-20261006/README.md).
