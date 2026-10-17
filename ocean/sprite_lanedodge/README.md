# Sprite Lane Dodge

This environment benchmarks learning from state versus pixels, with identical
rules and paired worlds. A single agent drives a car along a 20x15 road,
dodging oncoming red cars while collecting coins that share the lanes. It
need not reach 100% perf to be useful: reproducible learning and time curves
are the main comparison.

One agent on a 20x15 board. In `config/sprite_lanedodge.ini`, select:

| `env.observation_mode` | Observation | Native policy encoder |
|---:|---|---|
|0 (default)|21 floats|Standard PufferLib linear encoder|
|1|480x640x3 RGB, flattened to 921,600 floats|Nature CNN: 32/64/64 channels, kernels 8/4/3, strides 4/2/1, then linear|
|2|3x3 window at the gaze cell, 9 tiles x 32x32x3 + 3 geometry floats = 27,651 floats|Standard PufferLib linear encoder|
|3|Agent view: 3x3 window in which the player is the bottom-center cell plus a 3x3 window at the gaze cell, 18 tiles x 32x32x3 + 3 geometry floats = 55,299 floats|Shared tile CNN in the research build (below); standard binary uses the default linear encoder|
|4|One-hot identities for the same18 cells as mode3, plus the same3 geometry floats:111 floats|Standard PufferLib linear encoder|

The policy has two heads: a 3-way movement head (0 wait, 1 left, 2 right)
and a 300-way gaze head whose action is a direct grid cell index (Forage
mode-0 style). The gaze head is present in every observation mode; it only
affects observations in modes 2, 3 and 4, where it moves the 3x3 gaze window. An
out-of-range gaze action is ignored (the gaze keeps its previous cell).

State is a 21-float vector, all values in [0,1]:

- `obs[0]` = lane / 9, with lanes at columns 5..14 (10 adjacent lanes).
- `obs[1..10]` = per-lane nearest hazard distance, `(14 - row) / 14`, absent 0.
- `obs[11..20]` = per-lane nearest coin distance, same scale, absent 0.

Rows 13..0 are scanned top-down per lane, so objects that have reached the
player row in another lane stay visible (distance 1/14, the nearest value);
they persist one tick and can still be collided with, keeping the state
consistent with the pixels. The player's own cell is always clear at
observation time: a landing hazard kills and a landing coin is collected,
both consuming the object. Mode 2 observes the 3x3 window of
32x32 tiles centered on the gaze cell (out-of-bounds cells read as grass, the
player's cell reads as the player), plus `lane / 9`, `gx / 19`, `gy / 14`.
Mode 3 observes the 18-tile agent view: a 3x3 window (columns px-1..px+1,
rows py-2..py) in which the player is the bottom-center cell, so the window
looks ahead along the lane, followed by the 3x3 window centered on the gaze
cell, with the same out-of-bounds/player conventions and the same 3
geometry floats. Neither mode includes a reward map, shortest path or hidden
information.
Changing mode changes only observations and the encoder. Spawns, rules,
rewards, actions and episode endings are shared. `env.seed` offsets the native
environment-slot RNG; `base.seed` controls the learner. Never sweep either
seed. `representation` chooses one of three skins; it does not consume RNG.

Mode4 is the matching partial-state reference for mode3. It does not receive
the global nearest-object summaries from mode0. Both windows, movement/gaze
actions, timing and geometry are identical between modes3/4. The road's
player window deliberately looks ahead (two rows above through the player
row), rather than spending a row on the shoulder behind the vehicle.

## Mode7 2.5D render

`env.mode7` (0/1) is a headful rendering option with no effect on
observations, actions, learning or headless runs. In full-map render (modes
0 and 1) it draws a Mode7-style perspective view: a sky horizon at y = 120,
the player row filling the screen bottom (the shoulder row below it is off
camera). Row mapping `dist = 2 * (480 - y) / (y - 120)` (clamped at 13):
the bottom row shows the player row, which spans only the bottom 120
screen px (y = 360..479); the row-12 boundary is at y = 300, the row-11
boundary at y = 264, and row 0 sits above y = 165. The far band
(y = 120..165) samples the repeated top world pixel row; it is drawn as
solid sky through y = 157 and the repeated row fades in over the last
8 rows (y = 158..165), so it does not read as a vertical smear of the
lane dashes. The horizontal
sampling scale is halved, `zx = 180 / (y - 120)` world px per screen px
(clamped at 7): the bottom row shows a 320 world px span (half the
original), so the 320 px road fills the full screen width at the bottom,
and far rows show ~91 world px, shrinking the road toward the vanishing
point at (320, 120) to a ~46 px sliver (clamped). Rows fade with distance
fog `1 - 0.03 * dist` (0.61 at the farthest visible row). The headful
render is paced at 16 FPS by the window loop, and the per-frame projection
(measured 294.7 us CPU at -O2) fits inside that frame budget, so mode7
does not visibly slow the render; headless runs are unaffected. The mode-2
ROI window render is unchanged (flat 3x3 view, drawn scaled 5x at
(80, 0)); mode7 only affects the full-map branch. Mode 3 renders the agent
view instead of the full map: the 3x3 player window (the player is the
bottom-center cell) scaled 2x at (224, 0) and the 3x3 gaze window scaled 2x
at (224, 240), with out-of-range cells drawn as grass.

## Rules (sprite-lanedodge-v2)

The road spans columns 5..14. The ten lanes are the adjacent columns 5..14:
column 5 is plain asphalt and columns 6..14 carry a dashed white edge stripe
on their left edge, so every lane boundary is visible. The road/grass
boundaries (columns 4 and 15) are plain color contrast. Row 14 is the road
shoulder. The player starts lane 1 (column 6) on row 13. Hazards and coins
spawn on row 0, one of each kind per tick at most, and advance one row per
tick toward the player; row 13 entities exit.

Actions: the movement head selects 0 wait, 1 one lane left, 2 one lane right
(clamped at lanes 0 and 9). The gaze head selects a grid cell (0..299) for the
mode-2 window. Each decision adds `env.step_cost` (-0.001). A hazard on the
player's cell, or one landing on the player, kills the episode
(`env.reward_death`); a coin landing on the player is collected exactly once
and removed from the board (`env.reward_coin`), and `coins_spawned` counts
every spawned coin. Episodes end on death or `env.max_steps` (400 by
default). Same-step autoreset preserves terminal reward.

Perf is fixed independently of reward settings, clipped per episode before
averaging:

`perf = clamp(0.5 * survival + 0.5 * coins / coins_spawned, 0, 1)`

with `survival = tick / max_steps` when died and 1 otherwise. `successes`
counts survived episodes and `deaths` counts hazard episodes separately.
`score`/`episode_return` report actual reward return including step costs.
PROTEIN maximizes `perf`, not reward return.

Each reset advances the environment-slot world RNG exactly once; the episode
object schedule is a function of (seed, episode count), so episode lengths and
actions never shift a later world. Spawn draws use a separate episode-local
object RNG and never advance the world RNG. Rendering and observations never
consume RNG.

Drawings: 0 original palette, 1 swapped global palette, 2 mirrored sprites.
Drawing never advances RNG or changes collisions/spawns/rewards.

## Build and inspect

Reuse the existing runtime and toolchain. Native trainer build:

```bash
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash build.sh sprite_lanedodge build/sprite_lanedodge/puffer --float
```

```bash
# Separate run IDs/checkpoint directories: state and pixel weights are incompatible.
# The 1M-decision smoke override (the INI budget is larger):
build/sprite_lanedodge/puffer train --env.observation_mode=0 --base.run_id=lanedodge-state \
  --train.total_timesteps=1048576
# Full-pixel runs need much smaller batches than the state default; ROI (mode 2)
# is cheap (27,651 floats) and runs the standard encoder at the state default.
build/sprite_lanedodge/puffer train --env.observation_mode=1 --base.run_id=lanedodge-pixels \
  --train.total_timesteps=1048576 \
  --vec.total_agents=4 --vec.num_threads=2 --train.minibatch_size=64 --train.horizon=64
build/sprite_lanedodge/puffer train --env.observation_mode=2 --base.run_id=lanedodge-roi \
  --train.total_timesteps=1048576
# Mode 3 (agent view, 55,299 floats) is about 1/17 the full-map input and
# runs the default linear encoder at the state default in the standard
# binary; the tile CNN for mode 3 is the research build below. The 55,299-
# float view also needs encoder/learner buffers, so use the verified A32
# recipe below (same for the research binary).
build/sprite_lanedodge/puffer train --env.observation_mode=3 --base.run_id=lanedodge-view \
  --train.total_timesteps=1048576 \
  --vec.total_agents=32 --vec.num_threads=2 --train.minibatch_size=2048 --train.horizon=64
```

State training uses the tuned INI recipe (256 agents, H128/L1). Full
640x480 pixels need much smaller batches than that state default: the
earlier A64/H64/MB256 recipe needs 14.06 GiB for one observation rollout
alone (64 * 64 * 480 * 640 * 3 * 4 bytes) on an 8 GiB card, and A8 with
MB64 peaks near 8.4 GiB. Use `--vec.total_agents=4 --vec.num_threads=2
--train.minibatch_size=64 --train.horizon=64` (~6.4 GiB) for pixel runs.

Evaluate a final checkpoint with the native executable, the matching mode,
and a held-out `env.seed`:

```bash
build/sprite_lanedodge/puffer eval PATH_TO_FINAL.bin --headless \
  --env.observation_mode=0 --env.seed=2027 \
  --base.eval_agents=1 --base.eval_episodes=64 --base.async=0 \
  --vec.num_buffers=1 --vec.num_threads=1 --train.horizon=1 \
  --train.minibatch_size=1 --sweep.metric=perf
```

Game/observation correctness, without a CNN or GPU:

```bash
bash ocean/sprite_lanedodge/tests/run.sh build/lanedodge-tests-FRESH
```

This runs ASan/UBSan, state/pixel/ROI trajectory equivalence, world-stream
isolation, forced collision/coin semantics, timeout survival, ROI geometry
and slot checks, art checks and observation-range checks. Successful host
tests/builds are not evidence of learned quality.

Optional human play (A/D or left/right, Escape to close), without a model:

```bash
clang -std=c11 -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -O2 \
  -I. -Isrc -Iraylib-5.5_linux_amd64/include ocean/sprite_lanedodge/play.c \
  raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lm -lpthread -ldl -lrt \
  -o build/sprite_lanedodge/play
build/sprite_lanedodge/play 0
```

Pass 0..2 to choose a drawing and optionally 0/1 as a second argument to
choose state/pixels (the gaze is fixed at cell 0 in the manual driver). In
the normal policy viewer, hold Left Shift with A/D or left/right to take
control; the gaze follows the policy's gaze-head action.

## Tile CNN research build (mode 3)

Mode 3 learns with the shared tile CNN: the same architecture as the Sprite
Forage partial-pixel model (a constant-renamed port of
`research/forage_attention_encoder.cu`). Each 32x32 RGB tile passes through
three shared layers - 4x4/s2 3->8, 3x3/s2 8->16, then a learned784-to-16
projection with ReLU per tile - and a single linear head maps the spatially ordered
18*16 + 3 = 291 floats (288 tile features plus the 3 geometry floats) to the
H128-L1 core from `config/sprite_lanedodge.ini`, which feeds the movement and
gaze heads. The standard binary has no tile CNN: mode 3 runs the default
linear encoder there (mode 1 keeps the Nature CNN).

Build the opt-in research binary. `CORE_SOURCE` is a retained compatible core
research tree (its `src` and `ocean/connect4cnn`); this checkout contributes
the lane-dodge header and config:

```bash
research/build_lanedodge_tilecnn.sh /path/to/CORE_SOURCE build/sprite-lanedodge-tilecnn-FRESH
```

The script copies the tree, replaces `research/sprite_encoders.cu` with
`research/lanedodge_tilecnn_dispatch.cu` (observation size 55,299 -> tile
CNN; `policy.encoder=4` -> flex; otherwise Nature), applies
`research/lanedodge_tilecnn_core.patch` (core gate for the 55,299-float
observation) under `-DLD_TILE_CNN -DSPRITE_CNN_RESEARCH`, and records
`binary.sha256`, `source.sha256` and `build.log` in the fresh output
directory. Build only: no GPU query, no training.

Short training and determinism (a same-seed repeat must produce byte-identical
checkpoints):

```bash
build/sprite-lanedodge-tilecnn-FRESH/puffer train --env.observation_mode=3 \
  --base.run_id=ld-tilecnn-d1 --train.total_timesteps=1048576 \
  --vec.total_agents=32 --vec.num_threads=2 --train.minibatch_size=2048 --train.horizon=64
build/sprite-lanedodge-tilecnn-FRESH/puffer train --env.observation_mode=3 \
  --base.run_id=ld-tilecnn-d2 --train.total_timesteps=1048576 \
  --vec.total_agents=32 --vec.num_threads=2 --train.minibatch_size=2048 --train.horizon=64
sha256sum checkpoints/sprite_lanedodge/ld-tilecnn-d{1,2}/*
```

Mode-3 observation rollout memory is A x H x 55,299 x 4 bytes:3.375 GiB at
A256/H64,0.844 GiB at A64/H64,0.422 GiB at A32/H64. These figures exclude
CNN activations, optimizer and other allocations; they do not establish that
larger batches fit. The verified recipe is `--vec.total_agents=32 --vec.num_threads=2
--train.minibatch_size=2048 --train.horizon=64` (A32 x H64 =2,048, so the
MB2048 default divides evenly; reported total VRAM was about6 GiB).
The headful viewer shows the
agent view (two 3x3 windows) for mode-3 policies.

## Mode-3 tile CNN results (2026-10-16, RTX 5060)

All runs below: mode 3 (18-tile agent view), the verified A32/T2/MB2048/H64
recipe, 1,048,576 decisions, default base seed 7. 139,560 parameters
(558,240-byte checkpoint): tile layers 14,120 + head (291 to 128) 37,376 +
MinGRU core H128-L1 49,152 + decoder (303 actions + value, x 128) 38,912;
the dashboard reports 139.6K. Native trainer SPS12.7K-13.7K
(76.2-82.3s native uptime; no separate process-clock receipt),
VRAM 6.0/8G at A32.

| run | world | train terminal (perf/score) | 16-episode in-process eval (perf/score) |
|---|---|---|---|
| ld-tilecnn-d1 / d2 (same-seed repeat) | 0 | 0.695 / 61.873 | not requested |
| ld-tilecnn-eval9 | 2027 | 0.686 / 58.908 | 0.684 / 58.663 |
| ld-tilecnn-eval10 | 0 | 0.680 / 56.957 | 0.682 / 57.913 |

eval9/eval10 both show deaths 0.000, successes 1.000 and episode_length 400
in the eval. d1/d2 determinism: four checkpoints each (262144/524288/
786432/1048576) byte-identical across the repeat, identical metrics, wall
82.34 s / 78.91 s.

The standalone `puffer eval` binary could not open checkpoints written by a
separate `puffer train` process in this sandbox (file-visibility defect; see
`research/LANE_DODGE_REVIEW_LOG.md`, 2026-10-16 entries), so the 16-episode
evaluations use the in-process post-train evaluation
(`--base.eval_episodes=16 --base.eval_agents=1`), which evaluates the
just-written final checkpoint inside the training process. That evaluation
shares the training world (in-world, not held-out); eval7/eval8 (world 2027
and 0) also completed their post-train eval blocks. eval9's recipe repeated
its final metrics exactly on a same-seed re-run (ld-tilecnn-render1: 0.684 /
58.663). A world-0 re-run (eval10) reports lower dashboard scores (56.96 /
57.91) than d1/d2 (61.87) under the same recipe; treat dashboard scores as
indicative.

Headful render check passed: a debug `LD_RENDER=1` build path opens a render
pufferl (1 agent, horizon 1, env window on DISPLAY=:0, random weights) and
completed 40 rollout+render iterations of the mode-3 env in about 10.7 s
with no crash or assertion. The render draws the full map; observations use
the 18-tile agent view.

## 5060 smoke (2026-10-16, new geometry, 1M decisions, world 2027)

Short same-day smoke on the v2 geometry (seed 2027, 1M decisions, 64
evaluation episodes; not a quality claim). These runs predate the row-13
state-scan fix above and were selected on world 2027, which the pre-geometry
budget grid used for selection: 2027 and 9001 are development worlds, not
untouched holdouts, and confirmation needs new evaluation worlds.

| recipe | wall | dashboard SPS | eval perf | eval score | eval deaths/successes |
|---|---:|---:|---:|---:|---:|
| state mode 0, A256/H64 default (90,752 params) | 5.28s | 572.0K | 0.544 | 28.816 | 0.235 / 0.765 |
| ROI mode 2, A256/H64 default (3.6M params, standard encoder) | 26.02s | ~40K | 0.423 | 19.408 | 0.574 |

Both final checkpoints are finite:
`checkpoints/sprite_lanedodge/1792170526384/0000000001048576.bin` (state)
and `checkpoints/sprite_lanedodge/1792170534459/0000000001048576.bin`
(ROI). Pre-geometry weights are shape-incompatible (9 state values, 4 lanes,
1-way actions versus 21/10/2-head) and stay historical.

## 5060 results (pre-geometry v1: 4 lanes at columns 7, 9, 11, 13; historical)

The 2026-10-16 seed-7 results below were measured on the original v1
geometry (lanes at columns 7, 9, 11, 13, 9-float state, movement-only
actions). They are preserved, not pooled with the v2 numbers above.

Canonical task rates `hazard_rate=1, coin_rate=40`, re-tuned on the
consume-once coin semantics (the pre-fix defaults 20/40 were tuned while a
repeated-collection bug inflated `coin_frac`; see
`research/results/sprite-lanedodge-5060-20261016/` for the rate grid and the
historical pre-fix table).

| recipe | wall | process SPS | 2027 | 9001 |
|---|---:|---:|---:|---:|
| state A256/H64/MB2048, 83,886,080 decisions (fixed code) | 83.325s | 1,006,745 | 0.823194 | 0.836414 |
| state repeat (new binary) | 82.907s | 1,011,809 | all 80 checkpoints byte-identical | |
| pixel A4/H64/MB64, 140,800 decisions (300s timeout on 16.7M budget, fixed code) | 300.092s | 469.2 | 0.407720 | 0.427634 |

Explicit allocations: state is `--train.total_timesteps=83886080` on the
tuned recipe; pixels are `timeout 300` around
`--train.total_timesteps=16777216` (the INI budget), stopped at the
140,800-step checkpoint.
State is ~2,145x faster in process SPS and reaches 0.83/0.84 vs 0.41/0.43
perf. The ~90% state target is approached, not reached, at rates that keep
hazards in play (the h0c40 ceiling is 0.88/0.88 but removes the dodge
element); the ~50% pixel target is approached from below (0.42/0.43).

These state and pixel allocations are descriptive, not controlled: state ran
the 84M-decision tuned recipe at A256/H64/MB2048 while pixels ran the
16.7M-decision INI budget under a 300 s wall timeout at A4/H64/MB64, with
different networks; the ~2,145x throughput ratio and the perf gap are
specific to these complete systems/recipes.

Historical pre-fix table (pump metric; preserved, not pooled with the above):

| recipe | wall | process SPS | 2027 | 9001 |
|---|---:|---:|---:|---:|
| state A256/H64/MB2048, 83,886,080 decisions | 83.037s | 1,010,224 | 0.969907 | 0.962675 |
| state repeat (new binary) | 82.803s | 1,013,080 | all 80 checkpoints byte-identical | |
| pixel A4/H64/MB64, 140,800 decisions (300s timeout on 16.7M budget) | 300.104s | 469.2 | 0.612785 | 0.572703 |

State training-world overfit beyond 84M at the original rates
(280M -> 0.77/0.67); the 84M point was kept.
