# Sprite Lane Dodge

This environment benchmarks learning from state versus pixels, with identical
rules and paired worlds. A single agent drives a car along a 20x15 road,
dodging oncoming red cars while collecting coins that share the lanes. It
need not reach 100% perf to be useful: reproducible learning and time curves
are the main comparison.

One agent on a 20x15 board. In `config/sprite_lanedodge.ini`, select:

| `env.observation_mode` | Observation | Native policy encoder |
|---:|---|---|
|0 (default)|9 floats|Standard PufferLib linear encoder|
|1|480x640x3 RGB, flattened to 921,600 floats|Nature CNN: 32/64/64 channels, kernels 8/4/3, strides 4/2/1, then linear|

State is a single 9-float encoding, all values in [0,1]:

- `obs[0]` = lane / 3, with lanes at columns 7, 9, 11, 13.
- `obs[1..4]` = per-lane nearest hazard distance, `(14 - row) / 14`, absent 0.
- `obs[5..8]` = per-lane nearest coin distance, same scale, absent 0.

Rows 12..0 are scanned top-down per lane; row 13 entities have already exited
or been handled, so scans stop at row 12. Neither mode includes a reward map,
shortest path or hidden information. Changing mode changes only observations
and the encoder. Spawns, rules, rewards, actions and episode endings are
shared. `env.seed` offsets the native environment-slot RNG; `base.seed`
controls the learner. Never sweep either seed. `representation` chooses one
of three skins; it does not consume RNG.

Each 32x32 cell contains 16x16 art enlarged 2x. Ground fills carry a
deterministic per-art-pixel shade, so road and grass are not flat; entity
art composites over the shaded road instead of the grass default. Each
requested skin is built once at init, before environment workers start, then
shared across environments. Unused skins are not built; the fixed cache
still reserves space for all three. Both modes show the same scene in
`puf_render`. Headless simulation needs no window, GPU graphics context or
image decoder.

## Rules

The road spans columns 7..13. Lanes are columns 7, 9, 11, 13; columns 8, 10,
12 hold white dashed dividers; row 14 is the road shoulder. The player starts
lane 1 (column 9) on row 13. Hazards and coins spawn on row 0, one of each
kind per tick at most, and advance one row per tick toward the player;
row 13 entities exit.

Actions: 0 wait, 1 one lane left, 2 one lane right (clamped at the road
edges). Each decision adds `env.step_cost` (-0.001). A hazard on the player's
cell, or one landing on the player, kills the episode
(`env.reward_death`); a coin landing on the player is collected exactly once
and removed from the board (`env.reward_coin`), and `coins_spawned` counts
every spawned coin. Episodes end on death or
`env.max_steps` (400 by default). Same-step autoreset preserves terminal
reward.

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
build/sprite_lanedodge/puffer train --env.observation_mode=0 --base.run_id=lanedodge-state
# Pixel runs inherit the 256-agent state default; override as documented above.
build/sprite_lanedodge/puffer train --env.observation_mode=1 --base.run_id=lanedodge-pixels \
  --vec.total_agents=4 --vec.num_threads=2 --train.minibatch_size=64 --train.horizon=64

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

This runs ASan/UBSan, state/pixel trajectory equivalence, world-stream
isolation, forced collision/coin semantics, timeout survival, art checks
and observation-range checks. Successful host tests/builds are not evidence
of learned quality.

Optional human play (A/D or left/right, Escape to close), without a model:

```bash
clang -std=c11 -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -O2 \
  -I. -Isrc -Iraylib-5.5_linux_amd64/include ocean/sprite_lanedodge/play.c \
  raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lm -lpthread -ldl -lrt \
  -o build/sprite_lanedodge/play
build/sprite_lanedodge/play 0
```

Pass 0..2 to choose a drawing and optionally 0/1 as a second argument to
choose state/pixels. In the normal policy viewer, hold Left Shift with
A/D or left/right to take control.

## 5060 results (2026-10-16, seed 7, holdout worlds 2027/9001, 64 episodes)

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

State is ~2,145x faster in process SPS and reaches 0.83/0.84 vs 0.41/0.43
perf. The ~90% state target is approached, not reached, at rates that keep
hazards in play (the h0c40 ceiling is 0.88/0.88 but removes the dodge
element); the ~50% pixel target is approached from below (0.42/0.43).

Historical pre-fix table (pump metric; preserved, not pooled with the above):

| recipe | wall | process SPS | 2027 | 9001 |
|---|---:|---:|---:|---:|
| state A256/H64/MB2048, 83,886,080 decisions | 83.037s | 1,010,224 | 0.969907 | 0.962675 |
| state repeat (new binary) | 82.803s | 1,013,080 | all 80 checkpoints byte-identical | |
| pixel A4/H64/MB64, 140,800 decisions (300s timeout on 16.7M budget) | 300.104s | 469.2 | 0.612785 | 0.572703 |

State training-world overfit beyond 84M at the original rates
(280M -> 0.77/0.67); the 84M point was kept.
