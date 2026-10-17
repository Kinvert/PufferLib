# Sprite benchmarks

Current primary-checkout results:
[cleanup compatibility/SPS](../../research/results/sprite-cleanup-5060-20261016/README.md)
and [matched local/gaze pilots](../../research/SPRITE_ATTENTION.md).
Full observations remain the normal build. Optional player/gaze research
uses a centered3x3 plus a separately selected3x3, matching one-hot state/RGB
visibility, separate X/Y gaze actions and the native shared tile CNN.
Use `env.partial_observation=1` only with the attention research binary;
see the linked protocol for exact dimensions, build and captured recipes.

Six small native games share cached tile copying and a Nature pixel encoder.
Each game owns its rules and original8x8 sprites. No PNG decoding, window,
Raylib graphics context or per-step allocation is needed for training.

| Environment | Task | State floats | Actions |
|---|---|---:|---|
|sprite_catch|Catch crystals, avoid skulls|6|0 wait,1 left,2 right|
|sprite_platform|Cross gaps, jump onto raised terrain, touch the flag|41|0 wait,1 left,2 right,3 jump-left,4 jump-right|
|sprite_maze|Branching maze, dead ends, lethal traps and gold|29|0 wait,1 up,2 down,3 left,4 right|
|sprite_runner|Collect ground/air rewards, jump over hazards|7|0 wait,1 jump|
|sprite_key|Choose the safe chest, avoid mimic, unlock exit|37|0 wait,1 up,2 down,3 left,4 right|
|sprite_push|Avoid fake crate/enemy, push real crate past obstacles|36|0 wait,1 up,2 down,3 left,4 right|

Current rules: Catch v4, Runner/Maze v3, Platform/Key/Push v2. First-release
learning results and recordings retain their frozen sources/binaries/configs;
they do not measure the changed tasks. Every game has lethal failure and reports
deaths separately from successes. See [TODO.md](TODO.md) for current validation.

Latest [scoped results](../../research/results/sprite-solid-5060-20261015/README.md)
separate state learning, the short Nature pixel pilot and execution-only smokes.
Four tasks pass all final assigned state episodes at three seeds; Runner42 is
weak and Catch42 is incomplete. The next larger matched comparison preassigns
seven fixed training replicas7/42/69/123/173/2027/9001. Never sweep seeds or
replace weaker/failed replicas. Existing packets retain their original scope.

## State and pixels

Set `env.observation_mode=0` for state or1 for pixels in the game's INI.
Pixels are640x480 RGB, flattened HWC float32 values in[0,1]. State uses the
standard PufferLib linear encoder/MinGRU. Pixels use `src/nature.cu`:
Conv8/s4/32 → Conv4/s2/64 → Conv3/s1/64 → linear(hidden), with ReLU/bias.
Weights start from random initialization. No pretrained backbone is included.
Each environment uses identical game/reward/reset functions in both modes.
Catch's calibrated recurrent core has four layers; the other five use one.
Its entropy schedule is also game-specific. These settings stay identical
between state/pixels within a game; they are not shared architecture-search axes.

`env.representation=0` selects the original artwork,1 rotates its palette,
and2 mirrors individual sprites. These choices change no game logic or RNG.
`sprite_make_tiles` prepares each requested skin once during serial init.
Four grain-texture variants are baked for every tile, including empty background
and the background behind foreground sprites. Episode/cell hashes choose the
variant without consuming world RNG. This covers Catch's night backdrop,
Platform's sky, Maze's paths, Runner's desert, and Key/Push floors. No empty
background tile is a solid color. Key/Push also have three base floor patterns.
Each environment caches a tile/texture stamp per board cell. Moves copy only
changed cached rows; repeated dirty cells and unchanged waits copy nothing.
Resets update cells whose tile or episode texture changed. No noise is generated
per pixel at step time. The viewer uploads the same pixels.
This pixel contract is `sprite-background-v2`; old recordings/checkpoints keep
their original contract. Making textures less uniform is not itself evidence
that CNN learning became harder or that an encoder generalized better.

The starting INIs use8 agents/horizon8/minibatch8 so either observation mode
can run on an8GB GPU. These are safe starting allocations, not the fast state
benchmark allocation. State/pixel policies have different weight shapes; retain
the resolved recipe alongside every checkpoint.

## Build and run

Use the existing CUDA/NCCL runtime setup and architecture for your GPU:

On G240, run from `/home/claude/cnn` and first source
`ocean/connect4cnn/runtime_env.sh`. It sets process-local paths to the existing
toolkit/NCCL libraries and installs nothing. Configured machines can build
directly; the environments have no research-runtime dependency.

The current main directories are real local copies, not links to the older
worktree. Captured research builds copy them explicitly. The ongoing cleanup
and matched local/gaze work is tracked in
[SPRITE_CLEANUP_PLAN.md](../../research/SPRITE_CLEANUP_PLAN.md). Partial views
require the separate attention research build; the normal build retains its
original movement-only actions and full observations.

```sh
NVCC_ARCH=sm_120 bash build.sh sprite_platform build/platform-puffer --float
build/platform-puffer train --base.run_id=bright-robot-7 --base.seed=7
build/platform-puffer train --env.observation_mode=1 --base.run_id=bright-robot-pixels-7
```

Fast state allocation, used in the measured learning panels:

```sh
build/platform-puffer train --env.observation_mode=0 --base.seed=7 \
  --vec.total_agents=256 --vec.num_buffers=1 --vec.num_threads=4 \
  --train.horizon=64 --train.minibatch_size=2048 \
  --train.total_timesteps=16777216 --base.run_id=steady-robot-7
```

Use this large allocation for state. It is too large for640x480 pixel rollouts
on the5060. State throughput and pixel throughput from different allocations
are descriptive upper-bound/reference measurements, not an architecture-only
comparison. Future CNN comparisons must match learner/allocations/budgets.
For Catch use33,554,432 decisions with its INI's H128/L4/entropy schedule;
the other five calibrated state recipes use16,777,216 decisions and L1.

Evaluate using the matching core/mode and training runtime config:

```sh
build/platform-puffer eval PATH_TO_WEIGHTS.bin --headless \
  --base.seed=7 --env.seed=420 --base.eval_agents=1 --base.eval_episodes=128 \
  --train.horizon=1 --train.minibatch_size=1 --vec.num_buffers=1 --vec.num_threads=1
```

For visual playback omit `--headless`; the native viewer opens the640x480 game.
Swap the environment name/binary for the other games. No research launcher is
required to train or watch an individual environment.

## Deterministic evaluation

Each environment owns an unsigned32-bit LCG. `env.seed` offsets its native slot
RNG; `base.seed` controls the learner/action RNG. Never sweep either seed.
Maze/Key/Push/Platform draw randomness only at reset. Catch and Runner advance
their world RNG once at reset and derive a separate episode-local object stream.
Lethal contact can end an episode early without shifting the next world's items.
Rendering consumes neither stream. Push's enemy is deterministic and draws no RNG. Floor textures in Key/Push
are stable episode/cell hashes. Drawings and rendering consume no world RNG.

For the stock `eval` command, use one evaluation agent to assign exactlyN sequential episodes without
first-finished batch bias. Keep the same world/action seeds, episode counts,
time limits and final-checkpoint rule across policies. Train on multiple fixed
seeds, then evaluate on different seeds. Preserve failures and every scheduled
replica. Reproducibility on one GPU/toolchain does not prove cross-device
bitwise equality.

The opt-in research evaluator assigns explicit episode IDs and reseeds world/
action RNG independently per episode. Its quota-safe waves support multiple
slots without first-finished selection, exporting every episode's metrics in
one pass. All seven sprite games/both modes pass the bounded current checks.
See [protocol and scope](../../research/SPRITE_EXACT_EVALUATOR.md); the ordinary
commands above remain the simple native training/viewing path.

Catch/Runner perf is correct decisions divided by the full assigned item budget,
including items not reached before death. It is not accuracy among only attempted
items. `successes` counts perfect fully completed episodes. Platform/Maze/Key/Push perf and successes
both measure completed episodes. Do not label item accuracy as episode win rate.
`score` and `episode_return` are actual shaped reward; they are separate from perf.
`deaths` records lethal outcomes, not every failed or timed-out episode.

## Correctness checks

```sh
bash ocean/sprite_common/test.sh build/sprite-tests-FRESH
```

This host-only ASan/UBSan suite checks state/pixel world/reward/RNG parity,
all three drawings, compact observation guards, dirty/full redraw equality,
viewer uploads and100 solvable assigned worlds per game under different
policies. It performs no CPU neural training. Oracle policies establish
solvability; they are not the learned agents and are not included in the nets.

The measured overnight development/confirmation receipts and pixel checkpoint
smokes live under `build/sprite-night-20261015/`; compact reports are maintained
under `research/results/sprite-suite-5060-20261015/`. Native C/CUDA performs the
actual learning and inference. Research Python only supervises processes and
audits/export artifacts. No CNN/Pareto/SOTA claim follows from environment tests.
