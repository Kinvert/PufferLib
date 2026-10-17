# Matched player/gaze sprite research

October16 follow-up: the optional packed96x192 benchmark input is described in
[SPRITE_PIXEL_BENCHMARKS.md](SPRITE_PIXEL_BENCHMARKS.md). The completed pilots
below used tile-contiguous inputs and their captured sources remain unchanged.
Do not use their weights with the new packed format without an explicitly
verified conversion. New preparation is not new training or baseline acceptance.

October16, primary checkout `/home/claude/cnn`. Catch, Platform, Maze,
Runner, Key and Push have an opt-in attention build. Their normal builds
retain full observations and original movement-only action heads.

## Observation and action contract

```ini
[env]
observation_mode = 1
partial_observation = 1
```

`observation_mode=0` selects one-hot state identities;1 selects raw RGB.
`partial_observation=1` exposes eighteen ordered cells: the centered3x3
around the player, followed by a centered3x3 at the chosen gaze location.
Overlapping windows can duplicate cells. Both modes receive the same four
geometry values: normalized player worldX/Y and signed gaze-minus-playerX/Y.
There are no hidden coordinates, nearest targets, velocity or timing fields
in the partial state path. Motion must be inferred temporally in both modes.
Full-state baselines retain their original engineered observations separately.

Three categorical action heads select original movement, absolute cellX
(20 choices) and cellY (15 choices). Looking costs the same game step; there
is no extra glimpse, gaze reward or separate simulation clock. Gaze is applied
before the normal game update and the resulting observation uses that center.
Episode reset centers gaze on the new player without consuming extra RNG.
Waiting/blocked movement still refreshes the partial observation.

Cells outside the board use a game's existing background/wall category.
State uses18*tile_types +4 floats; RGB uses18*32*32*3 +4 =55,300 floats.
Tiles come directly from the same cached atlas, palette and episode/cell
texture variants as the640x480 human/full-policy view. No full frame is built
before selecting tiles, no runtime resizing or new graphics context is needed.
The viewer remains a full-board debug view; it is not a declaration that the
partial policy sees the full board.

The partial RGB encoder shares4x4/stride2/C8 and3x3/stride2/C16 convolutions,
then a learned784-to-16 projection per tile. Ordered tile features and four
geometry values feed a learned projection to H128, then each game's existing
MinGRU depth and movement/gaze decoder. There is no global average pooling
or pretraining. Research aliases reuse the actual Forage native CUDA
implementation instead of maintaining a second copy of its math.

Changing view/action heads changes parameter count, loss entropy and the
observation contract. Never relabel ordinary one-head checkpoints as attention
policies. Full controls in the attention build still have unused gaze heads.
State is a reference; primary architecture comparisons must use identical
pixel observations, actions, per-game learners, decisions/evaluation worlds.

## Build and run

Build from current main environment/helper files and an explicit compatible
core with the ordinary full-image encoder dispatch. The frozen Forage
attention core already has a different patch and is not this builder's input.

```bash
source ocean/connect4cnn/runtime_env.sh
bash research/build_sprite_attention.sh /home/claude/cnn-audit-20261008 \
  build/sprite-attention-FRESH
```

The builder captures the supplied core and connect4cnn helpers, copies the
current six games/common helper, applies its small research dispatch patch,
and compiles six float32 native binaries. Source/binary/build receipts live
in the output. It queries/runs no GPU. Main core files are not patched.

Run from the captured runtime directory so the correct config is loaded:

```bash
cd build/sprite-attention-FRESH/runtime
../sprite_platform train --env.partial_observation=1 --env.observation_mode=1 \
  --base.seed=7 --base.eval_episodes=0 --base.run_id=happy-maple-1 \
  --vec.total_agents=32 --vec.num_threads=2 --vec.num_buffers=1 \
  --train.horizon=32 --train.minibatch_size=256 --train.total_timesteps=1048576
```

That is a short pilot recipe, not an optimized learner. Do not use full-image
pixel allocations based on this small-view memory requirement. Every larger
matched comparison uses the predetermined seven fixed training replicas and
separate evaluation worlds. Never sweep seeds or replace failed replicas.

## Current validation and allocation

Each game passes native ASan/UBSan partial/full game parity and three skins,
3072 paired steps/game, exact full-image crop parity, one-hot/geometry bounds,
gaze/reset timing and a hidden-cell mutation check. These are scalar game
tests, not CPU neural tests or proof of learned gaze.

Captured build `build/sprite-attention-20261016-v2` compiles all six targets
and a CUDA-only independent float64 oracle harness for the common18-tile
encoder. The original builder attempt is preserved under
`build/sprite-attention-20261016`: it rejected an already Forage-patched core
before native execution. The explicit compatible-core build is a fresh packet.

`run_sprite_attention.py` prepares/executes the new bounded local5060 packet
`build/sprite-attention-pilots-20261016`. Read COMPLETE/failure.json and raw
receipts for actual status; preparation/compilation alone is not success.
Allocation: ten positive/zero numerical cases at B1/3/8/64/256,24 canary
trainings (8192 decisions, paired same-seed repeats),36 exact graph/repeat/
eager evaluations (three assigned episodes/two slots), then twelve matched
1Mi-decision pilots and24 final evaluations (32 assigned episodes at each
development suite seed420/1337). Training seed7, no seed search. State/pixels
have identical per-game learner/batch/budget and visible cells, different
encoder/input sizes. GPU processes use the shared lock and idle checks;
process timings are monotonic and native timing/SPS remain separate.

The pilot's small decision budget, one independent training seed and reused
development suites cannot establish a Pareto/generalization win. Numerical
positive/zero fixtures do not qualify arbitrary ReLU branches/full learner
derivatives. Preserve losses, failures, incomplete quotas and all checkpoints.
No system/CUDA/dependency changes, CPU neural execution or upstream push.

The first execution passed the numerical checks and Catch/Platform canaries,
then its reader stopped after a successful Maze state canary with no completed
episodes (`env/perf` absent). That is a reporting failure, not a policy/native
failure. The original runner/failure/receipts remain captured. Explicit
`resume` verifies and reuses completed successful commands/configs/logs and
checkpoints, then executes only missing work; failed native processes are
never retried. Missing canary perf is now null, not fabricated zero. The
original campaign deadline remains in force, including the recovery interval.
