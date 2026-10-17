# Matched sprite CNN benchmarks and architecture discovery

October16. Primary checkout `/home/claude/cnn`. This is temporary research
infrastructure; the eventual PufferLib PR remains a separate minimal change.
The executable overnight development supervisor is documented in
[NEXT_5090_SPRITE_BENCHMARKS.md](../NEXT_5090_SPRITE_BENCHMARKS.md). Its own
captured recipes/receipts supersede preparation-only status for that new
allocation only; prior packets stay unchanged.
Completed sprite cleanup/pilots are committed at `837ba6f6`. Their reports
describe their original tile-contiguous observation format, not this new one.

## Identical input; independently learned actions

All candidates receive one192-high by96-wide RGB image, row-major NHWC:

```text
          96 pixels
    +-----------------+
    | player 3x3      | 96 pixels
    +-----------------+
    | chosen gaze 3x3 | 96 pixels
    +-----------------+
```

Each tile is still32x32; no downsampling, hidden object labels or full-map
image is added. Four geometry values follow the55296 pixel values: player
worldX/Y and signed gaze-minus-playerX/Y. Every family gets the same55300
floats. The join between windows is artificial and identical for all models.
Tile boundaries are an architectural prior for our tile encoder; do not claim
this benchmark alone proves unstructured-camera/image generalization.

Movement and gaze are **learned**, not prerecorded or frozen. Every policy
starts from random weights and chooses movement, gazeX20 and gazeY15 each
step. Match available actions, per-game learners, decision schedule, skins
and assigned evaluation worlds; do not force policies to take identical
actions. Different policies naturally visit different training trajectories.

The six shared-helper games are Catch, Platform, Maze, Runner, Key and Push.
Forage and Lane retain their earlier distinct layouts/actions. They are
excluded from this first packet rather than silently relabeled as equivalent.
Future adapters must preserve their rules and explicitly version new inputs.

## Native encoder choices

| `[policy] encoder` | Family | Packed-input behavior |
|---|---|---|
|6|Our existing tile CNN|Extract the18 tiles on GPU, shared4/s2/C8 and3/s2/C16, learned16 features/tile, ordered features+geometry to common core width|
|2|Nature|Valid8/s4/C32,4/s2/C64,3/s1/C64; flatten and project|
|0|IMPALA|Three stages C16/32/32,3x3 convolution,3x3 maxpool/s2, two residual blocks/stage; flatten and project|
|7|Impoola|Same IMPALA stack with global average pooling readout|
|4|Configurable CNN|Existing18-coordinate native Flex grammar on the entire packed RGB image|

Nature/IMPALA/Impoola/Flex append the same four geometry values after their
image projection, then apply a shared-form learned linear/ReLU adapter into
the recurrent core. Report that adapter and its parameters; these are RL
encoder baselines with a common core, not complete original DQN/IMPALA agents.
The tile encoder already has its geometry projection and gets no extra one.

Native Nature/IMPALA/Flex descriptors now use their actual image dimensions
and RGB channel count. Legacy36x44 grayscale contracts stay enforced outside
the new research build. Legacy native descriptor/scalar regressions must
continue passing; old compiled evidence is never overwritten or relabeled.

## Build and prepare without GPU work

Use an existing uv Python3.12 venv and existing CUDA/NCCL/raylib dependencies.
No installation, Torch change, system change or GPU discovery is necessary.
The builder captures the committed core plus current named environment/encoder
inputs. It uses no external research worktree, leaves main core untouched,
and records the actual copied sources and binaries. Fresh output required.

```bash
source ocean/connect4cnn/runtime_env.sh
bash research/test_sprite_packed.sh build/packed-raster-FRESH
bash research/build_sprite_benchmarks.sh build/packed-build-FRESH
.venv/bin/python research/sprite_pixel_benchmarks.py prepare \
  --build build/packed-build-FRESH --out build/packed-canary-FRESH
.venv/bin/python research/sprite_pixel_benchmarks.py inspect build/packed-canary-FRESH
bash research/build_sprite_attention_checks.sh build/packed-build-FRESH sprite_catch packed
```

`prepare`/`inspect` launch only native host registration/raster-manifest modes;
no CUDA context/query, tensor computation or policy training. Every model has
the same starting world/pixel hashes for each game/skin/evaluation assignment.
`checks` is an independent float64 CUDA tile-encoder test **compiled only**;
building it does not constitute a numerical pass for any family.

Default canary preparation allocates30 training jobs,8192 decisions/job,
eight checkpoints, seed7 and three assigned episodes per skin/world.
It executes none. The current common A32/H32/MB32 batch is deliberately
conservative and needs GPU calibration. Per-game LR/entropy/gamma/core depth
come from captured game INIs and are identical across encoder candidates.
This differs from the earlier MB256 pilot; don't splice old scores into it.

For later calibration, explicitly prepare a fresh packet with e.g.
`--mode calibration --decisions 1048576`. `--mode confirmation` allocates all
seven fixed training replicas7/42/69/123/173/2027/9001 for EVERY candidate.
Neither training nor evaluation seeds are sweep axes. Reserved evaluation
world suites2026101601/2026101602 differ from the prior development suites.
Preparation manifests disclose initial frames only, not learned scores.

The packet retains exact configs, binary/source hashes, native parameter
counts and encoder activation/gradient estimates. Estimates exclude the
complete learner/core/optimizer/concurrency footprint; they are not a claim
that a batch fits. No cross-hardware timing comparison follows from them.

## Sweep preparation

`research/recipes/sprite_pixel_architecture.ini` exposes the existing18 CNN
coordinates: depth, projection, GAP and each stage's channels/kernel/stride/
pool/residual. Preparation removes EVERY inherited default sweep axis first.
No seed, LR, clipping, core, observation, action or appearance coordinate varies.

Each Flex job additionally contains `sweep/config/default.ini` (empty) and a
complete `sweep/config/sprite_GAME.ini`. Running its native binary in that
`sweep` directory with mode`sweep` uses those CNN-only axes. These are separate
single-game discovery inputs, not part of the fixed benchmark allocation.
Those individual-game sweep files remain preparation only. The new
`sprite_overnight.py` supplies actual all-six-sprite aggregate feedback to
the native PROTEIN bridge, using its own sprite/evaluator bindings. A single-
game discovery winner is not a shared architecture winner.

## Next execution gates

1. Actual packed-input GPU numerical checks, including baseline spatial/RGB
   descriptors, geometry adapter forward/backward, maxpool/GAP and learner
   batches. Positive/zero tile fixtures alone do not qualify these paths.
2. Short memory/runtime calibration and same-seed training repeats; reload
   every family and compare exact graph/repeat/eager assigned episode rows,
   including a partial final evaluation wave. Keep failures; no silent retries.
3. Fix common feasible batch settings and per-game training lengths from
   calibration, then freeze the packet before comparing models. Keep approximately
   eight checkpoints over each curve; evaluate all, never just the best one.
4. Add fixed/random-gaze controls and the sprite aggregate-feedback adapter.
   Keep each game's metric separate; don't average raw incompatible scores.
5. Schedule the real5090 campaign when available. Shared GPU reservation,
   serial jobs, monotonic checkpoint/process receipts, all declines/failures,
   seven fixed training replicas and complete held-out skin/world quotas.
   Report measured quality/time curves and uncertainty separately per hardware.

The new supervisor has separate local short-smoke/numerical receipts; read
the current result report for actual scope. No overnight campaign or paper/
Pareto claim follows from the older preparation. The5090 launcher reruns
its own numerical and learner/reload gates before the new bounded campaign.
