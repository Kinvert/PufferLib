# BreakoutCNN

October 5, 2026. Native PufferLib Breakout with float32 grayscale observations
generated in memory. Original simulation copied from `ocean/breakout/breakout.h`,
SHA256 `8b7f025ee65a93295e311b8942dcc6bb4239d80de21dc5d381ec5ddab8840d34`.
Original Breakout is unchanged. This is **not ALE/Atari Breakout**.

## Pixels and simulation

One frame, CHW `[1,36,44]`. Scale the stock 576x330 world into this canvas:
background 0, intact bricks 0.5, paddle 0.75, ball 1. Clear the buffer, then draw
clipped floor/ceil rectangles in brick/paddle/ball order. Destroyed bricks
disappear. No screenshot, window, texture, framebuffer, allocation or RNG draw
is needed. The optional original viewer is independent of policy observations.

Stock geometry is 6x18 bricks, three discrete actions and frameskip 3.
Simulation, RNG, rewards, collisions, paddle changes, ball acceleration,
brick-board refill, lives and resets are unchanged. Other positive frameskips
are accepted; fix them across models. Initialization rejects nonstock brick
counts/continuous actions instead of claiming unsupported protocols work.

Pixels omit explicit velocity, score, remaining lives, brick-row colors and
HUD text. They lose subpixel precision; neighboring bricks can merge under
downsampling. H128/L1 recurrence can infer some motion but doesn't restore all
hidden state. All CNNs receive identical pixels. State is an optional reference,
not the performance target.

| `env.representation` | Appearance |
|---:|---|
| 0 | Base raster |
| 1 | Horizontal reflection |
| 2 | Vertical reflection |
| 3 | Grayscale inversion |
| 4 | Swap brick/paddle intensities (0.5 and 0.75) |

These transforms are reversible on the raster. Physics/action IDs stay fixed;
reflection does not rename left/right actions. Image shape/encoder FLOPs stay
fixed. Mode 0 uses a fixed ID. `representation_mode=1` assigns one ID per slot
from the explicit `representation_seed` using the shared appearance hash.
Assignments survive resets and consume no game RNG. Model/appearance seeds are
separate controls. An optimizer choosing the easiest drawing isn't robustness.

## Verified preparation

```bash
bash ocean/breakoutcnn/tests/run_all.sh
.venv/bin/python ocean/connect4cnn/tests/test_sweep_tools.py
.venv/bin/python research/tests/test_pixel_robustness.py
```

The environment harness compares state, brick arrays, logs, rewards, terminals
and RNG against original Breakout: 18,432 decisions per panel, three game seeds,
frameskips 1/3/8, all five fixed and three mixed appearance panels. Literal
pixel fixtures check clipping, overlap, removal, offscreen objects, overwrite
and hidden-state isolation. Launch/life-loss/terminal, repeat and ASan/UBSan
checks pass. These validate simulation, **not a CPU neural model**.

Shared encoders are reused from `ocean/connect4cnn`; none are copied. Quality
is encoder 4, Nature 2; IMPALA/Impoola use encoder 0 with per-build selectors.
The default/IMPALA/Impoola native float32 targets compile on G240 (sm_120).
Nature is dispatched by INI from the default binary. Encoder-5 validation
remains a separate 5090 gate. Compilation isn't GPU train/reload validation.

```bash
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash build.sh breakoutcnn build/breakoutcnn/train --float

bash ocean/connect4cnn/sweep.sh --environment breakoutcnn \
  --recipe ocean/connect4cnn/tests/flex_kernel.ini --max-runs 3 \
  --canary --prepare-only --wandb disabled
```

Preparation writes configs/receipts, no training. Current GPU holds remain.
[Matched fixed-model preparation](../../research/MULTI_ENV_ROBUSTNESS.md) is the
next comparison path. [Validation receipts](../../research/results/breakoutcnn/environment-20261005/README.md).

## Evaluation gates

Native `perf` is score/max_score (864 under stock geometry), not wins. Report
score, normalized score and duration separately. The stock game starts with
`num_balls=5` and ends on `<0`, allowing six losses. A frame-skip action can end
an episode and continue into its reset game before returning. No episode cap
exists. All this behavior is deliberately preserved and tested for parity.

The [exact native adapter and suite tools](DETERMINISTIC_EVAL.md) now capture
ending logs and stop before frame-skip continuation, with explicit episode RNG
and administrative frame caps. Four targets compile; CPU world/raster and
host/audit tests pass. The supervised checkpoint launcher now supports GPU-free
`run --prepare-only`; its GPU acceptance remains open.
Training semantics are unchanged. Do not call a cap a loss or treat pooled-v1
counts as exact quotas. No BreakoutCNN GPU run or Pareto result exists yet.
