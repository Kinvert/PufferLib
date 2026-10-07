# FlappyCNN

October 5 evaluation groundwork: [dedicated deterministic evaluation](DETERMINISTIC_EVAL.md)
now has a native exact adapter, frozen host spawn/image manifests and complete
episode audits. Four native targets compile; simulation/start/quota-corruption
checks pass. **GPU evaluator acceptance is pending**; pooled-v1 historical
scores below are unchanged and must not be relabeled exact.

Native PufferLib Flappy with synthetic grayscale pixels, generated directly into the observation buffer. This is PufferLib's own Flappy game, not ALE or an inherited external leaderboard. Copied from `ocean/flappy/flappy.h`, blob `d5b194c7fc9ccde430689b1562c975203cc58f0d`; original Flappy remains unchanged.

## Image and rules

The policy receives float32 CHW `[1,36,44]`: background 0, visible pipe rectangles 0.5, bird collision box 1. World and image y point down. Scale the original 420x640 world to the image, using floor/ceil bounds and clipping; draw pipes first, then the bird. A buffer clear plus at most six pipe rectangles and one bird rectangle generates a frame, with early rejection of offscreen pipes. No window, screenshot, texture load, framebuffer, allocation or RNG use occurs in image generation. The optional existing Raylib viewer shows the original decorated game and is not an exact policy-image preview.

World dimensions, bird radius, pipe geometry/movement/spawning, gravity, flap impulse, two actions, rewards and same-step resets match native Flappy. One native decision advances one physics tick. The native episode cap is 4,096 decisions; the original task treats reaching that cap as terminal and applies its crash reward. This behavior is preserved, not relabeled as a reward-free truncation. Pixel generation runs on reset and at the end of each nonterminal step.

The renderer displays only visible geometry. No velocity, elapsed tick, score, hidden/offscreen pipe coordinates or state vector is added to the image. The state baseline exposes bird velocity and the nearest two pipes even when offscreen; pixels lose subpixel precision and can show additional pipes if visible. A recurrent learner must infer motion from image history. State/pixel comparisons share the game but differ in information; encoder comparisons share exactly the same image.

Native `score` is mean pipes passed per completed episode. Native `perf` is mean `min(pipes_passed/20, 1)` per episode, **not wins or survival fraction**. For learning evidence report both, plus episode lengths/returns and full training/evaluation curves. The stock center-distance reward shaping is preserved for all models.

## Deterministic appearances

October 6: [the seven-ID drawing contract](REPRESENTATIONS.md) adds rounded
birds and outlined pipes. [Checks and retained receipts](../../research/FLAPPY_GEOMETRIC_ROBUSTNESS.md)
pass without executing a policy. Historical scores below use drawing 0.

| `env.representation` | Image |
|---|---|
| 0 | Base image |
| 1 | Horizontal reflection |
| 2 | Vertical reflection |
| 3 | Grayscale inversion |
| 4 | Rounded bird, filled pipes |
| 5 | Rectangular bird, outlined pipes |
| 6 | Rounded bird, outlined pipes |

Image shape stays fixed, with physics/actions unchanged. IDs 1–3 are reversible
transforms; IDs 4–6 change visible geometry. `env.representation_mode=0` selects
a fixed ID. Mode 1 assigns one ID per native environment slot using
`env.representation_seed` and the shared appearance hash; the assignment
persists through resets and consumes no game RNG. Default
`representation_mix_catalog=0` preserves the old four-ID mixture; value 1 opts
into all seven. Native environment RNG starts from its slot initialization;
`base.seed` is the learner/action seed, not a replacement for the appearance
seed. Keep mode/catalog/seed/slot count on reload. A mixed-mode search is not
proof of per-appearance robustness; use a declared fixed-ID evaluation panel.

## Native encoders and common recipe

For fixed-model learning calibration, [shared learner preparation](../../research/SHARED_LEARNER_RECIPES.md)
uses `learner_stock.ini` as an explicit stock actor/vector/learner overlay while
keeping all four CNN graphs and H128/L1 fixed. It changes no shared defaults or
game semantics. Two all-drawing/three-seed candidate panels and paired native
host evaluation starts are prepared; no new training occurred. The larger
minibatch remains unqualified for the baseline families, and both GPU holds
remain. Preparation is separate from the historical pilot launchers below.

The custom encoder hook recognizes `PUFFER_FLAPPYCNN` and reuses the existing CUDA code under `ocean/connect4cnn/`. No kernels or Raylib dependencies are copied into this environment. IDs 1–5 retain their meanings; ID 0 uses the compiled reference/default. Nature is ID 2, existing Flex is ID 4. IMPALA/Impoola use ID 0 with per-build `C4_IMPALA_CNN`/`C4_IMPOOLA_CNN` selectors. Encoder 5's new math still needs its own GPU acceptance checks; this environment does not validate it automatically. Float32 only is the supported research workflow.

`config/flappycnn.ini` preserves stock Flappy environment/learner/core settings and adds the existing Flex shape. It is a starting config, not pixel tuning. `compare.ini` is a separate **untuned matched canary**: state plus four pixel encoders, same original environment, H128/L1 core, learner, 64 agents, 65,536 decisions and four checkpoints. Its Flex shape was selected on Connect4; it is not a Flappy-selected winner. State policy and pixel encoders have unequal parameter counts and observation information.

## Checks and bounded execution

Environment/pixel safety checks do not run a CPU neural model:

```bash
bash ocean/flappycnn/tests/run_all.sh
```

September 26 checks pass: 49,152 same-seed transitions agree byte-for-byte with original Flappy across eight slots/seeds and three episode caps, with separate reward/terminal comparisons. Fixtures force boundary/pipe crashes, passing, pipe respawn, cap/reset behavior, fractional rectangle coverage, clipping at all corners, draw order, dirty-buffer overwrite and absence of velocity/score/tick leakage. All four fixed appearances and three mixed seeds preserve original physics/RNG; repeated traces match and invalid appearance options reject. ASan/UBSan pass. These validate CPU environment semantics, not CNN math, training or GPU determinism.

Prepare isolated five-model configs/source receipts without accessing a GPU:

```bash
bash ocean/flappycnn/canary.sh --prepare-only
```

Build all models with existing shared dependencies, without GPU execution:

```bash
NVCC_ARCH=sm_120 NVCC_EXTRA='--threads 1' bash ocean/flappycnn/canary.sh --build-only
```

When GPU access is available and no competing job is present:

```bash
NVCC_ARCH=sm_120 bash ocean/flappycnn/canary.sh
```

This serial bounded canary trains each model for 65,536 decisions, retains all four checkpoints, and reloads the final checkpoint for separate-seed evaluation of at least 64 completed episodes. Each training/evaluation process has a 120-second timeout. Raw native metrics, process time, source/config/build commands and hashes, GPU metadata and failure status are retained in a fresh `build/flappycnn/canary.*` directory. The external reporter verifies common executed settings, finite checkpoint weights, cadence and evaluation completion, then produces CSV/JSON/Markdown with process SPS and native timing separately. It performs no optimization or training. No W&B upload or search is launched by this canary.

Evaluation currently uses native pooled-v1 episode accounting: completed counts may exceed the requested total. The finite episode cap helps bound trajectories but does not implement exact per-episode allocation or establish an unbiased confirmation estimator. This is plumbing, not the final paper evaluator. Timeouts/failures stop the canary and remain recorded; no selective recovery is performed.

## Quality model with stock Flappy learner settings

Kinvert authorized a single 5060 learning pilot using our existing quality encoder and H128/L1 core with stock Flappy environment/vectorization/learner settings. The stock requested 20M decisions yield 19,922,944 actual decisions after native rollout rounding. This is separate from the five-model short canary and changes no shared configs. It starts from random weights, retains intermediate checkpoints and evaluates the final checkpoint separately. [Exact settings and interpretation](../../research/FLAPPY_STOCK_PILOT.md).

```bash
NVCC_ARCH=sm_120 NVCC_EXTRA='--threads 1' bash ocean/flappycnn/stock_pilot.sh
```

The runner refuses unavailable or competing GPU work and writes fresh `build/flappycnn/stock-pilot.*` evidence. No W&B upload or adaptive search. The September 26 5060 pilot completed: 19.923M decisions in 56.12 process seconds, 355K process SPS, final mean score 52.25 pipes and clipped perf 0.9759 across 257 pooled-v1 episodes. All 19 checkpoint hashes/counts/finiteness and executed stock settings pass inspection. [Report/evidence](../../research/results/flappycnn/stock-pilot.DN7C2JF6/REPORT.md). One seed, appearance 0, quality encoder only; no baseline comparison or determinism replication.

Original stock Flappy was subsequently measured once with `stock_pilot.sh --state`: unchanged six state observations, H64/L2 and stock learner settings. It completed the same 19.923M decisions in 9.73 process seconds (2.048M SPS), with final mean 48.53 pipes / clipped perf 0.9472 over 266 episodes. It is 5.77× faster than the pixel pilot by process SPS; core sizes and information differ, so this does not isolate the pixel encoder's cost. [Full comparison/evidence](../../research/results/flappycnn/stock-pilot.tgKBYrNs/REPORT.md). Other CNN baselines and repeated seeds remain unmeasured on this task.

## Shared architecture-search preparation

The existing temporary preparation/reporting glue now selects this environment while the optimizer and training remain native C/CUDA:

```bash
# Existing validated Flex family, one architecture knob, short plumbing budgets.
bash ocean/connect4cnn/sweep.sh --environment flappycnn \
  --recipe ocean/connect4cnn/tests/flex_kernel.ini --max-runs 3 \
  --canary --prepare-only --wandb disabled

# Expanded grammar preparation only; remove no validation gates.
bash ocean/connect4cnn/sweep.sh --environment flappycnn \
  --recipe ocean/connect4cnn/sweep_flex2.ini --depth 2 --max-runs 3 \
  --canary --prepare-only --wandb disabled
```

Preparation uses Flappy's environment/learner recipe, environment-specific appearance bounds and artifact paths; it does not reuse Connect4 physics. Without `--prepare-only`, the first command runs a bounded native discovery canary on an available idle GPU. Short budgets establish plumbing, not learnability. Discovery uses `perf` (clipped pipes/20); held-out raw score and full curves remain necessary. Do not launch a long architecture/hyperparameter campaign from these examples.

Current status: environment/pixel fixtures and all five model builds pass. The quality encoder's 5060 stock-learner training/reload pilot passes with archived learning/timing evidence. The five-model canary reporter, other model runs, current-source numerical/gradient regressions and seeded GPU repeatability remain pending. This task is a development environment, not an untouched final-test game for architecture-selection claims.
