# Sprite suite cleanup and local-view research

Completed cleanup/working-pilot goal, October16. All changes are in `/home/claude/cnn`.
Forage is the reference for simple native C, deterministic state/pixel rules,
cached sprites, changed-cell copies, readable observations and logs.
Research macros are acceptable while action/observation variants are explored.

## Scope and completion evidence

- Catch, Platform, Maze, Runner, Key and Push: real local source directories
  and matching INIs, with preserved original sources.
- Simplify lifecycle/state code, avoid unnecessary redraw/copy/reset work,
  build only requested skins, and retain original game/RNG behavior for
  readability/performance edits. Version intentional changes.
- Add opt-in centered3x3 plus independently selected3x3 gaze where useful,
  factored cellX/Y actions, matching state/pixel visibility and geometry.
  Keep full observations and normal one-head action contracts as controls.
- Connect the native shared tile CNN. Benchmark pixel candidates against
  pixel candidates; state is a learnability/throughput reference.
- Verify game/RNG/state-pixel parity, dirty/full raster equality, bounds,
  gaze timing/hidden-cell isolation, checkpoint reload and numerical scope.
- Run short serial RTX5060 state/pixel trainings at meaningful stages;
  retain recipes, seeds, hashes, decisions, monotonic process time, native
  timing/SPS separately, and exact assigned-episode outcomes. Keep failures.
  Initial pilots use fixed seed7; larger comparisons use the predetermined
  seven replicas7/42/69/123/173/2027/9001. Never sweep seeds.
- Finish with runnable instructions, per-game results/limitations, and an
  explicit completion audit. Single-seed pilots are not Pareto/generalization
  evidence; compare identical observation/action/learner/world contracts.

Lane Dodge has another implementation owner; inspect current work before
overlapping edits. Its research/standard binaries remain distinguished.
Review findings are in `LANE_DODGE_REVIEW_LOG.md`.
The goal included its narrow cleanup: the live header had not changed
since midday. Main viewer tile offsets and documentation were corrected;
mode4 adds matching18-cell one-hot state for existing mode3 pixels without
changing that pixel/action contract. Its separate sanitizer, numerical,
repeatability, matched training and exact evaluation all completed.

## Completed

- Replaced six game symlinks plus sprite_common with real local copies and
  copied their missing INIs. Earlier sources/evidence remain unchanged.
  Original links and pre-cleanup copies: `build/sprite-cleanup-20261016/`.
- All twelve baseline host/sanitizer tests passed, then all twelve cleanup
  tests passed with uncached full redraw and copied-byte assertions. Six
  before/after state traces match byte-for-byte (three fixed seeds,6144
  decisions/game). Readable struct fields/resets, requested-skin atlas init,
  and changed tile+texture copy caching are implemented in all six games.
- Twelve captured before/after float32 targets compile. A new24-state-job
  ABBA allocation plus12 full-image pixel canaries and exact evaluations
  COMPLETED: all96 state checkpoint identities and six pixel checkpoint pairs
  match across versions/repeats; all exact state episode rows match. The
 291.444s allocation and measured scores/SPS are in
  `results/sprite-cleanup-5060-20261016/README.md`. No state speedup claim.
- Opt-in matched18-cell player/gaze observations and factored X/Y actions
  are implemented in all six main headers. All six native ASan/UBSan
  attention/crop/hidden-cell checks PASS, six research targets compile, and
  ten independent CUDA positive/zero fixtures pass outputs/all51624 encoder
  gradients at B1/3/8/64/256 with eager/graph repeats. The separate bounded
  training/evaluation allocation COMPLETED; read its actual receipts/status.
  A reporting-reader error after a successful episode-free Maze canary is
  retained; explicit receipt-checked continuation reuses completed native
  work. See `SPRITE_ATTENTION.md`; no learning/frontier result is assumed.
- Main `build.sh all` skips directories without matching environment entry
  files (currently osrs, sprite_common, vision). This is a four-line discovery
  fix, not a new environment/core abstraction; shell syntax checks pass.
- Lane Dodge now has mode4 matching the existing mode3 visible cells with
  one-hot identities, leaving mode3 pixel/action shapes unchanged. The
  tile-CNN builder captures the exact per-episode evaluator. Its distinct
  protocol is `LANE_DODGE_MATCHED_STATE.md`; verification is complete.

## Completion audit and lessons

[Seven-game report](results/sprite-attention-5060-20261016/README.md) and
`results/sprite-attention-5060-20261016/completion-audit.json` bind the
implemented scope to actual files and observations. Offline audit verifies
local directories,16 current source/header identities matching tested builds,
19 final sanitizer suites, six exact old/new state traces, cleanup checkpoints,
114 nonoverlapping successful native attention processes,20 CUDA numerical
cases,14 exact same-seed canary-training pairs,56 finite pilot checkpoints,
1,022 exact assigned episode rows and eight ordinary main native builds.
Lane's new pixel pilot also reproduces the earlier retained final weights.

This fulfills cleanup plus working state/pixel pilots; it does not complete
the wider paper/SOTA research. Temporary macros remain accepted research
scope. No clean upstream PR, new dataset, dependency/CUDA change, commit or
push was produced. Completed allocations must not automatically restart.

Useful patterns from Forage applied here:

- Keep game rules/observations explicit and readable; avoid compressed struct
  declarations and chained resets. Do not invent a generic game engine.
- Cache immutable sprites per requested skin and copy changed cell/texture
  rows only. Preserve independent uncached redraw checks so a cache cannot
  hide stale pixels. Actual state speed changes here were only noise-sized;
  do not turn fewer copies into an unmeasured training-speed claim.
- Reduce pixel processing at its source with selected cells, paired with
  matching semantic visibility. Keep the strong full-state reference and
  ordinary full-image paths available as separate controls.
- Verify repeatability with actual checkpoints, exact assigned evaluation
  and all action heads. Determinism does not itself prove learned quality.
- Measure changes before asserting they help. Forage's connectivity rejection
  could be removed after its own ablation; Maze/Key reset distance fields
  implement existing reward shaping and were not silently removed as though
  they were the same kind of optional generation check.

Remaining research priorities, outside this completed cleanup allocation:
fixed/random versus learned gaze; precomputed smaller tiles; architecture
comparisons at matched pixel/action/learner regimes across time checkpoints;
then the seven fixed replicas and new evaluation suites. Current single-seed
results expose weaker tasks honestly and do not establish a Pareto win.

## Original sequence (completed)

1. Run the copied suite's native sanitizer/renderer checks here.
2. Improve shared cache/copy paths, preserving game traces; measure short
   before/after state learning/SPS at the appropriate stage.
3. Pilot matched local/gaze observations, tile CNN and exact evaluation on
   one game, then apply the verified pattern to the others.
4. Run bounded state/pixel pilots and record every result per game.

Use the existing runtime/toolchain and shared GPU lock. No system/CUDA/Torch
changes, CPU neural training or upstream push. Build snapshots copy current
main sources; old worktrees are dependencies, never the new edit target.
