# G240 independent intake and full-curve analysis

Received research revision: `0010423d`. Native policies were trained from the
retained `b102678e` sources. Analysis is offline scalar/artifact work on G240;
no GPU query, policy execution, continuation or native source change occurred.

## What was checked

`research/analysis/continuation_exports.py` reconstructs every completed native
feedback from the transported seed/drawing/checkpoint points. It clips each
point using the frozen per-game anchors, averages drawings and the two seeds,
then weights the six games equally. Training cost is charged once per game/seed.
It checks the exact 45-row PROTEIN history, proposal/effective-graph/duplicate
identities, 15,416 unique point assignments, 7,708 paired means, 1,128 game means,
282 final rows, and all 47 final aggregate scores/costs. It also reconciles
594 process receipts, 2,376 checkpoint timings and 10,890 native metric bins.

All 336 original per-game checkpoint export rows are unchanged. Six completed
jobs from partial trial46 remain in the broader SPS collection; they never
enter the completed comparison or PROTEIN feedback. The allocation is still
**failed at its time cap**, with 33 additional observations, not 36. The last
native proposal records replay of all45 observations.

Nine transported files match hashes in the remote prefix audit. The other253
hashed files from that audit are not in this compact Git delivery. Input hashes
and the missing-file list are in `analysis.json`. This checks exported aggregates
and their internal consistency; it is not a local audit of episode CSVs,
checkpoint contents, binary provenance or numerical GPU results. The remote
full audit remains a separate receipt.

Eight new artifact/scalar tests, five earlier export tests and twelve existing
continuation/prefix host tests pass: **25 checks**. Negative cases preserve the
failed status and reject changed history/graphs, rehashed missing point rows,
double-counted costs and clipping after game averaging. No CPU CNN was used.

## The search found a useful new tradeoff

Trials44/45 share this graph:

```text
36x44 grayscale frame
  -> C16, 8x8 SAME convolution, stride4, ReLU
  -> C16, 3x3 convolution, stride1, residual connection
  -> flatten -> projection64 -> H128/L1 recurrent core and policy/value heads
```

This is **one stage containing two spatial convolutions**, not one convolution.
No stage pooling or global average pooling is used. The quality reference has
one C16/7x7/stride4 convolution, projection64, no residual. Per-game learners,
training appearance mixtures, decision budgets and evaluator assignments stay
matched; only CNN coordinates differ.

Final discovery objective: new **0.422731**, quality **0.353913**, Nature
**0.298506**. New versus quality is +19.4% objective and +42.4% checkpoint cost
(899.981s versus631.885s). It expands the higher-quality part of the descriptive
frontier; it does not replace quality's faster tradeoff. On Connect4 it has
163,296 full-policy parameters versus quality's160,736, only1.6% more, despite
slower training. Spatial convolution reuses weights at many positions; parameter
count cannot stand in for compute or speed. This comparison changes both the
first kernel and the residual, so it does not establish a causal residual effect.

## Full curves, not only final checkpoints

The new CSV includes **188 aggregate points**: all47 models at all four scheduled
checkpoint fractions. Its cost sums the six games and both seeds. Fractions refer
to each game's own fixed decision budget; they are not a common decision count.
These are prefixes of full-budget learner schedules, not separately trained runs
with shorter declared budgets. No interpolation or synthetic time point is used.

Lower-endpoint descriptive frontier across the complete collection:

| Model | Budget fraction | Paired six-game training seconds | Normalized lower score |
|---|---:|---:|---:|
| swift-fox-3 |25%|144.49|0.0736|
| quiet-owl-17 |25%|151.75|0.1799|
| quiet-owl-45 |25%|229.71|0.1906|
| quiet-owl-17 |50%|299.10|0.2453|
| quality-reference |50%|319.43|0.2639|
| quiet-owl-17 |75%|445.08|0.3001|
| quiet-owl-45 |50%|453.56|0.3213|
| quality-reference |100%|631.88|0.3539|
| quiet-owl-45 |75%|677.00|0.3828|
| quiet-owl-45 |100%|899.98|0.4227|

For example, the residual model's50% checkpoint has a higher conservative
objective than Nature's final checkpoint (0.3213 versus0.2985), at31.8% less
cost. Their censoring bounds overlap: new0.3213–0.3363 versus Nature
0.2985–0.3776. This is a promising region to confirm, not certified dominance.
By the same lower-score comparison it dominates the cheap C8 model's final point.
Several apparent final-only frontier points disappear once earlier checkpoints
participate. See `aggregate-checkpoint-frontiers.csv` for every point, both bounds,
full-curve and final-only flags. Game curves retain all1,128 points and declines.

The existing `../review/combined/curves.html` retains the finer per-drawing curves
for all47 models. Pong measures point fraction; its bounds are censoring bounds,
not confidence intervals. None of these frontiers establishes statistical/SOTA
superiority.

## What makes confirmation necessary

Final equal-game conservative objective by training seed:

| Seed | New44/45 | Quality | Nature | New minus quality |
|---:|---:|---:|---:|---:|
|64173|0.39885|0.39763|0.31167|+0.00121|
|64174|0.44661|0.31019|0.28534|+0.13642|

About99.1% of the summed paired improvement over quality comes from the second
seed. The winner was selected adaptively from45 observations. The two seeds
therefore do not support a robust effect-size estimate or confirmatory interval.
At the final aggregate level, new lower0.42273 exceeds quality upper0.40578,
which resolves this collection's censoring ambiguity but not sampling uncertainty.

Per-game results are not uniformly better. New versus quality regresses Maze
(18.69% versus28.28% goals) and Flappy (23.66 versus24.40 pipes). Its Breakout
improvement reverses between seeds:8.06 versus10.68 in64173,15.88 versus8.21
in64174. Snake improves on both seeds. Final Connect4 average hides0% wins on
the one-pixel-per-cell drawing8 for new, quality and Nature; new also regresses
drawing6 (4x4 cells) versus both references. Pong's reflected/inverted drawings
remain difficult. Higher pooled score is not evidence of uniform drawing robustness.
Per-game seed means and worst/best drawing scores are retained separately.

The45 observations contain35 effective graphs. All six duplicate groups have
identical checkpoint hashes and identical scores at every seed/drawing/checkpoint.
Their runtime spans range from about0.4% to5.4%; repeat scores are a determinism
check, not additional independent training seeds. The C8 model's repeated total
costs range590.55–613.29s. Its0.008s advantage over the first happy-cat measurement
must not be presented as an architectural speed gain. All duplicates stay in the
curves and history; no fastest-repeat selection is suitable for a paper claim.

## Recommended next allocation, not launched

Freeze the residual44/45 graph now, along with quality and Nature. Confirm full
curves using fresh paired training seeds, identical fixed per-game learners and
all41 drawings. Predeclare the primary objective and retain per-game/per-drawing
results, Pong bounds, every checkpoint, decline and failure. Use repeated timing
blocks with a recorded balanced order to assess drift rather than comparing a
single fastest measurement. Five fresh seeds for these three models means90
training jobs and2,460 drawing/checkpoint evaluations, plus required repeat/eager
checks; derive a wall-time cap from real train **and evaluation** receipts before
launch. No five-seed campaign is authorized or executed by this document.

The cheap C8 graph and Maze-specialist11 remain optional separate fast-front
confirmation candidates. The winner is not a replacement for these tradeoffs.
An ablation with C16/K8/no-residual (existing23/28) can separate the kernel change
from the residual; it needs fresh paired evidence before causal claims. Profiling
the residual stage may identify speed work after confirmation, without changing
the frozen models mid-experiment.

Before any future continuation, add an explicit audited-prefix import contract
and a budget-aware stop before starting another all-game panel. Preserve the
failed allocation and partial trial46. Never turn its status into success, silently
reseed/reset the optimizer, or rerun the old allocation. Further adaptive search
is lower priority than establishing whether the new curve survives fresh seeds.
IMPALA/Impoola are absent from this campaign; no claim against their curves follows.

## Reproduce the artifact analysis

From the research checkout, using its existing uv Python3.12 venv:

```bash
.venv/bin/python research/analysis/continuation_exports.py \
  research/results/learning-continuation-5090-20261009-prefix45 \
  --out build/analysis-prefix45-FRESH
.venv/bin/python -m unittest \
  research.tests.test_continuation_exports_analysis \
  research.tests.test_learning_exports_analysis \
  research.test_continue_learning_feedback \
  research.test_review_learning_continuation
```

On G240's audit worktree the interpreter used was
`/home/claude/cnn/.venv/bin/python`. Output directories must be fresh; the analysis
does not modify input packets. For completed runtime campaigns use the retained
5090 receipts, not this script's scalar execution time.
