# Overnight RTX 5090 continuation: audited completed prefix

The allocation ended **failed at its time cap**, with **33 additional complete
trials**, following the original12: **45 total native PROTEIN observations**.
Elapsed native execution:43,080.265637486s (11h58m). The46th trial completed
six of its12 training jobs, then Connect4 training timed out with48.060s left
in its panel deadline. No partial-trial feedback was submitted. The requested
36 additional trials were not all completed; don't label this full success.

The final successful proposal for trial46 replayed all45 completed observations:
GP45/success45/failure0/random0. The separate offline prefix reader verifies
that receipt, every history row, source/build/config/layout identities, encoder
numerical receipts, all completed panel raw outcomes, learner consistency,
training costs and original calibration/references. Offline audit passes, with
no GPU query or policy execution. The original failed result remains unchanged.
Twelve host-only continuation/prefix rejection tests pass.

## Results

Best final equal-game fixed-anchor score: **quiet-owl-44 / quiet-owl-45**, duplicate
effective architectures, **0.422731105917145**. Quality-reference:0.35391309916541164;
Nature:0.2985055231419429. This is about19.4% above quality's discovery objective,
with about42.4% greater monotonic training cost:899.980731207s for45,
905.6305914229999s for44, versus631.884930176s for quality and664.976743635s
for Nature. Cost sums the final checkpoint time once per game/seed, not evaluation.
Quality-reference and the new architecture are different quality/time tradeoffs;
the new model does not dominate quality on cost. These are adaptive discovery
seeds, not held-out evidence or statistically certified superiority.

Architecture44/45: encoder4, depth1, projection64, no global pool, first stage
C16/K8/stride4/no pooling/residual1; H128/L1 core. Learners remain fixed per game.
The33 new suggestions contain24 new effective architectures and9 duplicates.

Final scores averaged across every drawing and both paired training seeds:

| Environment / metric | New architecture44/45 | Quality | Nature |
|---|---:|---:|---:|
| Connect4 / win fraction |45.91%|43.79%|42.58%|
| Pong / final point fraction bounds |72.36–81.45%|41.57–72.69%|15.63–63.06%|
| Flappy / pipes |23.66|24.40|33.04|
| Breakout / score through terminal/cap |11.97|9.45|6.46|
| Snake / ending length |7.77|6.15|7.19|
| Maze / goal fraction |18.69%|28.28%|22.98%|

Pong bounds reflect censoring, not confidence intervals or a win-rate metric.
Candidate44/45 regresses Flappy and Maze. Other architectures lead individual
games: trial37 Connect4 48.94%; trial25 Pong point fraction85.66% (uncensored);
trial35 Breakout13.62; trial36 Snake9.73; trial30 Maze28.54%. Those are separate
per-game selection outcomes, not one common architecture or confirmatory wins.
Nature still has the highest Flappy final score in this collection.

## Artifacts for the 5060 coding agent

- `analysis.json`: completed-prefix audit, original failed-result SHA, final
  replay counts, scalar feedbacks and retained artifact hashes.
- `exports/`:282 final per-environment rows,1128 checkpoint rows and47 normalized
  final model scores/costs.
- `review/combined/curves.html`: complete per-condition paired score/time viewer;
  JSON/CSV tables alongside it.47 models,41 conditions,15,416 raw observations,
  7708 paired means,564 completed comparison training jobs; zero missing cells
  within this explicitly selected completed prefix. Partial trial46 is excluded
  and declared in the selection warning, not hidden or upgraded to success.
- `sps-report/`:594 completed training jobs including calibration and six jobs
  from the partial trial;2376 timed checkpoints,10,890 native metric bins.
  Keep this broader cost coverage distinct from the completed comparison.

Native SPS arrays are step-bin means, not instantaneous samples or qualified
native-average SPS. Process SPS is decisions divided by process seconds;
checkpoint cumulative/interval SPS uses monotonic receipts. New architecture45
process SPS (two-seed means): Connect4 154,061; Pong286,263; Flappy330,446;
Breakout269,716; Snake312,417; Maze180,780 decisions/s. Corresponding quality
means:177,015;372,840;474,859;369,134;442,923;297,533. The new model is slower.

Code delivered with this report: `research/continue_learning_feedback.py`,
`research/run_continue_feedback_5090.sh`, `research/review_learning_continuation.py`,
both host-test files and the offline SPS/per-game exporter. The retained native
source worktree remains `b102678e`; never edit its sources or frozen packets.
The initial preparation failure due to using a different Python executable is
retained; corrected preparation uses the exact sibling executable path.

Local packet:
`/home/keith/Git/ml/cnn-5090/build/learning-continuation/5090-20261009-overnight-v2`.
Local prefix review:
`/home/keith/Git/ml/cnn-5090/build/learning-continuation-review/5090-20261009-prefix45`.
Failed-result SHA256:`806f7e0b8c1fb8798a51714806ff9691d30f9e6125ae25fa1727743f29997003`.

No new GPU allocation is running or launched by this review. Before another
authorized continuation, the coding agent should add an explicit audited-prefix
import contract and a budget-aware stop between panels. The current continuation
runner accepts successful parents and deliberately refuses this failed parent;
do not rewrite the failed status, discard the partial evidence, reset/reseed
PROTEIN or silently restart its36-trial allocation. All45 successful feedbacks
are available for a subsequent explicitly bounded continuation.
