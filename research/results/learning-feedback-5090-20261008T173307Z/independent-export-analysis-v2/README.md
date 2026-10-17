# G240 analysis of the completed 5090 exports

Received by Git at `6e30f9bb`; executed campaign source was `b102678e`.
Analysis lives in a separate G240 worktree, `/home/claude/cnn-audit-20261008`.
The original `/home/claude/cnn` checkout and frozen 5090 inputs are unchanged.
No GPU query, training, evaluation or continuation runs in this analysis.

The 5090 agent reports its overnight continuation is now running from all 12
observations, at most 36 additional trials with a 12-hour cap. These files describe
only the completed first campaign. Continuation ownership stays with
`research/continue_learning_feedback.py` on the 5090; this creates no competing path.

## Main findings

- Quality-reference has final fixed-anchor score0.353913 at631.885s total paired
  training cost, versus Nature0.298506 at664.977s: a higher observed aggregate
  score with4.98% lower cost. This is not dominance in every game.
- The final aggregate frontier contains swift-fox-3, happy-cat-1 and
  quality-reference. Happy-cat-1 scores0.311465 at590.555s. Swift-fox-3 scores
  only0.102099 at564.535s; being the cheapest mathematical frontier endpoint
  does not make it an attractive delivered model.
- Maze is the strongest clear short-time confirmation target: quiet-owl-11 at
 16,777,216 decisions has goal fraction0.234848 at50.841s; Nature's final
 33,554,432-decision checkpoint has0.229798 at120.559s. The searched checkpoint
  costs57.83% less, with a slightly higher observed score. At the matched final
  budget, quiet-owl-11 also scores0.270202 at102.918s. Quality-reference reaches
 0.282828 at112.596s. Inspect all four checkpoints, not only the favorable pair.
- Early Pong is promising: happy-cat-1 at2,097,152 decisions has point-fraction
  bounds0.529523–0.535697 at5.124s; Nature at the same decisions has
  bounds0.069668–0.306925 at6.075s. The averaged deterministic censoring bounds
  are separated here. Later Nature/quality-reference upper bounds overlap the
  searched result, so general true Pong superiority is unresolved. These are
  points won, not match wins, and the bounds are not confidence intervals.
- Quality-reference is faster than Nature in all six final-budget job pairs by
 2.23–6.61% using monotonic checkpoint cost. It leads final Connect4, Breakout
  and Maze means; Nature leads Flappy and Snake means. Connect4's final score
  advantage is small and especially needs independent seeds.

The current mixed-drawing/all-game recipe differs from old single-drawing
Connect4/Pong experiments; do not compare these scores as if those recipes and
evaluation conditions were unchanged.

## What was checked

`research/analysis/learning_exports.py` checks all seven committed tables:
192 unique jobs, four checkpoints each/768 hashes,3,520 native bins,336 game/model
checkpoint rows,84 final rows,14 final aggregate observations, all assigned
seeds/drawings, process and interval SPS arithmetic, two-seed mean costs and
once-per-job aggregate costs. It binds their exact bytes in `analysis.json`.
No calibration seeds enter the search/reference comparison. Duplicates and
declines remain; neither quality selection nor filtering can remove source rows.

`REPORT.md` lists every per-game lower-endpoint frontier; `checkpoint-frontiers.csv`
retains all336 points and both endpoint frontiers plus dominance using
nonoverlapping censoring bounds. `curves.html` is a self-contained exploratory
view of all14 models and four checkpoints per game. Filtering never recalculates
frontiers or axis ranges. The aggregate view contains final points only.

This is an export-consistency audit. The raw archive, weight checks, per-episode
CSVs, candidate configurations and native audit receipts have not been received
here. We cannot independently confirm the remote raw audit or reconstruct every
per-drawing/per-seed score from these averaged tables. The fixed-anchor aggregate
clips individual cells before averaging; it cannot generally be recomputed from
game means alone. It is retained from the published final feedback table.

Five artifact/scalar checks cover actual data, corrupted SPS, missing checkpoints,
incorrectly charged drawing costs and overlapping Pong bounds. They do not test
a CNN on CPU or establish numerical/statistical neural qualification.

## Next evidence and confirmation

The existing `prepared/review/combined/curves.json` and original search result
contain the point-level seed/drawing/checkpoint data and trial architecture
descriptors. Export those existing artifacts after coordinating with the 5090
agent; no new training or alternative continuation supervisor is needed. The full
archive is still required for an independent raw-receipt audit. In particular,
parameter counts alone do not identify quiet-owl-11's actual architecture.

After the authorized continuation finishes, freeze candidates, target checkpoints,
learner recipes, appearances, evaluation caps and primary claims before fresh
confirmation seeds. Maze and early Pong are discovery targets, not findings that
can be certified by searching longer on the same two seeds. Paired-seed
uncertainty must treat renderings/episodes as nested conditions, not independent
training seeds. Timing confirmation should use an idle GPU and balanced model
order; retain process and checkpoint costs separately. Generalization claims
must retain the games where Nature wins. No publication/SOTA claim yet.

Reproduce without any GPU or policy execution:

```bash
/home/claude/cnn/.venv/bin/python research/analysis/learning_exports.py \
  research/results/learning-feedback-5090-20261008T173307Z --out FRESH_OUTPUT_DIR
```
