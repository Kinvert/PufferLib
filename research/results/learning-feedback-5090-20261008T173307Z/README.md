# Completed RTX 5090 learning feedback campaign

Source: `b102678e`, Kinvert/PufferLib `cnn-research`. Packet:
`/home/keith/Git/ml/cnn-5090-learning/build/learning-feedback/5090-20261008T173307Z`.
The audited campaign finished successfully in **17,668.335 seconds (4h54m28s)**,
excluding preparation/build/archive. All six games passed quality/Nature learning
controls. Native PROTEIN completed 12 architecture trials and its final unused
proposal replayed all 12 observations, with GP feedback active.

Counts including calibration: 192 training jobs, 5,248 drawing/checkpoint
evaluations, 384 graph-repeat/eager checks and 768 timed checkpoints. Final
search/reference comparison: 168 training jobs, 4,592 observations, 2,296 paired
means over 41 game/drawing conditions, no missing observations or failures.
Four checkpoints, two paired training seeds, every drawing; full-catalog mixed
training. Learner recipes/budgets are fixed per game; only CNN architecture varies.

## Per-environment final results

Each row averages all drawings and both search-phase seeds at the fixed final
budget. The HTML viewer's default condition is Connect4/drawing 0, not an
across-environment score. Pong intervals are deterministic censoring bounds,
not confidence intervals; keep both endpoints.

| Environment / metric | Quality reference | Nature CNN |
|---|---:|---:|
| Connect4 / win rate | 43.7879% | 42.5758% |
| Pong / bounded win rate | 41.5671–72.6890% | 15.6278–63.0632% |
| Flappy / pipes | 24.3961 | 33.0411 |
| Breakout / score at terminal or cap | 9.44545 | 6.46061 |
| Snake / ending length | 6.14646 | 7.18939 |
| Maze / goal fraction | 28.2828% | 22.9798% |

Quality-reference leads final Connect4, Breakout and Maze scores; Nature leads
Flappy and Snake. Search candidates `happy-cat-1` and `quiet-owl-6` are duplicate
architectures with Pong bounds 59.7979–61.6428%; this is a stronger lower bound,
not proof of true Pong superiority.

The fixed-anchor equal-game final objective is **0.3539130992** for quality,
**0.2985055231** for Nature. Best searched aggregate is `quiet-owl-9` at
**0.3160023664**. Aggregate cost is final monotonic checkpoint training time,
charged once per game/seed: quality 631.884930176s; Nature 664.976743635s;
quiet-owl-9 1,046.831524599s. Compare complete checkpoint curves and per-game
tradeoffs, not just final aggregate scores. No searched aggregate winner over
quality has been established by this allocation.

## Tables and timing

- `per-environment-final.csv`: 84 final model/game rows.
- `per-environment-checkpoints.csv`: 336 model/game/checkpoint rows.
- `normalized-final-feedback.csv`: 14 model objectives/costs. All 12 searched
  rows match the original audited PROTEIN feedback within 1e-12 score / 1e-9s cost.
- `training-sps.csv`: 192 process-level costs and decisions/process-second SPS.
- `checkpoint-sps.csv`: 768 monotonic checkpoint costs, cumulative and interval SPS.
- `native-bin-mean-sps.csv`: 3,520 original native metric bins.
- `snapshot-summary.json`: snapshot coverage and timing semantics.

Checkpoint time uses native post-rename CLOCK_MONOTONIC minus process launch;
startup and checkpoint writes are included, evaluation excluded. Interval SPS
uses differences of checkpoint receipts. Native metrics are step-bin means,
not instantaneous samples or a qualified native-average SPS. Do not substitute
native realtime uptime for monotonic training cost. Keep 5090 and 5060 separate.

Reproduce exports with `research/report_learning_progress.py PACKET --out FRESH`.
The original per-condition Pareto viewer is at
`PACKET/prepared/review/combined/curves.html`. Native raw receipts/weights/arrays
are in the archive below, rather than Git.

Archive: `learning-feedback-5090.5090-20261008T173307Z.Q1txeyXm.tar.gz`

SHA-256: `963993959ce965bde4033b083d22105f33d9522d976b0f0f9218761ffa331e08`

Local archive directory:
`/home/keith/Git/ml/cnn-5090-learning/build/hardware-artifacts/`.

## Continuation and interpretation

Kinvert authorized up to **36 additional trials / 12 hours** on October 8.
Continuation preparation is underway on the 5090; this report records only the
completed campaign. The native engine already supports exact full-history replay;
the separate continuation supervisor must import all 12 observations, verify its
first proposal against the parent's final unused proposal, reuse audited controls
and references, and preserve frozen sources/learner recipes/seeds. Do not restart
the completed packet or create a competing GPU allocation on either host.

These are adaptive discovery seeds with two paired seeds and fixed evaluation
episodes, not held-out confirmation. Pareto marks are descriptive; neither
statistical dominance nor publication-quality superiority is certified. Extra
search on the same seeds does not supply held-out evidence. Native optimizer
duplicates and losing trials remain in the ledger. Full learner math and other
scientific qualifications remain separate.
