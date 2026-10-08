# Received RTX 5090 native cross-game feedback canary

October 7, 2026. Verified received archive
`feedback-5090.5090-20261007T223659Z.V75JFtFs.tar.gz` against its sidecar:

`019134c2c2289665c6a27a1c5fc4451869673e912a0e486068468f0ffbd7d779`.

Raw archive remains in the repository root. Extracted without overwriting into
`build/hardware-artifacts/feedback-5090-20261007-received/5090-20261007T223659Z/`.
All 7,491 archive members passed relative-path/regular-file/directory checks.
Remote preparation reports source commit
`bb4d1c10683e06e8373a4f971bee8bc447761279`, with dirty auxiliary Pong work;
do not describe it as a clean checkout. Captured owned-source hashes match G240
for the native core, feedback bridge/supervisor, panel, active evaluators,
game headers and current per-game comparison recipes. Thirteen ancillary Pong
audit/legacy recipe/package files differ or are absent locally; their captured
source is preserved. Full build/vendor/system link closure is not certified.

Hardware: RTX 5090, driver 580.105.08, CUDA compiler 13.1.115,
host `keith-MS-7E70`. Keep separate from G240 RTX 5060 measurements.

## Actual execution

Remote native receipt audit reports success for all three trials. Each uses the
same CNN across Connect4, Pong, Flappy, Breakout, Snake and Maze, with fixed
per-game learners and one seed (60173), 65,536 decisions/game, all 41 drawings,
and 17 assigned evaluation episodes per drawing/check.

- 18 completed training jobs, 123 drawing evaluations, 36 repeat/eager checks.
- 2,703 assigned episode executions, retaining zero scores.
- Campaign execution: 102.862307 seconds, excluding builds/preparation.
- Training process time: 9.108919 seconds total.
- Native monotonic checkpoint training cost: 7.240580 seconds, charged once/job.
- Optimizer process time: 1.155920 seconds, including the final unused proposal.

Do not add or substitute those different timing definitions. These few-second
training jobs include substantial startup/checkpoint overhead and are not a
steady-state speed benchmark.

The final unused proposal records `observations_replayed=3`,
`gp_observations=3`, `success_observations=3`, `failure_observations=0`.
Native suggestions used different normalized coordinates, but all round to
**C16**. Trials 1 and 2 correctly retain `duplicate_of=0`; C8 was not visited.
They are not three independent architectures or independent training seeds.

All trials return the same normalized feedback quality, 0.004399724802201582.
Connect4/Pong/Flappy/Maze normalized scores are zero; Breakout and Snake supply
small nonzero scores. This short canary is not useful evidence of learning,
architecture superiority, a Pareto win, generalization or SOTA.

## Local intake verification, no GPU

Current offline input inspection succeeds on the extracted frozen packet.
The retained top-level result hash matches `review/analysis.json`. Independently
verified all 18 checkpoint SHA256s/registered byte sizes/finite float32 weights,
all 159 evaluation result JSON hashes and episode CSV hashes, 17 rows per CSV
(2,703 total), and all 36 repeat/eager CSVs byte-identical to their matched
drawing-zero graph reference. No GPU query, policy execution or training here.

This supplements the original remote full native audit. It is not a new local
re-execution of the entire source-bound audit, model math or learning acceptance.
The original packet uses absolute 5090 paths; preserve those and its raw archive.
Small original reports, observation/training tables and native proposal receipts
are retained alongside this README. Broader numerical/learner/calibration,
meaningful-budget/architecture-diversity and independent confirmation gates
remain. Do not automatically rerun or extend the completed canary.
