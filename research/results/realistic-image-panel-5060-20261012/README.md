# Larger-image panel / RTX5060 intake

October14 retention update: with explicit owner approval, the historical raw
`build/image-panels-20261012` tree was deleted after verified compact receipt
preservation. Sources, logs, configurations, hashes and summaries remain; local
arrays/weights/binaries no longer support raw replay. See AGENTS.md for the
archive path/hash. The historical outcomes below remain unchanged.

G240, October12. Research sources in `/home/claude/cnn-audit-20261008`, based on
6a0a2190 with retained subsequent experimental changes. This is not an upstream
PR or a completed overnight5090 handoff. Hardware collections remain separate.

Actual direct RGB policy frames: Connect4576x672, Pong640x620, Flappy640x420,
Breakout330x576, Snake/Maze616x616 (height x width). Snake/Maze retain local
11x11 visibility. No preprocessing; float32NHWC. One common C8/K8/s4/GAP/
projection64 CNN, H128/L1, per-game fixed four-actor/H16/B16 plumbing recipes.

## Passed independent CUDA encoder checks

`build/image-panels-20261012/all-shape-cuda-checks/` contains actual fresh
two-worker checks on all five distinct RGB shapes (Snake/Maze share a shape),
B1/4/16. All90 process/case instances/360 native calls pass forward and
backward-on-actual-float32-branches, selected CUDA finite differences, parameter
preservation and eager/graph/repeat/process byte comparisons. Tolerances remain
unchanged. No CPU neural reference, full-policy numerical qualification,
Nature/IMPALA new-image math or learning/frontier claim follows.

The later `cuda-catalog-v2` freezes new supervisory code with the unchanged
source-checked numerical libraries. That catalog is a fresh allocation contract,
not a copied success or retrospective update of the completed first checks.

## Failed eight-checkpoint panel, audited prefix

`build/image-panels-20261012/eight-checkpoint-validated/` completed all six
512-decision native training jobs and all48 finite checkpoints at64-decision
cadence. Training seed7, appearance seed17, evaluation seed7007. Every scheduled
checkpoint targets all drawings of its game at five exactly assigned episodes
and four slots,328 evaluations planned. This is plumbing, not learning adequacy.

The panel failed on its first Maze evaluation's30s timeout. It completed280
evaluations and ten repeat/eager checks before that failure. The offline audit
reconstructs those retained receipts; missing48 Maze evaluations and two Maze
repeat checks stay missing. Four Maze episode rows completed before termination;
the fifth assignment is not discarded or averaged as a success.

| Game | Process seconds | Process SPS | Full-policy parameters |
|---|---:|---:|---:|
| Connect4 |2.824148|181.29|60,616|
| Pong |2.619903|195.43|60,104|
| Flappy |2.018939|253.60|59,976|
| Breakout |1.422811|359.85|60,104|
| Snake |2.420784|211.50|60,232|
| Maze |2.620344|195.39|60,360|

Process SPS includes startup/eight checkpoint writes. These few-second jobs do
not measure sustained throughput or a Nature comparison. Native timer/profile
arrays remain separately retained, not substituted for process measurements.

## Separate final-Maze inference probe passed

`build/image-panels-20261012/maze-final-timing-probe/` uses the scheduled final
512-decision Maze checkpoint, unchanged first-drawing suite, five assignments,
native map-area horizons and four slots. New bounded allocation:60s/process,
200s total, no training and no restart of the failed panel.

Graph36.839s, repeat36.637s, eager36.826s; supervision window111.606s. All three
completed five assigned episodes and their entire CSV bytes match, including
start RNG/world/raster and action hashes. Two goals per run are a plumbing
observation, not a trained-quality comparison. This explains why the earlier
30s allowance was insufficient without shrinking or excluding episodes.

## Admission and next work

The new all-game native host proposal preflight actually registers six
layouts/storage cases and passes artifact replay without GPU query. Unsafe
indices or storage floors now stop before per-proposal CUDA checks. This is
not measured peak VRAM. Numerical workers now share the outer campaign deadline.

The focused scalar/config suite passes42 executed tests with four
explicit-native-registry tests skipped; those skips are not accepted native
evidence. Earlier explicit registry/native construction reports keep their scope.

## Actual native image feedback loop passed

`image-feedback-v3-prepared/` completed three trials in600.494s:18 native
trainings at64 decisions/game,123 all-drawing evaluations,36 repeat/eager checks,
18 repeated full-policy layouts and18 pre-CUDA layout/storage admissions. Every
evaluation has one assigned episode and four slots, with unchanged horizons.
Across proposals, all270 CUDA process/case instances/1080 native calls passed.
Training processes total13.290472s; the large remainder is checks/evaluations
and auditing, not substantial learning. Actor/learner batches remain4/16.

| Trial | Projection | Duplicate of | Plumbing score | Charged train seconds |
|---|---:|---|---:|---:|
| happy-cat-1 |64|none|0.472222|3.877917|
| mystic-tree-2 |32|none|0.500000|3.856274|
| swift-fox-3 |32|mystic-tree-2|0.500000|3.758256|

Scores are the existing final-checkpoint equal-game clipped-anchor plumbing
objective, with uncalibrated0/1 anchors. They are not quality/generalization
evidence. The small episode quota is explicit; do not quote them as performance.
The duplicate is preserved. Final native replay records all three observations,
nonzero GP observations and three native success observations. The offline
`image-feedback-v3-audit/analysis.json` independently reparses raw CUDA arrays,
all panel/episode receipts, learner matching, duplicate identities and native
history, and reproduces every feedback exactly. No policy runs in that audit.
All six duplicate full-policy checkpoint hashes match, not just aggregate scores.

Initial preparation failed because the new recipe lacked the native coordinate
`scale`. Its failed describe receipt remains in`image-feedback-v3/`; a fresh
allocation added`scale=auto`. The Python recipe gate now catches the missing
field before native/GPU work. Completed allocations are never restarted.

Final regressions pass54 executed host/scalar/artifact tests including the
explicit retained image metadata registry, with one unavailable legacy-native
fixture skipped. This is separate from the actual CUDA and policy evidence.

`receipts/` retains1300 compact files: complete captured reader/native source
snapshots, numerical reports/requests/worker inventories, audits, feedback/GP
ledgers, actual all-drawing episode receipts and failure/timing diagnostics.
Verify with`cd receipts && sha256sum -c ../receipts.sha256`. The15MiB collection
omits raw numerical arrays, checkpoints and native binaries. Its checksums
establish retained-file identity, not independent full transported acceptance.

The overnight handoff still requires new-image Nature qualification, meaningful
per-game calibration, the versioned curve/coverage objective, fresh confirmation
allocation and calibrated approximately12h admission/reporting. Read
[the protocol](../../IMAGE_OVERNIGHT_PROTOCOL.md). All raw artifacts and captured
reader sources remain local; compact receipts are not a transported full raw
numerical archive. No CUDA/system changes, CPU CNN or long G240 campaign.
