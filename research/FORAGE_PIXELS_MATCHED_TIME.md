# Partial Forage: matched five-minute state and pixel comparison

October15, RTX5060 on G240. This is a single-seed engineering comparison,
not publication confirmation or a Pareto/generalization claim. Current raw
packet: `build/sprite-forage-pixels-matched-5060-20261015`. Read its
`full/COMPLETE` and `audit.json` before describing it as complete.

Both policies use movement5 plus independent categorical gaze X20/Y15.
They see eight cells around the player and nine around the selected gaze.
Geometry is identical: player world X/Y and signed gaze-minus-player X/Y.
State receives thirteen one-hot object categories in each cell (225 values).
Pixels receive the corresponding seventeen full-resolution32x32 RGB tiles,
plus the four geometry floats (52,228 values). There is no nearest-target
helper, hidden semantic pixel input, full-board CNN or downsampling.

The pixel policy is the existing custom shared-tile CNN, trained from scratch:
4x4/stride2/C8, then3x3/stride2/C16, then sixteen learned features per tile.
Concatenated tile features and geometry project to128 features. Both policies
use H128/L1 MinGRU and the same three action heads. State has83,200 parameters;
pixels103,976. Neither Nature nor IMPALA is being tested in this allocation.

## Common recipe and timing

Both new runs use64 agents, horizon64, minibatch256, four threads, one buffer,
synchronous float32/native CUDA training, fixed train seed7. Rewards, game
configuration and all remaining learner settings come from the successful
factoredXY/one-hot/relative state experiment. The common batching changes
the previous state's256 agents/minibatch2048; do not compare these runs as
if they retained that older recipe. Matching the smaller feasible batching
avoids confounding this state/pixel comparison with different update sizes.

LR0.0141914776 anneals to zero over each run's fixed decision budget.
Gamma0.998099923, GAE0.850000024, replay1, entropy0.00611945381, clip0.2,
value coefficient2, value clip0.2, gradient norm1.5 and momentum0.95 remain
matched. Training cap676, copies3, trees8, apple reward1, grape3.71950364,
bad-1.68730032, death-0.5, other0 and step-0.001 remain unchanged.

Separate from-scratch calibrations measure throughput. Decisions for each
five-minute run are selected from throughput alone, rounded to eight rollout
batches. Calibration checkpoints are not used to initialize the full runs.
The full runs have different decision counts and LR schedules in decision
space, aiming at the same elapsed time. All eight checkpoints are retained.
The primary checkpoint is prospectively defined as nearest300 monotonic
seconds since process start; ties choose earlier. If it differs from the
final checkpoint, evaluate and report both, including declines. Never select
by score. Record actual times rather than calling every budget exactly300s.

Training runs execute serially under the shared hardware reservation. Each
native process requires an empty compute PID list before launch. Process
SPS includes startup/checkpoint writes; checkpoint elapsed SPS uses native
post-rename CLOCK_MONOTONIC receipts. Native adjusted uptime/SPS are retained
separately. These checks do not certify absence of all hardware contention
throughout training, and no intrinsic hardware speedup is inferred.

## Evaluation and checks

Each compared checkpoint receives exactly128 assigned episode IDs for each
development suite seed420/1337,256 episodes total,16 slots,512-step cap.
Native manifests establish equal initial worlds/policy RNG independent of
policy architecture and episode completion order. Observation hashes are
checked against each observation mode's own manifest. These are reused
development worlds, not untouched publication tests. Collection perf is
`(apples + grape - penalties)/4` clamped to[0,1]; it is not win rate.
Grape exits are tabulated separately.

Both fresh calibration policies must pass graph/repeat/eager/independent-tail
exact per-episode equality, including all-head action hashes, before the
full allocation. Check every saved parameter for finite values, exact
checkpoint length and hash; compare resolved learner/game recipes against
the requested files. The independent offline auditor rechecks the receipts,
nonoverlapping native execution, world identities and timestamp-only choice.

The shared encoder source matches the earlier independent CUDA float64
smooth-positive/zero-weight checks at B1/3/8/64. That older scoped acceptance
does not independently qualify every ReLU branch, B256 gradients or the
complete learner; actual new training/evaluation is reported separately.
No system/CUDA/dependency changes or CPU neural computation are required.

## Commands for a fresh explicitly requested allocation

Run from `/home/claude/cnn` with an unused output directory:

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/forage_pixels_matched_time.py prepare build/FRESH_PACKET
.venv/bin/python research/forage_pixels_matched_time.py calibrate build/FRESH_PACKET > build/FRESH_PACKET/calibration-launch.log 2>&1
.venv/bin/python research/forage_pixels_matched_time.py run build/FRESH_PACKET > build/FRESH_PACKET/full-launch.log 2>&1
.venv/bin/python research/audit_forage_pixels_matched_time.py build/FRESH_PACKET
```

Do not automatically restart or extend a completed allocation. Native learning
is C/CUDA; these Python commands only prepare, supervise and audit artifacts.
Final publication work still needs the fixed seven training replicas, new
held-out worlds, and appropriate architecture/reference comparisons. Seeds
are never swept or replaced after seeing outcomes.
