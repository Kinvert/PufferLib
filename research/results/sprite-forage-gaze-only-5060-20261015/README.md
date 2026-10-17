# Gaze-only pixel ablation: faster, but learning collapses

Complete, RTX5060/G240, October15. Removing the automatic player-neighbor
view reduces collection perf from53.71% to4.88% in this single-seed pilot.
Keep the existing17-tile configuration as the default; gaze-only is an opt-in
research ablation, not a replacement or a successful architecture result.

| Pixel view | Train seconds | Decisions | Process SPS | Collection perf | Grape exits | Parameters |
|---|---:|---:|---:|---:|---:|---:|
|8 player neighbors + selected3x3, retained reference|295.053|4,587,520|15,548|53.71%|55.08%|103,976|
|Only selected3x3, new ablation|270.241|7,569,408|28,010|4.88%|8.20%|87,592|

The new run is80.15% faster in observed process SPS and receives65% more
decisions, yet collection falls48.83 percentage points and grape exits46.88
points. This is consistent with the automatic local view being important
under this learner; it does not prove learned gaze was unused, or that a
gaze-only policy cannot learn with different training. One training seed,
different parameter counts, decision counts and decision-space LR schedules
do not establish a general architectural ranking or an isolated visibility
effect. This is an unsuccessful quality tradeoff despite faster execution.

Both allocations aimed at300s using speed-only calibration and fixed decision
counts for scoped reproducibility. Actual new training is270.241s,29.759s
short of target; the reference is295.053s. Do not label them exactly matched
time. The new final checkpoint, at270.120s after process start, is also the
prospectively specified checkpoint nearest300s. No score-based checkpoint
selection or automatic extension was performed. The full new training/eval
campaign takes274.743s, excluding prior calibration/build/check work.

## What changed

`FORAGE_PLAYER_VIEW=0` removes the eight player-neighbor tiles completely
from partial state/pixel observations and the shared CNN batch. It sends
only nine tiles from the policy-selected gaze window, in row-major order.
An agent can still choose to look at itself, and gaze starts on the player
at reset. The player avatar is visible only when inside that chosen window.
No full-board CNN is computed and masked afterward.

The same full32x32 RGB tiles are copied from the same cached atlas. No image
downsampling or pretraining. Input becomes27,652 floats instead of52,228.
The shared per-tile convolutions/features are unchanged:4x4/s2/C8,
3x3/s2/C16, sixteen learned features per tile. The H128 readout shrinks from
276 to148 features, reducing encoder parameters from49,576 to33,192 and
complete policy parameters from103,976 to87,592. H128/L1 MinGRU remains.

Movement5 and gaze X20/Y15 actions stay identical, as do the four existing
player-position and relative-gaze geometry floats. This is restricted pixel
visibility, not removal of explicit position metadata. Rewards, game rules,
drawing0 and the inherited learner stay fixed: A64/H64/MB256, four threads,
one buffer, sync/float32, seed7. LR0.0141914776 anneals over each run's own
decision budget. The reference's full recipe is unchanged and verified.
The same flag also supports matching121-value one-hot state observations,
but no new state-policy training was done in this allocation.

## Evaluation and checks

Both final policies receive exactly128 assigned episodes at each development
suite seed420/1337,256 total,16 slots,512-step cap. Native manifests match
initial world/policy RNG and world hashes across both view configurations.
Each view's observation hash is checked against its own manifest; sample
selection is independent of episode completion order. These are reused
development worlds, not final publication tests. Seeds were not swept.

| Suite seed | Reference collection | Gaze-only collection | Reference grape exits | Gaze-only grape exits |
|---|---:|---:|---:|---:|
|420|52.73%|4.49%|60.16%|11.72%|
|1337|54.69%|5.27%|50.00%|4.69%|

New gaze-only ASan/UBSan checks pass24,000 matched world steps and all1,500
skin/gaze-center pairs: exact crops from the full render, borders, identities,
matching state/pixel geometry, hidden-cell isolation, gaze actions and RNG
independence. All seven default game/raster suites also pass; the normal
environment and default17-tile attention contracts remain intact.

Independent CUDA float64 forward/gradient checks pass ten fixtures at
B1/3/8/64/256, all33,192 encoder gradients and outputs, unchanged device
parameters and exact eager/two-graph-repeat bytes; the check process takes
1.13s. Tolerances remain6e-5 absolute +8e-4 relative. Fixtures are smooth-
positive and zero weights; arbitrary ReLU branches and complete learner
derivatives are not independently qualified. The test builder now propagates
captured gaze/geometry/one-hot/player-view flags. The default17-tile numerical
allocation retains its original eight-fixture scope for old launchers.

The fresh131,072-decision calibration policy passes exact graph/repeat/eager/
independent-tail episode rows before full training. The offline artifact audit
passes eight native training/evaluation processes,308 actual episode
executions and nine finite/hash/size-checked checkpoints. It checks the
reference binding, common learner/game recipes, exact worlds, complete
assigned episode IDs, nonoverlapping processes and timestamp-only selection.
The earlier reference was independently reaudited without GPU execution.

GPU work uses the shared reservation and idle checks; these do not certify
absence of every external contention source throughout training. Process
times include startup/checkpoint writes, exclude evaluations, and use
CLOCK_MONOTONIC. Native adjusted metrics remain in receipts. No system/CUDA/
dependency changes, main core overlay, CPU neural work, commit or push.

## Evidence and next diagnostic

Reference code/evidence commit: `47f966ba`. Ablation sources and binary hashes
are captured here. Raw packet: `build/sprite-forage-gaze-only-5060-20261015/`;
frozen build: `build/sprite-forage-gaze-only-build-20261015/`.
[Protocol and commands](../../FORAGE_GAZE_ONLY.md).
[Retained reference report](../sprite-forage-pixels-matched-5060-20261015/README.md).

Compact sources/build/test receipts, configs, exact episode CSVs and metrics
are preserved here with `SHA256SUMS`; weights/full training stdout stay in the
raw packet. This is not complete vendor/system/toolchain dependency closure.
Reaudit without GPU execution:

```bash
.venv/bin/python research/audit_forage_pixels_matched_time.py build/sprite-forage-gaze-only-5060-20261015
```

Read [audit.json](audit.json). Do not restart or extend this completed packet
automatically. A future fixed-player-gaze9-tile control and matching gaze-only
state policy could distinguish basic local navigation from learning where to
look; neither has been run. A separately declared fixed-decision comparison
would isolate sample budget more closely. Preserve this negative outcome.
