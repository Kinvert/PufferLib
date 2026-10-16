# Sprite Forage: five-minute attention learning pilots

RTX5060/G240, October15, native float32. Kinvert requested about five minutes
after the preceding11-second comparison. These are **fresh policies**, not
continuations of the short checkpoints. Training seed7 is fixed for both;
there is no seed, architecture or hyperparameter sweep.

**Complete; offline artifact audit PASS.** Both trainings, all16 checkpoints
and the final-checkpoint evaluations passed. Because pixel training overshot
the intended duration, an additional timestamp-only checkpoint selection
gives the requested approximate five-minute comparison below.

## Results near five minutes

| Policy | Checkpoint elapsed seconds | Decisions | SPS through checkpoint | Collection perf | Grape exits |
|---|---:|---:|---:|---:|---:|
| Partial state |301.806|201,719,808|668,375|61.33%|76.56%|
| Partial pixels |303.072|2,332,032|7,695|5.47%|7.81%|

The full-state reference remains97.66% after about11 seconds. Restricted
state improves from the short pilot's44.14% to61.33%; restricted pixels
improve from1.56% to5.47% at the time-selected checkpoints. Pixel play remains
poor. These are fresh trainings with different LR annealing lengths, not a
continuation curve or a test of training duration alone. Nothing here
establishes useful learned gaze, a better CNN or a Pareto-front advantage.

Checkpoint time is native CLOCK_MONOTONIC **after atomic weight publication**
minus the Python monotonic process start. Target errors are+0.60% and+1.02%.
Times include initialization and earlier checkpoint writes, but exclude final
process teardown. They are not a strict300-second cap.

The selection rule minimizes absolute distance to300 elapsed seconds, with
earlier time breaking ties, across all eight scheduled checkpoints. It was
recorded after observing the pixel runtime overrun, before the supplemental
checkpoint evaluation, and never uses scores. State's selected point is its
final checkpoint, so its existing evals are reused. Pixels' selected point is
checkpoint6/8 and receives two extra32-episode evaluations. All checkpoint
timestamps and the explicit rule are retained in `time-matched/`; this is
development analysis, not a preregistered publication test.

## Complete outcomes, including the pixel decline

| Mode | Full process seconds | Decisions | Process SPS | Native adjusted-average SPS | Final collection perf | Final grape exits |
|---|---:|---:|---:|---:|---:|---:|
| Partial state |301.927|201,719,808|668,108|669,026|61.33%|76.56%|
| Partial pixels |377.147|3,109,376|8,244|8,253|3.52%|4.69%|

The pixel run took6m17s,25.72% over its intended target. The final score is
worse than the time-selected point; both are preserved. Last-bin native SPS
is10,517 for pixels and658,434 for state. Do not mix the native adjusted
CLOCK_REALTIME and monotonic process clocks.

Pixel checkpoint intervals are approximately37/37/37/60/73/58/37/37 seconds.
The slowdown is concentrated in the middle; its cause is not established.
Idle checks and our shared reservation do not prove absence of unrelated
external GPU/CPU contention throughout the job. Treat these as observed
costs, not a controlled intrinsic GPU-speed claim.

## Frozen allocation

| Mode | Decisions | Target seconds | Saved checkpoints | Parameters |
|---|---:|---:|---:|---:|
| Partial state |201,719,808|300|8|91,008|
| Partial pixels |3,109,376|300|8|137,896|

Budgets were fixed before either launch using the preceding measured native
steady SPS, estimated startup, and whole-rollout rounding. Scores did not
choose budgets. Actual monotonic process times must be reported; these are
approximate equal-duration runs, not hard stops at exactly300 seconds.
Compilation and the earlier speed calibration are separate costs.

The full-state reference is the already reproduced original one-head
8,388,608-decision policy:97.65625% collection perf /100% grape exits,
11.850 seconds in the fresh control (11.246 mean historically). It was **not**
trained for five minutes or silently loaded into the attention build.

Both partial policies have movement5/gaze300 outputs, H128/L1, and the same
17 visible cell identities/images plus4 geometry coordinates. State input
has21 values; pixel input has52,228, including17 native32x32 RGB tiles.
The shared tile CNN uses4x4/s2 C8 →3x3/s2 C16 →16 tile features, followed
by concatenation with geometry and a projection to128 recurrent features.
No full-board pixel image is generated before cropping.

## Learner, game and evaluation contract

Mode-specific settings are copied unchanged from the short pilot except
total training decisions, run/output locations and checkpoint cadence.
State uses256 agents/H64/MB2048/four environment threads; pixels use
8 agents/H8/MB64/two threads. Both use one buffer/synchronous training.

The historical tuned state learner is used for both: LR0.0141914776 with
annealing to zero over each run's complete budget, gamma0.998099923,
GAE0.850000024, entropy0.00611945381, replay1, clip0.2, value coefficient2,
value clip0.2, maximum gradient norm1.5 and momentum0.95. Attention/pixel
tuning has not established these as good settings. In particular this is
not the earlier smoke's LR0.001 and MB8.

The game is20x15, copies3/trees8, representation0, world RNG offset0,
training cap676, step cost-0.001, apple reward1, grape3.71950364,
other items0, bad items-1.68730032 and death-0.5. Both modes see the same
kind of game/rules; information and network/batching differ.

The primary panel evaluates the last scheduled checkpoint:32 sequential
one-slot episodes each at world offsets420/1337, action seed7,512-step cap.
The supplemental time-selected pixel checkpoint gets the identical suite.
All64 assigned episodes/checkpoint are retained. These offsets have already been used
for development, so they are not untouched final test data. The native
evaluator exposes aggregates; this packet does not establish per-action or
per-episode trace equality. World RNG is consumed only at reset, independent
of policy trajectories; the sequential board stream is reproducible.

Perf is clamped `(apples + grape - penalties) / 4`, not wins. Grape exits
are reported separately. The original full-state policy has access to more
information and a different one-head action/loss contract; comparisons cannot
isolate CNN overhead or prove learned use of the gaze action.

## Evidence

Raw packet: `build/sprite-forage-attention-five-minute-5060-20261015/`.
The supervisor holds the shared GPU lock and checks idle status before native
processes. Logs go to disk; no TUI stream or competing test workload is used.
Compact recipes/commands/native metrics/process receipts are preserved beside
this report, with weights retained in the raw packet. The primary campaign
took690.282 seconds including evaluations; the supplement adds two native
eval processes/64 episodes. Total: two trainings, six actual eval processes
and192 episode executions. The independent offline artifact audit passes:

```bash
.venv/bin/python research/audit_forage_attention_long.py \
  build/sprite-forage-attention-five-minute-5060-20261015
```

This checks all16 scheduled checkpoints for expected length/finiteness/hashes,
unchanged mode-specific learners, requested/resolved recipes, exact step
counts, process nonoverlap and evaluation summary reconstruction. The extra
timepoint audit reconstructs nearest-time selection from native receipts and
checks selected weight identities and reused/new evals. It does
not certify independent full-learner derivatives, generalization, useful gaze
behavior or a Pareto win. Independent training n=1. Preserve all checkpoints
and weak results; never select a best seed/checkpoint or automatically restart
a completed allocation. No system/CUDA/dependency change or push is needed.
