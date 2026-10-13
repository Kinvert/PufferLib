# Realistic-image curve groundwork — RTX 5060

October 13, 2026. Worktree: `/home/claude/cnn-audit-20261008`.
This is temporary Kinvert research, not an upstream PR or a performance claim.
Historical allocations and the original checkout remain unchanged.

The new curve-feedback implementation uses eight fixed Nature checkpoint-time
anchors, each candidate's latest affordable checkpoint, and its own measured
initial policy when no trained checkpoint is affordable. It preserves declines
and separately reports every game, drawing, replica and Pong censoring bound.
Development and confirmation seeds are explicitly distinct and never swept.

Search now also requires both predeclared quality/Nature controls and measured
learning in every game. A qualifying control must reach the fixed final-score
floors and improve over its own initial policy on every development replica.
The normalized floors are mean ≥0.1, every seed ≥0.025, paired improvement
≥0.025. These are calibration filters, not statistical superiority tests.
Eligibility is reconstructed from the frozen calibration rows before launch
and again from the native artifact audit afterwards. An earlier high score,
high untrained return or successful subset of games cannot rescue the gate.

Measured runtime admission reserves a complete next panel plus final native
feedback replay and reporting. Mocked controller checks cover stopping before
the first panel and after two completed panels, preserving all observations
and launching no partial next panel. A rehashed, altered eligibility receipt
is rejected by scalar reconstruction. Actual overnight calibration remains
outstanding.

## Actual GPU checks

Hardware: G240, RTX 5060, driver 591.86, CUDA compiler 12.8.93, `sm_120`.
No system/CUDA dependency change or CPU neural reference was used.

- Nature RGB640×620, H128, B1/4/16: **18 cases / 72 native calls pass**.
  This checks layers/gradients, selected finite differences, parameter
  preservation and eager/graph/repeat/independent-process bytes.
- One-stage C8/K8/stride4/GAP/projection64, H128, all five distinct RGB image
  shapes: **90 cases / 360 calls pass**. Snake/Maze share the same input shape.
  Separate bounded allocations are independently audited and collected against
  the same frozen catalog/configuration/hardware, without repeating a completed
  GPU allocation.
- Two independent native Flappy trainings save finite full-policy weights at
  **0,64,128,192,256,320,384,448,512 decisions**, with byte-identical checkpoint
  hashes at every matching stage. There are 59,976 parameters. Process times
  are 2.019283315s and 2.019242834s; these few-second measurements include
  startup/checkpoint overhead and are not sustained throughput evidence.

The first full Pong Nature finite-difference allocation remains **failed**.
Its independent float64 oracle used a large uncompensated dense reduction.
A separate CUDA diagnostic reproduced the cancellation and motivated
compensated summation only in the oracle's loss-forward path. With the original
perturbation and tolerance, error fell from `1.56889e-5` to `2.85371e-8`.
The fresh full allocation above passes; it does not relabel the failed packet.
Production Nature kernels and acceptance tolerances are unchanged.

## Build and runtime scope

All six default RGB trainers and native host metadata compile/register. The
new `research/build_image_delivery.sh` prepares those builds without GPU work.
The initial launcher lost its venv packages by resolving the interpreter
symlink; a subsequent metadata command lacked the common module path. Both
failures are retained. The launcher preserves the venv interpreter path and
uses the ordinary panel import. Successful native builds were reused for fresh
host registrations; no GPU allocation was restarted.

The dedicated initial-checkpoint evaluator passes five episodes/four slots
with exact graph/repeat/eager/independent-tail rows in a 5.73002936s supervision
window. The separate all-six-game/all-41-drawing panel-v5 passes its raw audit:
six trainings, 54 finite checkpoints, 369/369 drawing/checkpoint evaluations
(41 initial-policy cells), and 12 exact repeat/eager checks. Campaign time is
599.80070706s; training process time totals 13.530211734s. Each job trains only
512 decisions, with one episode per drawing/checkpoint.

| Game | Process seconds | Process SPS | Parameters |
|---|---:|---:|---:|
| Connect4 | 2.574 | 199 | 60,616 |
| Pong | 2.523 | 203 | 60,104 |
| Flappy | 1.973 | 259 | 59,976 |
| Breakout | 1.468 | 349 | 60,104 |
| Snake | 2.421 | 212 | 60,232 |
| Maze | 2.571 | 199 | 60,360 |

Process SPS includes startup/checkpoint writes. Original native SPS/uptime arrays
remain in the metrics INIs; do not substitute these few-second measurements for
sustained throughput or compare them with historical tiny-image measurements.

Reporting admission now grows with the retained complete-panel count and slower
observed optimizer calls. Sixty-three focused host/scalar/mock tests pass after
the integration; those tests execute no neural reference or GPU work.

This does not establish full-policy numerical correctness, useful learning,
all-shape Nature qualification, an architecture advantage, generalization,
or a Pareto/SOTA result. The new balanced curve-feedback campaign still needs
real eligible learning controls and measured 5090 runtime calibration. No
long local campaign, final handoff push or 5090 launch occurred here.

Raw evidence: `build/image-panels-20261013/`; earlier failures and diagnostics
remain under `build/image-panels-20261012/`. Source revisions in build receipts
include the dirty worktree; do not call these binaries clean commit `66b52e6a`.
Owned source hashes and binary identities describe the actual tested state.

The new offline image reporter also replays the completed panel's raw native
audit and produces six process-SPS rows, 54 checkpoint rows, 110 unchanged native
metric bins and all-drawing interactive curves. It treats the initial policy's
measured startup/checkpoint cost separately from training throughput. Three
export tests plus nine related reserve checks pass. A future 5090 runtime pilot
must time these full exports/curves, not only its raw audit; that campaign route
has not executed on real eligible learning data.

Compact receipts and captured source are in `receipts/`; check their contents
with `sha256sum -c receipts.sha256` from this directory. Raw numerical arrays,
weights and binaries stay in their original ignored allocations. This compact
record does not replace those arrays for an independent numerical replay.

New paired-control preparation takes explicit fixed per-game learners/budgets
and both completed all-shape numerical manifests. It assigns all development
training/evaluation seeds and includes the measured initial policy. Offline
qualification keeps the full failed-learning report and only creates Nature
anchors after all-game eligibility. Four scalar/mocked preparation and
negative-qualification checks pass; the final focused suite passes 71 tests.
An initial inconsistent checkpoint-unit fixture and its failed log are retained.
actual meaningful control learning/qualification remains pending.
