# Partial-state attention: matched action and observation ablations

RTX5060/G240, float32, October15. The first seven-job panel is complete;
its independent offline artifact audits pass. The one-hot and coordinate
interaction follow-ups are also complete, with results below.
No pixel training or downsampling was performed.

## First panel: same scalar object IDs, different gaze/geometry

All rows train from scratch for83,886,080 decisions, fixed seed7, the same
historical learner/rewards,256 agents/H64/MB2048, H128/L1, one buffer/four
threads, synchronous training. Every partial policy sees the same17 cells
and21 input values. Relative geometry replaces gaze world X/Y with signed
ROI-minus-player X/Y, retaining player world X/Y. No hidden objects enter.

| Gaze interface | Geometry | Train seconds | Process SPS | Collection perf | Grape exits |
|---|---|---:|---:|---:|---:|
|Absolute300|Absolute|123.824|677,462|64.26%|64.84%|
|Absolute300|Relative|126.309|664,135|12.21%|10.55%|
|Directional5|Absolute|85.203|984,543|61.82%|56.64%|
|Directional5|Relative|86.138|973,862|60.64%|53.91%|
|Factored20×15|Absolute|89.424|938,068|61.13%|66.41%|
|Factored20×15|Relative|102.379|819,369|64.16%|64.06%|
|Full state, absolute300 unused gaze|Absolute|149.832|559,866|99.41%|99.22%|

Directional5 is exactly noop/up/down/left/right, moves one cell and clips at
edges. Factored20×15 adds separate categorical X/Y heads so gaze can jump
anywhere in a step. Neither adds recenter/diagonal actions or free looks.
Player movement remains its own five-choice action. State parameter counts
are91,008 /53,248 /57,088 for the three partial interfaces; geometry does
not change counts. The full-state attention-build control has141 inputs and
106,368 parameters. It is not the historical one-head67,968-parameter policy.

The initial action/geometry changes do not clearly improve score here.
Smaller action heads provide higher observed process SPS at similar partial
scores. Relative coordinates can harm learning badly; they are not assumed
an improvement. All variants retain enough geometry to reconstruct absolute
gaze position. One training seed cannot establish a robust ranking.

All jobs have the same decision budget/LR schedule, rather than an exact
two-minute stop. They take85–150 seconds. Fixed decisions preserve scoped
training determinism; stopping by a varying wall clock would alter counts.
Cost is measured separately with monotonic process receipts including startup
and eight checkpoint writes. This is not an exactly matched-time comparison.
Idle checks/shared reservations do not certify uncontended GPU/CPU execution
throughout a process; no intrinsic hardware-speed or Pareto win is claimed.

## Matched exact evaluation

Each final scheduled checkpoint receives128 episode IDs at each suite seed
420/1337:256 episodes,16 slots,512-step cap. These are the same generated
boards for every variant, with explicit per-ID world/policy RNG and reset
recurrent state. Every assigned result is retained, independent of finishing
order. Native scalar manifests and actual episode start/world hashes match
across every case. CSVs include all-head action hashes and exact hexadecimal
perf/returns. The evaluator copies and validates both/all three action heads.

Perf is clamped collection fraction `(apples + grape - penalties)/4`, not
wins. Grape exit alone is not a perfect episode. Evaluation worlds have been
used for development; they are not untouched final publication tests.

Training settings include LR0.0141914776 annealed over the common budget,
gamma0.998099923, GAE0.850000024, entropy0.00611945381, replay1, clip0.2,
value coefficient2, value clip0.2, max gradient norm1.5, momentum0.95.
Training cap676, copies3/trees8, apple reward1, grape3.71950364,
bad-1.68730032, death-0.5, other0 and step-0.001. Changing head counts
changes maximum entropy and parameter count; equal coefficients do not mean
identical exploration. These are the inherited full-state settings, not
proven optimal partial-observation learners.

## Acceptance evidence

Six ASan/UBSan action/geometry suites pass144,000 matched game steps and
9,000 skin/gaze-center cases, explicit edge clipping/XY action checks,
full-frame crop parity, geometry and hidden-cell isolation. Seven normal
environment suites also pass. Native GPU smoke passes seven65,536-decision
trainings,28 exact eval processes/364 episode executions. Repeated, eager
and independent final-wave replay rows match exactly. A fresh original/new
control training produces identical65,536-decision checkpoint bytes.

The full panel passes seven trainings,14 eval processes/1,792 assigned
episodes, all56 scheduled finite/checksummed checkpoints, source/binary/
recipe binding, native parameter lengths and nonoverlapping process receipts.
Campaign time777.805 seconds includes evaluation. No failed/weak run or
checkpoint was replaced. No system/CUDA/dependency change, CPU neural model,
5090 launch, commit or push was needed. These are scoped behavior/artifact
checks, not independent full-learner derivatives or statistical superiority.

Raw: `build/sprite-forage-state-ablations-5060-20261015/`.
Build/use: [FORAGE_STATE_ABLATIONS.md](../../FORAGE_STATE_ABLATIONS.md).
Do not restart this completed allocation.

## Next observation ablation: categorical identity

The follow-up changes only the visible object representation: each of the
same17 cells gets a13-way one-hot category instead of a scalar ID divided
by12. Four existing absolute geometry values remain; input length becomes
225. This adds no reward values, hidden targets or nearest-object helper.
It avoids treating object IDs as ordered numerical quantities. Input/encoder
size and cost increase and must be reported rather than called identical.

Test all three gaze interfaces with absolute geometry, matched to their
existing scalar controls at the same83,886,080 decisions and exact256-world
suites. One-hot/relative-coordinate combinations are not in this first
follow-up. Preserve failures and all declines. This remains seed7 development
work; broader claims require the predetermined seven training replicas and
new evaluation worlds, never a seed sweep.

## One-hot and relative-coordinate outcomes

| Gaze interface | Geometry | Input | Parameters | Train seconds | Process SPS | Collection perf | Grape exits |
|---|---|---:|---:|---:|---:|---:|---:|
|Absolute300|Absolute|225|117,120|135.649|618,405|82.23%|57.42%|
|Directional5|Absolute|225|79,360|96.559|868,756|75.49%|55.47%|
|Factored20×15|Absolute|225|83,200|96.064|873,234|85.35%|61.72%|
|Factored20×15|Relative|225|83,200|97.307|862,074|**93.55%**|**90.23%**|

All use the same learner/rewards, seed7,83,886,080 decisions and256 exact
episode worlds as the original panel. Object identity changes, but the visible
cells and information do not. One-hot increases input width and encoder
parameters; the factored head still leaves the complete policy smaller than
the original91,008-parameter scalar/absolute300 policy. These are observation/
action ablations, not identical parameter budgets or CNN architecture results.

One-hot improves collection perf by17.97 /13.67 /24.22 points with the three
absolute-geometry controllers. After this result, a deliberately separate
paired interaction test changes only geometry in the factored one-hot case:
85.35% becomes93.55%. The relative change previously harmed the scalar300
case and barely helped scalar factoredXY; its effect depends on encoding and
control. Preserve all those negative/mixed results, not just the final case.

The selected candidate improves the original partial control by29.30
collection points and25.39 grape-exit points, with lower observed elapsed
time (97.3s versus123.8s). It remains below full-state99.41% collection /
99.22% exits. Collection perf is not win rate: candidate grape exits are
90.23%. Useful active gaze versus a random/fixed-gaze controller is still
unproven. This is a promising partially observed state policy, not mastery,
cross-seed superiority, a CNN result or a publishable Pareto claim.

Follow-up packet: `build/sprite-forage-state-onehot-5060-20261015/`.
It passes three trainings, six evals/768 episodes and24 scheduled finite
checkpoints; campaign334.966 seconds. The coordinate interaction packet is
`build/sprite-forage-state-relative-5060-20261015/`, one training/two evals/
256 episodes/eight checkpoints,99.452 seconds. Their preceding smokes pass
three/one trainings,12/four graph-repeat-eager-tail evals and156/52 actual
episode executions. Three additional one-hot sanitizer suites plus one
one-hot-relative suite check visible identities, geometry, crops and RNG.

The interaction and one-hot trials were chosen during development after the
initial results. This is adaptive engineering, not preregistered confirmation.
All weak cases remain. [Paired per-world differences](paired.json) use the
same IDs/world hashes, but cannot substitute for independent training seeds.

### Reproduction and next gates

The candidate's complete same-seed training was repeated from scratch in
`build/sprite-forage-state-relative-repeat-5060-20261015/`.
**All eight checkpoint hashes and all256 episode rows match exactly.**
This is still one independent training seed, not two replicas.
The candidate scores higher than the old control on191/256 matched worlds,
ties52 and scores lower on13; mean improvement29.296875 collection points.
These paired development-world counts are descriptive, not training-seed
confidence bounds. Details are in `paired.json`.

Across the initial panel, one-hot follow-up, interaction and same-seed repeat:
12 complete long-budget trainings (11 distinct cases),12 short canaries and
one frozen-binary compatibility training. There are97 native processes and
3,696 actual episode executions including repeated canaries/evaluations.
All96 long-budget checkpoints,12 canary weights and the legacy-control
weight are retained. Every full/smoke artifact audit passes; no candidate
is silently retried or weak result discarded. Exact repeat is a separately
declared new allocation, not a replacement run. Compact receipts, native
metrics/CSV rows, captured sources/build flags and audit tools are preserved
beside this report; raw policies remain under build/. No full toolchain or
vendor/system dependency closure is certified by this compact copy.

First confirm the selected candidate against the retained baseline using all
seven predetermined training replicas and new fixed evaluation worlds before
claiming reliable superiority. Do not optimize seeds or replace bad replicas.
Fixed/random-gaze controls would separately test whether learned ROI choices
are useful. Downsampling has intentionally not begun; the full human renderer
and all32x32 pixel tiles remain unchanged. Future image reduction must be
precomputed as Kinvert requested, after these state ablations.
