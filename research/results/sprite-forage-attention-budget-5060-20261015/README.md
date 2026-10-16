# Sprite Forage: short equal-time attention pilot

Completed October15 on RTX5060/G240, float32, fixed training seed7.
This targets the recent state baseline's11.246 seconds, not the older33M
state experiment. Kinvert subsequently requested five minutes per partial
policy; that is a separate fresh allocation, not an extension of these runs.

| Policy | Decisions | Train seconds | Process SPS | Eval perf | Grape exits |
|---|---:|---:|---:|---:|---:|
| Historical full state |8,388,608|11.246 mean|745,912|97.66%|100%|
| Fresh full-state control |8,388,608|11.850|707,891|97.66%|100%|
| Partial state + gaze |7,012,352|11.102|631,612|44.14%|40.63%|
| Partial pixels + gaze CNN |112,192|11.191|10,025|1.56%|9.38%|

Perf is the clamped collection score `(apples + grape - penalties) / 4`,
not a win rate. Grape exits alone can occur without collecting the apples.
Each new final policy is evaluated for32 sequential one-slot episodes at
each world offset420/1337, action seed7,512-step cap. The extra partial-pixel
repeat agrees on aggregate perf/exit rate and is excluded from its mean.

The partial policies land1.28% and0.49% below the time target; the full-state
reproduction lands5.37% above it. These are approximate equal-wall-time
pilots, not identical-duration experiments. The fresh full-state control's
four checkpoint files all match their historical counterparts byte-for-byte.
Its lower measured throughput therefore does not indicate changed learning.

Native adjusted-average SPS is735,014 /650,155 /10,350 for fresh full state,
partial state and partial pixels respectively; last-bin SPS is746,384 /
674,165 /10,379. These use the trainer's adjusted CLOCK_REALTIME timer;
whole-process SPS uses monotonic elapsed time including initialization and
checkpoint writes. The timer definitions remain separate.

## What changed between policies

Full state is the original one-head67,968-parameter policy, input141.
The partial-state policy has21 inputs and91,008 parameters. Partial pixels
use17 raw32x32 RGB tiles plus4 explicit geometry values,52,228 inputs and
137,896 parameters. The latter two share the two-head movement5/gaze300
action interface; full-state weights cannot load into that build.

All modes use the previously tuned state learner scalars and rewards,
including LR0.0141914776, gamma0.998099923, GAE0.850000024,
entropy0.00611945381, grape reward3.71950364, bad reward-1.68730032,
death reward-0.5 and676 training steps/episode. H128/L1 is fixed.
State uses256 agents/H64/MB2048/four environment threads; pixels use
8 agents/H8/MB64/two threads. Both use one buffer and synchronous training.
Those batching and information/action differences make this a system pilot,
not an encoder-only comparison. State-tuned settings are not established
as good attention/pixel hyperparameters.

The attention encoder shares a4x4/s2 C8 →3x3/s2 C16 →16-feature CNN across
the17 tiles, concatenates their features and4 coordinates, then projects to
H128. The recurrent core chooses movement and the next gaze center. It does
not first render/downsample the full board. Gaze behavior has not been
audited: partial-state learning alone does not prove useful learned looking.

## Evidence and interpretation

The full allocation took56.637 seconds: three main trainings, two separately
recorded speed calibrations and seven evaluations/224 episode executions.
Calibration time was2.120 seconds for state and2.019 for pixels; it is not
charged to the main training row. Scores were never used to choose budgets.
Fifteen checkpoint files have expected lengths and finite weights.
The independent offline artifact audit passes process nonoverlap, requested/
resolved configs, commands/log hashes, step counts, checkpoint hashes and
aggregate metric reconstruction. This audit does not execute or independently
certify the learner/evaluator's neural math.

Restricted state learns some collection behavior; the pixel policy has poor
evaluation performance in this tiny budget. At10K SPS it sees about63 times
fewer decisions than partial state in a similar duration. This does not show
a useful pixel policy, superiority, convergence or a Pareto-front improvement.
Independent training n=1; development world offsets are reused, not untouched
publication tests. Sequential worlds are reproducible because world RNG is
used only on reset, but there are no per-episode/action-trace receipts here.

Raw packet: `build/sprite-forage-attention-budget-5060-20261015/`.
Compact configs, metrics, commands and audit receipts are retained beside
this report; weights remain in the raw packet. Reproduce the **offline** audit:

```bash
.venv/bin/python research/audit_forage_attention_budget.py \
  build/sprite-forage-attention-budget-5060-20261015
```

Do not restart this completed allocation. The separately requested five-minute
experiment uses fresh policies and frozen budgets inferred from these speeds.
No GPU stack/dependency changes, CPU neural work,5090 launch, commit or push
occurred. All edits and new evidence are in `/home/claude/cnn`.
