# Learning recipes before the multi-game comparison

October 6 [memory follow-up](LEARNER_MEMORY_PREFLIGHT.md): actual native
descriptors require more than 16 GiB for the stock minibatch's IMPALA/Impoola
training tensors alone. Unchanged stock batch is therefore unavailable on the
5060. A shared stock-except-minibatch-2,048 candidate is prepared, with 64
updates/rollout and 9,728 total, preserving graphs/core/world/actor settings.
It isn't the common recipe or proof of full-policy fit/learning. Keep separate
candidate frontiers and both GPU holds; no policy ran.

October 6, 2026. **Configuration arithmetic and experimental design only.**
Both GPU holds remain. None of these checks ran a policy, optimizer, CPU model,
training, GPU query or dataset generator. The frozen encoder is still trained
from random weights separately on each game; no pretraining is included.

Follow-up: [shared learner preparation](SHARED_LEARNER_RECIPES.md) now implements
the explicit per-game overlay needed for these candidates. Two Flappy panels
prepare 48 jobs each at the aligned 19,922,944-decision/19-checkpoint budget;
native host starts are paired across models, drawings and recipe allocations.
No learning ran. The larger stock minibatch still needs memory/math/reload
qualification for every baseline before a timed campaign.

## What the native learner actually counts

The compiled C helper `research/learner_geometry.c` uses the existing native
INI parser and reads full configurations/ordered overlays. For the supported
single-policy, no-selfplay float32 study:

- Rollout decisions per rank are `agents * horizon`.
- Train epochs are integer `requested / world_size / rollout_decisions`.
  Actual global decisions are whole epochs times rollout size and world size.
- Optimizer minibatches per epoch are the native **float32** replay product
  divided by minibatch size, truncated to `int`. The decimal replay value alone
  is not the effective number of updates or passes through the rollout.
- Minibatches traverse deterministic contiguous row slices, wrapping from the
  beginning. An incomplete pass covers a **prefix**, not a random subset.
- The helper reports local optimizer-update counts, processed transition
  usages (including replay), coverage, checkpoint cadence and sampled cosine
  LR endpoints. These aren't unique frame counts or measured GPU work/time.

The audit snapshots current `src/pufferl.cu`, `src/ini.h`, helper sources and
configs. Source markers guard the reviewed formulas, but aren't a complete
control-flow/compiler proof. The helper rejects unsupported geometry, invalid
scalars and unsafe native integer counts; it explicitly reports zero updates
or zero epochs rather than inventing learning. CPU scalar arithmetic is not a
neural execution substitute. Distributed arithmetic fixtures do not qualify
distributed runtime behavior.

## Findings from actual configurations

These are **current stock state configuration values**, not new stock training
results. SnakeBench is a separately versioned local-state control, not untouched
stock multi-agent Snake. Stock and fixed-CNN cores/vectorization differ; this
table does not compare their scores or isolate encoder overhead.

| State control | Agents | Horizon | Minibatch decisions | Updates/rollout | Effective replay | Actual configured decisions |
|---|---:|---:|---:|---:|---:|---:|
| Connect4 | 4,096 | 32 | 8,192 | 50 | 3.125 | 13,238,272 |
| Pong | 1,024 | 64 | 16,384 | 16 | 4 | 7,995,392 |
| Flappy | 2,048 | 64 | 16,384 | 8 | 1 | 19,922,944 |
| Breakout | 4,096 | 32 | 65,536 | 3 | 1.5 | 54,919,168 |
| SnakeBench | 64 | 32 | 2,048 | 1 | 1 | 13,312,000 |
| Maze | 512 | 256 | 16,384 | 22 | 2.75 | 337,903,616 |

The prepared fixed-CNN panel instead uses 64 agents, horizon 32, minibatch
2,048, replay 1: one update per 2,048 decisions. All **152 prepared jobs**
match effective geometry across the four encoders, have positive updates and
exactly their requested 65,536 decisions. That verifies fairness of the
plumbing panel, **not sufficient learning**. Raw `compare.ini` templates have
their own inherited budgets; the panel's explicit override is what makes its
budgets uniform. Do not confuse the template with its resolved job INI.

The executed Flappy quality/state INIs reproduce their retained pilot's
19,922,944 decisions, 152 rollouts, 19 checkpoints and **1,216 optimizer
updates**. At the same decision budget, the small-batch common recipe would
perform 9,728 updates—**eight times as many**, with different minibatch, LR,
gradient clipping and actor settings. A comparison with stock settings must
label these differences; equal decisions do not imply the same training regime.
Those original pilots remain single-seed, pooled-v1 evidence with unequal
state/pixel information and cores.

At the current 64-slot geometry, replay below one can produce **zero** optimizer
minibatches. Near a boundary, float32 rounding matters: decimal `0.99999998`
rounds to float32 one and produces one update, whereas independently flooring
the decimal gives zero. Later calibration must record effective counts and
reject unintended zero-update learning configurations **before launch**.
Retain any historical zero-update trial and its outcomes as an explicitly
untrained control; don't delete it from the recorded observed frontier. Do not
silently repair a historical recipe or label it an efficient learned policy.

The historical Pong A/B replay values would yield one/three updates at this
**current** geometry. This is a candidate-reuse arithmetic fact, not an audit
of the remote historical optimizer implementation. [Pong results](PONG_5090_RESULTS.md)
already show major recipe interaction: Nature substantially exceeded ours
under B, while Impoola reliably learned under A and failed under B. Retain both
and the missing evaluations. Choosing only a favorable recipe isn't robustness.

## Concrete next experimental sequence

1. Pass the five pending [exact-evaluator runtime gates](EVAL_ACCEPTANCE.md) in
   an explicitly scheduled window. Use existing Flappy weights first; no
   pilot relaunch or new training is needed for those available cases.
2. Start learning calibration with Flappy, fixed drawing 0 and all four
   **unchanged** encoders/H128-L1. The already successful stock Flappy learner/
   vectorization recipe is an evidence-backed candidate; the small-batch common
   recipe is a different candidate, not a validated cheaper equivalent.
   Each candidate must be crossed with every family and paired development
   seeds, with full executed INIs and independent assigned-ID evaluation.
3. If comparing those two geometries at equal decisions, explicitly request
   **19,922,944** for both: it is divisible by both rollout sizes. Keep native
   schedule/rounding semantics. Stock learner/core substitution must remain
   explicit, and all CNNs within a candidate receive that exact substitution.
   Their optimizer counts may differ **between candidates**, never secretly
   between encoders in one candidate. Align checkpoint steps to the common
   rollout multiple; retain all declared checkpoints/declines/failures.
4. For Pong, inspect saved development policies with the accepted replacement
   evaluator and calibrate duration-only caps before new comparative training.
   Keep A/B and the failed-learning .001-LR plumbing history visible. Source/
   seed/cap/evaluator revisions belong to separate records, not revised scores.
5. For Breakout and Maze, treat current stock learner recipes as starting
   candidates, not proven pixel recipes. Maze's horizon 256 and long budget
   contrast with the plumbing horizon 32; reducing both does not establish
   equivalent learning opportunity. Compare fixed-core CNNs under shared
   per-game recipes, preserve their original native games, and keep verified
   Maze level exclusion separate from drawing robustness.
6. For Snake, calibrate the declared one-agent SnakeCNN/SnakeBench protocol;
   don't import original multi-agent stock settings and call it the same task.
   Connect4 appearance comparisons retain corrected rules and the accepted
   assigned-episode evaluator, with all ten fixed drawing IDs reported.
7. Keep development calibration separate from final inference: equal declared
   tuning allowances, retained recipe outcomes/compute, fresh paired seeds and
   a predeclared stopping/analysis rule. A common recipe gives a controlled
   comparison; an independently tuned family comparison is a separate claim.
   No selection on favorable test checkpoints, drawings or seeds. Report each
   shared-recipe full frontier before any descriptive multi-recipe envelope.

This specifies the path, **not a newly authorized campaign**. Candidate counts,
development seeds, sufficiently long budgets, administrative caps and final
selection/inference rules still require freezing before GPU execution. A
bare config, stock tuning or build pass doesn't certify learning adequacy.

## Reproducible GPU-free tooling

```bash
# Native scalar helper; no CUDA/Torch/NCCL dependency.
mkdir -p build
cc -std=c11 -O2 -Wall -Wextra -Werror -ffp-contract=off \
  research/learner_geometry.c -lm -o build/learner_geometry
build/learner_geometry config/default.ini config/flappy.ini

# Source/config snapshots, host C build and per-recipe arithmetic audit.
.venv/bin/python research/audit_learning_recipes.py prepare \
  --panel build/pixel-robustness/FRESH_PANEL/protocol.json \
  --out build/learner-geometry/FRESH_ID

# Artifact inspection, no native command or executable transfer required.
.venv/bin/python research/audit_learning_recipes.py inspect \
  --audit build/learner-geometry/FRESH_ID/audit.json

.venv/bin/python -m unittest research.tests.test_learner_geometry -v
```

The optional panel input requires matching current learner/parser source;
different historical implementations need their own reviewed arithmetic.
Outputs include full configs/settings, `records.json`, `geometry.csv`, source/
compiler/binary hashes, original input receipts and explicitly false learning/
publication flags. Existing directories are refused; failures remain.

[Retained receipts](results/learner-geometry-20261006/README.md): 166 actual
configuration cases including 152 paired panel jobs; 10 tests pass, covering
hand-calculated controls, archived Flappy counts, float32 replay boundaries,
prefix coverage, schedule/cadence/rounding, invalid/overflow cases under UBSan,
source changes, and a 304-job two-seed fixture. No new score/SPS/frontier exists.
