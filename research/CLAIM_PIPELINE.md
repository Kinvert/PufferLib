# Connect4 measurement pipeline: groundwork, no GPU execution

October 5: [Connect4 draw semantics are corrected](CONNECT4_DRAW_CORRECTION.md).
Prepare fresh campaigns; manifests now record `connect4-full-board-draw-v2` and
reports display the rule label. Unversioned old preparations cannot be built or
executed with current tooling. The exact evaluator/auditor also reject early
draws. Historical build receipts below describe the earlier source, not GPU
acceptance of this update. The 5090 remains busy; no GPU launch is requested.

October 2, 2026. Goal: a reproducible full score-versus-training-time comparison
that Joseph and a paper reviewer can inspect. **This is prepared infrastructure,
not new evidence of an encoder advantage.** No 5060 or 5090 training/evaluation
was run for this work. See [the claim protocol](CONNECT4_CLAIM_PROTOCOL.md),
[paper plan](PAPER_PLAN.md), and [current priorities](../potential-todos.md).

## Native measurement changes

`eval_exact` is an opt-in Connect4CNN native command. Its implementation lives
in `ocean/connect4cnn/exact_eval.cu`; `src/pufferl.cu` supplies the dispatch and
uses the existing policy forward, sampling, recurrent reset and checkpoint load
paths. It supports the four existing encoder builds and does no optimization.
Historical `eval` and its pooled-v1 results remain separate.

An episode is identified by `(block_seed, episode_id)`. Fixed-size waves assign
one game per slot; completed slots never step again in that wave. A partial last
wave has inactive slots whose outputs are ignored. Each episode gets a recorded
environment RNG seed and its own Philox policy seed, derived from identity rather
than completion order. Waves start with terminal flags set, exercising the native
recurrent reset before the first action. The complete CSV records identity, slot,
appearance, both RNG seeds, decisions, outcome, invalid-action status and action
sequence hash. Games must terminate within 21 decisions. Missing records are
errors, never losses. The environment's original rules/opponent are unchanged.

Evaluation preserves stochastic sampling, including the original ability to
choose a full column and lose. It does not add a legal-action mask or change to
greedy play. The primary protocol fixes appearance 0. Other appearances require
their own matched panels and auditor support. RNG mapping is versioned in
`exact_protocol.h`; glibc `rand_r`, compiler/cuRAND versions and fixed batch shape
are part of reproducibility. Environment seeds have 32 bits and can collide;
episode identity remains unique. No cross-platform or different-batch-shape
bitwise guarantee is claimed.

The evaluator creates one native buffer, sets horizon 1 and evaluation minibatch
allocation to the slot count, disables async, and loads the existing policy.
These are evaluation-only settings; every model uses the same settings. Existing
CPU worker threads remain waiting while the evaluator steps assigned environments
directly. GPU runtime validation must verify this synchronization/reset behavior.
This implementation is deliberately outside the general learner while its
environment-specific measurement contract is under review.

`PUFFER_CHECKPOINT_RECEIPTS=1` emits a native `PUFFER_CHECKPOINT` line after the
weight download, complete write, checked close and successful atomic rename.
It contains steps, bytes and `CLOCK_MONOTONIC` nanoseconds. The launcher records
the same clock immediately before spawning training. Their difference includes
startup, graph capture, training and checkpoint writing. It describes availability
of the complete file, **not crash-durable fsync completion**. The ordinary native
uptime/SPS timer remains separate and is not silently relabeled. Checkpoint
receipt printing is opt-in and its overhead still needs measurement.

## Preparation, execution and audit

`ocean/connect4cnn/claim.py` is external configuration/process/reporting glue.
It does not implement a Python learner or optimizer. PROTEIN/native architecture
construction is unchanged. Run commands from the checkout with the existing
Python 3.12 research venv. No packages or system CUDA changes are needed.

Safe preparation, requiring no GPU/runtime setup:

```bash
.venv/bin/python ocean/connect4cnn/claim.py prepare --canary \
  --out build/connect4cnn/claim-canary-NEW_ID
```

This writes an immutable-hashed protocol, captured source files, complete per-job
INIs and planned order. The canary has four locked encoders (quality, Nature,
IMPALA, Impoola), two development training seeds, 65,536 decisions and two
checkpoints per run. Every checkpoint gets two evaluation blocks of 65 and 8
assigned games: the 65-game block tests wave rollover and a partial final wave at
64 slots. These small budgets establish plumbing only, never ranking/learning.

The locked quality model is the existing encoder 4, 16-channel 7x7/stride-4 model
with projection 64, **not an unvalidated encoder-5 search winner**. All models
use the shared `compare.ini`, H128/L1, float32 and identical learner settings.
Model order rotates over paired training-seed blocks; every position is balanced
in each complete block of four seeds. The two-seed canary is not fully balanced.

For a **draft only** of the eventual 51-checkpoint, 13.312M-decision panel:

```bash
.venv/bin/python ocean/connect4cnn/claim.py prepare \
  --seeds 48001 48002 48003 48004 \
  --out build/connect4cnn/claim-draft-NEW_ID
```

The four seeds above are an example for preparation, not the chosen publication
sample size. Every 262,144 decisions plus the final checkpoint gives 51 points.
The same full-horizon LR schedule applies to all; early checkpoints are not
separately annealed short runs. The script **refuses to launch any full draft**.
Replication/inference and measurement gates still need validation and freezing.

When the 5090 is available and Kinvert wants bounded GPU validation, safely update
the existing Kinvert checkout, preserve divergent work, source that machine's
existing runtime, then prepare a fresh campaign there. Do not transfer/reuse G240
build outputs or run the following execution command now:

```bash
source build/connect4cnn/runtime-5090.sh
export NVCC_ARCH=sm_120 OPENBLAS_NUM_THREADS=1
.venv/bin/python ocean/connect4cnn/claim.py build \
  --out build/connect4cnn/claim-canary-NEW_ID
.venv/bin/python ocean/connect4cnn/claim.py run-canary --allow-gpu \
  --out build/connect4cnn/claim-canary-NEW_ID
.venv/bin/python ocean/connect4cnn/claim.py audit \
  --out build/connect4cnn/claim-canary-NEW_ID
```

`build` compiles all four binaries using ordinary `build.sh`, records compiler
and binary identities, and rejects changed captured sources or compiler override
flags. It does not execute a model. `run-canary` requires explicit `--allow-gpu`,
checks for a single idle 5090, and refuses existing execution receipts. It runs
serially, with process-group timeouts, records every command/config/log, and
stops on failures without deleting them. Current idle checks are snapshots, not
a continuous exclusivity guarantee. The 5090 needs an exclusive reservation for
scientific timing. Execution has **not** been tried yet.

Each completed training job's final checkpoint additionally receives three
recorded repeat checks: unchanged execution, eager execution, and one CPU worker
instead of two. All must produce byte-identical episode CSVs at the fixed slot
count. These checks exercise saved-model evaluation/reset behavior, not full
same-seed training repeatability or all-gradient correctness. Those remain
separate requirements in [the math verification contract](FLEX2_VERIFICATION.md).

Audit rejects changed sources/configs/binaries/checkpoints/episode receipts,
wrong commands or loaded-checkpoint paths, missing/duplicate IDs, wrong RNG seeds,
wrong appearances, wrong terminal counts and impossible completion times. It
compares native-resolved learner settings against frozen INIs using parsed double
values (`%.17g` decimal spelling changes are not real hyperparameter changes).
It retains independently complete checkpoint cells from a partially failed job.
Missing seed outcomes never become selectively averaged complete-case means.

Every audit writes a **new** `analysis.*` directory with raw observations, complete
checkpoint means, missing cells, failure details, timing costs, Markdown tables,
interactive whole curves and analysis-source receipts. No old report is
overwritten. Build, training and evaluation/repeat-check costs are separate;
failed subprocess time is retained when measured. The plot preserves declines,
all observed budgets, and both time and decision axes. Connected points are a
visual guide, not interpolated policies. Existing historical results cannot be
fed in as though they used this exact evaluator/timing protocol.

## Statistical gate: intentionally still closed

`research/claim_frontier.py` implements descriptive frontiers and an experimental
joint bootstrap: resample entire paired training-seed blocks, keeping every
model/checkpoint/time/score together; use the maximum standardized deviation over
all coordinates to form one family of bands. Frontier membership is recomputed
per resample. Zero-variance score/time cells get the whole feasible range rather
than zero-width certainty. These bands and membership frequencies do not certify
dominance, and the report never claims otherwise.

The development simulation explicitly includes null, crossing, tied and declining
curves. A small initial diagnostic (40 synthetic experiments per scenario, 40
seeds, 250 resamples) obtained joint coverage **90%, 87.5%, 85%, 82.5%** against the
nominal 95%, respectively. Monte Carlo standard errors were approximately
4.7–6.0 percentage points. No false strict dominance was observed in these small
simulations; this does not rescue the coverage failure or establish a guarantee.
See [the retained diagnostic](results/connect4cnn/claim-groundwork-20261002/statistics-development.json).

The current candidate therefore remains unsuitable as the publication inference
method without further work. Do not increase real training seeds until a desired
p-value appears. Improve/calibrate the analysis using development-only simulations
and historical variance, with independent simulation validation, failure mixtures,
hardware timing drift, both axes and the full proposed 204-point panel. Freeze
practical effect size, final method, sample size and stopping/failure rules before
any new confirmation outcomes. The present toy simulations have only 20 points
and cannot validate the proposed publication panel.

Primary methodological context: [Agarwal et al., 2021](https://arxiv.org/abs/2108.13264)
emphasize uncertainty in few-run RL evaluation. [Chernozhukov, Chetverikov and
Kato](https://arxiv.org/abs/1412.3661) establish high-dimensional bootstrap results
under stated assumptions. Neither paper automatically validates this custom
finite-sample implementation or these game/hardware distributions.

Synthetic statistical development can run without a GPU or neural model:

```bash
.venv/bin/python research/claim_frontier.py --calibrate \
  --repetitions 200 --seeds 40 --draws 1000 \
  --out build/connect4cnn/claim-statistics-NEW_ID.json
```

## What has actually been checked

- Native Connect4CNN trainer/evaluator compilation with the existing CUDA 12.8
  toolchain and explicit `sm_120`; no GPU execution.
- Eleven artifact/configuration checks, including real native INI round-tripping
  and unsigned RNG boundary agreement, exact quota corruption, failed/timeout
  process receipts, matching settings, 51-point draft cadence, whole archived
  synthetic results and preservation of partial results.
- Twelve interactive chart state combinations plus empty data, using synthetic
  fixtures. This checks JavaScript behavior, not browser appearance.
- The preliminary synthetic inference diagnostic above, preserved despite its
  unfavorable result. It is not a training result or acceptance certificate.

Run preparation checks with:

```bash
.venv/bin/python ocean/connect4cnn/tests/test_claim_tools.py
node research/tests/test_claim_chart.cjs
```

Remaining GPU gates: finite/all-gradient and legacy model regressions, exact
episode/reset/RNG/reload behavior, eager/graph/worker repeatability, same-seed
training, checkpoint timing overhead, baseline backend efficiency and exclusive
resource timing. Pong's evaluator is not repaired by this Connect4-specific
implementation. Cross-task transfer and an external benchmark still precede a
broad pixel-RL/SOTA claim. No new GPU campaign, upstream push or message to Joseph
occurred during this preparation work.
