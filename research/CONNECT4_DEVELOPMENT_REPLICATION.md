# Exact full-frontier measurement and paired development replication

October 5, 2026. Kinvert authorized completing the existing full-frontier
reevaluation, a bounded timing/recovery canary, and fresh paired ours/Nature
replication on the RTX 5060. This is development evidence. Publication
confirmation, inference calibration, baseline efficiency review, other
renderings/tasks and encoder-5 qualification remain separate gates.

## Completed: all existing curves scored fairly

The four corrected-rules models from `compare.a2lt6qft` were reevaluated at
**all 13 saved checkpoints each**, using the frozen seed-51005 suite of exactly
1,000 games. All 52 evaluations completed; all 52,000 per-game records and
checkpoint/build/suite hashes pass an independent audit.

[Audited whole frontier](results/connect4cnn/exact-frontier-20261005/analysis.audited/REPORT.md)
and [interactive curves](results/connect4cnn/exact-frontier-20261005/analysis.audited/curves.html)
retain every chronological point, including ties, declines and dominated points.
Ours supplies most of the faster observed frontier; IMPALA supplies the higher
scores. The first tiny Nature score creates an early frontier point as well.
This describes one training seed and does not establish superiority.

Final exact-suite scores: quality 90.2%, Nature 77.0%, IMPALA 97.0%, Impoola
81.1%. Training was not repeated. Original pooled-v1 observations are retained
separately. Training-time coordinates remain the original **approximate
launch-to-file-mtime times**, not newly precise monotonic measurements.

A report writer initially rejected an extra checkpoint-hash field after all
evaluations had completed. The field/schema issue was fixed and a separate
GPU-free audit regenerated the report from retained episode evidence. No policy
was rerun for that reporting error; the failed report/log remain preserved.

## Completed: native checkpoint timing and failure recovery canary

The two existing binaries were rebuilt with the normal CUDA/NCCL setup and
float32. Sixteen short training runs used four paired development seeds, both
models, and receipts on/off in alternating order, at 262,144 decisions and
eight checkpoints per run. Each of the 128 checkpoint files had the same bytes
as its matching on/off control. Post-rename receipt times were strictly ordered,
within the launcher process interval, and paired with valid finite weights.

[Timing gate](results/connect4cnn/timing-canary-20261005/timing-gate.json):
median on/off process-duration ratios were 1.01869 for quality and 1.02659 for
Nature. Individual ratios vary; this short canary screens for large overhead,
not a precise overhead confidence interval. The predeclared practical screen
was a median ratio no greater than 1.10 for each model.

The extra interrupted fixture was killed immediately after the first completed
checkpoint receipt. Its training remains `failed`; the first 32,768-decision
checkpoint, post-rename time, finite weights and native startup config survived.
It completed the assigned 65-game exact evaluation. Later missing checkpoints
and final metrics were not invented. [Recovery receipt](results/connect4cnn/timing-canary-20261005/interrupted/recovery.json).

Native training now writes an opt-in atomic `resolved.ini` beside checkpoints
before training when `PUFFER_CHECKPOINT_RECEIPTS=1`. Default runs retain normal
behavior. Config persistence permits checking a partial run that never reaches
the final native log dump. `claim.parse_checkpoints(..., allow_partial=True)`
accepts only a correctly ordered completed prefix of the declared cadence;
malformed, duplicate, skipped, wrong-sized and out-of-process receipts fail.
Successful runs still require the complete declared sequence. Partial-run
training SPS is left unavailable rather than divided using the planned budget.

## Completed and independently audited: paired development replication

All ten training jobs and all 510 checkpoint evaluations completed, with zero
missing cells or failure records. The independent archive audit checked all
510 finite, correctly sized checkpoint files and the exact 510,000 assigned
episode outcomes. No sampled GPU contention/telemetry failures were recorded.

| Model | Mean final wins | Mean train process seconds | Mean process SPS | Mean native SPS |
|---|---:|---:|---:|---:|
| Quality Flex | 76.94% | 183.382 | 72,712 | 72,891 |
| Nature | 71.36% | 202.153 | 65,866 | 66,014 |

Each row averages five paired training seeds at 13.312M decisions. Ours used
9.29% less mean process time and had a 5.58 percentage-point higher final mean.
It had higher final wins on four of five seeds, but lost by 23 percentage points
on seed 53101. Paired final score differences (ours minus Nature) were -23.0,
+13.4, +3.4, +20.8 and +13.3 percentage points. Seed variability is substantial;
this five-seed development panel does not certify quality superiority or joint
frontier dominance.

Across the whole observed mean time frontier, 38 of 39 nondominated checkpoints
belong to ours. The one Nature point is an early 0.08%-win checkpoint. Every
chronological/seed curve, decline and dominated point is retained. Candidate
confidence bands remain disabled; frontier membership here is descriptive.
The best observed mean checkpoint is not a validated deployment selection.

[Independent report](results/connect4cnn/replication-20261005/completed/independent-analysis/REPORT.md)
· [Whole curves](results/connect4cnn/replication-20261005/completed/independent-analysis/curves.html)
· [Audit](results/connect4cnn/replication-20261005/completed/audit.json).
Total recorded training process time was 1,927.671 seconds (32m 7.7s);
post-training checkpoint evaluations/builds are separate. No new runs or seed
extensions were launched when reviewing these results.

Local campaign: `build/connect4cnn/replication-20261005`.
Detached sessions `cnn-replication-20261005` and
`cnn-replication-archive-20261005` have finished. Training/evaluation and the
independent GPU-free archive are complete. The campaign's GPU reservation is
released; this does not authorize automatic repeats or new campaigns.

- Two fixed models: existing quality Flex (160,736 parameters) and adapted
  Nature (138,528). H128/L1 recurrent core, float32, corrected game, appearance 0.
- Five fresh paired training seeds: **53101–53105**. Order alternates by seed.
  Five seeds are a development replication stage, not a calibrated final sample.
- Identical `compare.ini` learner, environment and LR schedule; **13,312,000
  decisions per job**. No per-family tuning, architecture search, early stopping,
  selective retries, extra seeds or checkpoint filtering.
- **51 checkpoints per job**, every 262,144 decisions plus the final checkpoint.
  All 510 checkpoints are evaluated, after their training job finishes, against
  the same frozen **1,000-game seed-253101 suite** with 64 inference slots.
- Training time: process launch to native post-rename `CLOCK_MONOTONIC` receipt.
  Builds and checkpoint evaluations are outside training timing. Process/native
  SPS and memory metrics remain separately labeled and recorded.
- Source/tool snapshots and hashes, binary hashes, frozen full INIs, episode
  manifest, native startup/final configs, logs, weight hashes, finite/count
  checks, resource checks and per-episode outcomes are retained. The runner
  refuses existing compute processes and holds the local benchmark lock per job.
  A five-second background competitor check flags observed contention/telemetry
  failure. Brief unsampled contention is not proven absent.
- A training failure preserves and evaluates independently complete earlier
  checkpoints while remaining a failure. Missing seeds/cells prevent a complete
  mean frontier; unavailable scores are never treated as zero or omitted quietly.
- Candidate joint confidence bands are **disabled** for this development panel
  because their earlier coverage diagnostics were inadequate. Report whole seed
  curves and complete checkpoint means; no certified dominance/SOTA conclusion.

The launch estimate was about **31 minutes of training plus checkpoint
evaluation**, roughly 40 minutes total on an otherwise idle 5060. This is an
estimate, not a recorded duration. The 5090 remains untouched.

The protocol is frozen and copied to
[`results/connect4cnn/replication-20261005/protocol.json`](results/connect4cnn/replication-20261005/protocol.json).
The complete independently audited evidence/report is under its `completed/`
subdirectory. `STATUS.json` records `completed_and_audited`.

## Commands and tests

Existing-checkpoint reevaluation:

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python ocean/connect4cnn/frontier_eval.py \
  --run build/connect4cnn/compare.a2lt6qft \
  --binaries build/connect4cnn/deterministic-validation \
  --suite build/connect4cnn/deterministic-validation/suite/suite.json \
  --out build/connect4cnn/exact-frontier-NEW_ID
```

`--audit-only` reconstructs a fresh `analysis.audited/` from preserved evidence
without GPU inference. It refuses to replace a prior audit. The completed panel
above must not be repeated automatically.

Preparing another development protocol is GPU-free, but does not authorize
launching another campaign:

```bash
.venv/bin/python ocean/connect4cnn/replication.py prepare \
  --binaries build/connect4cnn/deterministic-validation \
  --out build/connect4cnn/replication-NEW_ID
```

The binary directory must contain `timing-default` and `timing-nature` built
from matching source with float32 and the proper compiled default. Full launch
requires a passing matching-build timing gate:

```bash
.venv/bin/python ocean/connect4cnn/replication.py run \
  --out build/connect4cnn/replication-NEW_ID \
  --gate build/connect4cnn/timing-canary-20261005
```

The independent archive command, after execution is complete:

```bash
.venv/bin/python research/report_connect4_replication.py \
  --run build/connect4cnn/replication-20261005 \
  --out research/results/connect4cnn/replication-20261005/completed
```

Do not run that manually while the scheduled archive worker owns the task.
The archive keeps small reports/source/config/provenance and compressed complete
episode CSVs; original uncompressed hashes and a compression index are retained.
Full weights and binaries remain local, with their original audit identified.

GPU-free preparation/artifact checks:

```bash
.venv/bin/python ocean/connect4cnn/tests/test_replication_tools.py
.venv/bin/python ocean/connect4cnn/tests/test_claim_tools.py
.venv/bin/python ocean/connect4cnn/tests/test_deterministic_eval.py
node research/tests/test_claim_chart.cjs
```

Two frozen-design/recovery artifact tests, twelve claim-tool tests (including
partial-run recovery), eight deterministic-evaluation artifact tests and twelve
chart states pass. The timing/recovery GPU canary is the actual runtime gate;
CPU fixture checks do not run a policy or substitute for GPU validation.
