# Paired multi-game frontier reporting

October 6, 2026. `pixel_frontiers.py` is offline experiment reporting/audit glue.
It executes no native binary, policy, GPU query, training or dataset generator.
Policies, architecture construction and optimization remain native C/CUDA.
Both GPU holds and existing evaluator/math/memory/learner gates remain.

[Retained checks and actual empty collection](results/pixel-frontiers-20261006/README.md).
[Offline viewer follow-up](results/pixel-frontiers-viewer-20261006/README.md).
Kinvert has frozen the current inventory at six environments/41 drawings for now;
focus on trustworthy comparisons rather than adding games.

## Allocation before measurement

Start from an audited [evaluation binding packet](PIXEL_EVALUATION_BINDINGS.md).
Preparation checks the complete captured panel and creates an explicit ledger:
one training entry per original job, and one result entry per declared
checkpoint/target drawing. All artifact paths start as JSON `null`; no score,
duration or successful evaluation is invented.

```bash
.venv/bin/python research/pixel_frontiers.py prepare \
  --plan LOCAL_BINDINGS/plan.json --out build/pixel-frontiers/FRESH_COLLECTION

# Reporting is offline, even when measured paths have been filled in later.
.venv/bin/python research/pixel_frontiers.py report \
  --collection build/pixel-frontiers/FRESH_COLLECTION/collection.json \
  --out build/pixel-frontiers/FRESH_REPORT
```

Both destinations must be fresh; reports cannot write into the frozen binding
packet. Preserve earlier collections/reports when updating progress. An untouched
ledger produces an explicitly incomplete report containing every missing cell.
This is not a campaign launcher or permission to schedule a GPU.

## Training receipts after authorized execution

For each job, point its `receipt` to a JSON object with paths relative to that
receipt file, or absolute paths:

```json
{
  "process": "train.log.json",
  "log": "train.log",
  "hardware": "hardware.json",
  "checkpoints": {
    "32768": "checkpoints/flappycnn/trial/0000000000032768.bin",
    "65536": "checkpoints/flappycnn/trial/0000000000065536.bin"
  }
}
```

Those steps are a short plumbing example, not a proposed learning budget.
The process record follows `claim.process`: exact command/cwd, terminal status,
exit code, process-local receipt flag, integer launch/end monotonic nanoseconds.
`hardware.json` requires nonempty strings for `gpu_uuid`, `gpu_name`, `host`,
`cuda_compiler` and `driver`, captured during scheduled execution. Keep hardware
comparisons in separate collections; mixed metadata or overlapping recorded
training intervals are refused. These checks cannot detect unrecorded competing
GPU jobs or certify clock/hardware declarations independently.

The actual working directory's default/secondary INIs must match the captured
job bytes, and native `resolved.ini` must match their parsed settings. The exact
binary hash/command must match its registered family. Checkpoint paths must be
the planned native output paths; their list must be the complete ordered prefix
of the declared schedule. Float32 size, finiteness and constant parameter count
are checked. Timing comes exclusively from strict post-rename
`PUFFER_CHECKPOINT` receipts within the process clock interval. No file mtime,
interpolation, evaluation duration or inferred training duration is substituted.
Successful jobs require all checkpoints; failed jobs can retain a valid prefix,
but remain failures and cannot produce a complete frontier condition.

Training costs are charged once per job, regardless of target drawing count.
`training_cost_accounting_complete` is false when any planned job lacks audited
process timing. Process SPS is final decisions divided by whole-process time
for successful jobs. Native SPS is explicitly unavailable in this reader;
retain native metrics separately rather than filling it with zeros or substituting
process SPS. Build/evaluation/selection costs remain separate work.

## Evaluation receipts and score units

Fill each ledger `result` with the corresponding dedicated adapter's
`result.json` path after supervised, authorized evaluation. The reader checks
family/binary/config/suite/checkpoint identities, parameter count, copied inputs,
effective configuration semantics, empty secondary override, raw assigned-ID
CSV and native completion summary. Prepared inputs cannot count as evaluations.
It reruns each game's existing CSV auditor and requires stored counts to agree.
All inputs and executing reader sources are hashed and checked for changes.
This consistency audit does not independently prove binary compilation source,
GPU numerical/runtime acceptance or scientific authenticity of supplied artifacts.

| Task | Reported quality coordinate |
|---|---|
| Connect4CNN | Win fraction over all assigned games |
| PongCNN | Mean final point-fraction lower/upper censoring bounds |
| FlappyCNN | Mean pipes passed |
| BreakoutCNN | Mean score through first terminal or fixed frame cap |
| SnakeCNN | Mean ending length under the separate local episodic rules |
| MazeCNN | Goal fraction over all assigned episodes |

Pong intervals are deterministic censoring bounds, **not confidence intervals**.
No midpoint or completed-match-only estimate replaces them. Breakout's metric
is explicitly capped-horizon score, not a prediction of uncapped final score.
Other administrative/native endings stay in their corresponding estimands.

## Full curves and incomplete conditions

Reports keep every chronological observation, dominated point and score decline.
Means require every assigned training seed at that model/checkpoint, with equal
seed weight. No best-seed pooling, checkpoint selection or averaging unrelated
game scores. Group by game, training appearance treatment and evaluation drawing;
fixed-source transfer and mixed-catalog training are distinct treatments.

If any expected cell/failure is missing within a condition, its frontier flags
remain `null`. Complete conditions receive descriptive lower/upper score
envelope flags. They do not certify dominance; in censored Pong these are two
different coordinate envelopes, not a single identified frontier. Paired
confidence bands, held-out selection exclusion and deployment selection remain
disabled. There is no external SOTA inference from this reader.

Outputs: self-contained `curves.html`, `analysis.json`, `REPORT.md`, every observation/paired mean CSV and the
complete missing-cell CSV. Input/tool hashes and failures are in `analysis.json`.
The current actual six-game collection allocates 24 mixed jobs and 328 checkpoint
evaluations across 41 drawings; all 328 remain missing because no policy ran.

## Interactive offline viewer

Open `curves.html` locally; its data, JavaScript and SVG renderer are embedded,
with no CDN, browser fetch, external dependency or server. Adjacent report/CSV
links require the rest of the report directory. Choose a game/training/evaluation
condition, paired means or one assigned seed, and wall seconds or decisions.
All four families start visible, including IMPALA. Coverage always lists all
families even when one is hidden or has no observation. Optional individual
seed curves stay separate behind means. Scales cover the full visible cost
range, including overlaid seed curves; no fixed three-time-window restriction.

The viewer never recomputes scores, averages or frontier flags. Black rings show
the audited lower/upper **time** flags only for complete conditions and means;
they refer to all assigned models even after a legend toggle. Seed/decision
views have no rings. Missing expected checkpoints break lines. Full curves,
score declines and non-frontier points remain accessible in the checkpoint table.
Pong draws lower and dashed upper curves/vertical censoring ranges, never a
midpoint or confidence interval. There is no cross-game aggregate score.

Training job costs retain missing/failed statuses and the one-job allocation;
cross-drawing tables reuse the same jobs, so do not sum them across conditions.
Whole-collection cost accounting, hardware declarations, input/source hashes,
all missing identities and failures remain available. Native SPS stays explicitly
uncollected. Embedded JSON escapes script terminators/HTML and rejects nonfinite
payloads. The template and display-helper hashes are part of reader provenance.

Twenty Python artifact/synthetic checks and ten Node helper/application checks
pass; the related catalog/binding/transfer suite passes 45 tests. A minimal DOM exercises the actual rendering/event code for complete,
censored, empty and partial fixtures; it is **not** a browser layout engine,
real model or statistical acceptance. No real browser visual inspection is
claimed. Actual offline reproduction still has zero policy observations, with
328 missing cells and 41 incomplete conditions; it draws no invented curve.
The earlier archive remains unchanged: its verifier's source-match requirement
now needs the retained old reader revision, not this updated reporter.

## Executed checks

Seventeen focused tests cover paired means, declines, missing/bad seeds, all six
native summary formats, metric units, censoring, exact monotonic timing, raw
episode tampering, rehashed effective-policy drift, hardware/overlap rejection
and allocation/claim flags. Actual pending adapters independently produce
graph/eager prepared configs in the fixtures; native metadata/spawn subprocesses
are mocked and outcomes/weights are synthetic. Ten configurations audit 1,000
retained host starts each, without policy execution. The related 42-test suite
also passes. One intermediate Snake fixture wrote `2048.0` into an integer CSV
column; that failed report is retained, and only the fixture was corrected.

Next measured work remains scheduled evaluator and full-model acceptance,
shared learner/cap calibration, then matched paired learning curves across games
and drawings. This tool prepares their reporting path; it does not establish
multi-game learning, an efficient baseline backend or a Pareto improvement.
