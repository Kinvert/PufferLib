# Shared learner recipes for fixed CNN comparisons

October 6 [mixed-training preparation](MIXED_PIXEL_TRAINING.md) accepts these
same learner overlays under panel-v4. Appearance seed/catalog are explicit
panel controls, never hidden learner overrides; all models share the resolved
settings. Host/scalar checks pass, with no new GPU or learning acceptance.

Later October 6 [memory preflight](LEARNER_MEMORY_PREFLIGHT.md) establishes that
stock IMPALA/Impoola encoder training buffers alone exceed the 5060's 8 GiB.
An additional stock-except-minibatch-2,048 candidate now prepares twelve
drawing-0 calibration cells equally across all four CNNs; no training occurred.
Retain the earlier two all-drawing candidates and their separate regimes. Full
fit remains unverified even for the smaller batch and for the 5090. The
35-test follow-up preserves the original 34-test receipt below unchanged.

October 6, 2026. Preparation only. Both GPU execution holds remain. No policy,
training, GPU query, dataset generation or pretraining ran. The four encoder
graphs remain frozen and start from random weights, with the same H128/L1 core.

## Why this was needed

The six-task plumbing panel used 64 agents, horizon 32 and a 2,048-decision
minibatch. It could not represent the stock Flappy learner/vector settings that
previously learned successfully with the quality CNN. Equal decision budgets
do not erase differences in optimizer updates, actor batching, LR or clipping.
The [geometry audit](LEARNING_RECIPE_CALIBRATION.md) established those differences.

`prepare_pixel_robustness.py` now accepts repeatable `--learner-recipe ENV=INI`
overlays. Each overlay is applied equally to **every encoder** of that game.
This is shared-recipe comparison preparation, not hyperparameter selection or
a training/search framework. Native C/CUDA still constructs and trains policies.
Other requested games can keep their existing recipe; they get their own
scalar geometry receipt when any explicit overlay is used.

Allowed numeric controls are `[base]` async/graph/reset settings, `[vec]`
agents/buffers/threads and native `[train]` fields except GPU count and total
decisions. An overlay cannot change the game, drawings, policy/core, seed,
initialization, checkpoint cadence, budget, selfplay or historical policies.
DEFAULT inheritance, empty overlays, lists and nonfinite scalars reject.
The CLI controls budget/cadence/seeds separately; full resolved INIs retain all
fixed and inherited values. Numeric settings still need runtime/learning
qualification; whitelist acceptance isn't a blanket valid-training certificate.

## Current Flappy development candidates

`ocean/flappycnn/learner_stock.ini` records explicit stock Flappy actor,
vectorization and learner settings. It preserves the frozen CNN H128/L1 core;
it is **not** the original stock-state H64/L2 policy. Only the quality CNN has
historical learning evidence with this learner, on drawing 0 and one pooled-v1
seed. Nature/IMPALA/Impoola at this larger minibatch remain unqualified.

Both prepared candidates use 19,922,944 decisions and checkpoints every
1,048,576 decisions, including the final checkpoint: 19 checkpoints per job.
These counts are exact rollout multiples in both recipes.

| Candidate | Agents | Horizon | Minibatch decisions | Rollout decisions | Updates per rollout | Total updates |
|---|---:|---:|---:|---:|---:|---:|
| Stock Flappy learner | 2,048 | 64 | 16,384 | 131,072 | 8 | 1,216 |
| Existing small-batch recipe | 64 | 32 | 2,048 | 2,048 | 1 | 9,728 |

Each candidate crosses four fixed CNNs, Flappy drawings 0–3 and paired
development seeds 53121/53122/53123: 48 jobs each, **96 prepared, zero trained**.
Within each candidate, every encoder gets the same actual decisions, learner,
LR schedule, checkpoint steps and seed allocation. Between candidates, vector,
learner and actor settings differ; report their full frontiers separately.
Do not select only whichever recipe or drawing favors our CNN.

GPU-free preparations invoke the existing native host info/manifest commands
for 12 suites per candidate, with 1,000 assigned starts each and 64 evaluation
slots. All 48 job manifests match the canonical starts within each candidate.
The two candidates' 12,000 starting-world/RNG/raster rows also match byte for
byte. This proves allocation/start pairing; **it is not 12,000 played episodes**.
The [durable packet](results/shared-learner-recipes-20261006/README.md) retains
both panels, both evaluation bindings, scalar receipts, sources and checks.

## Prepare on the machine that will eventually run it

Reuse the existing Python 3.12 venv and native dependencies. Use fresh IDs;
existing output directories are refused. These commands do not launch training:

```bash
.venv/bin/python research/prepare_pixel_robustness.py \
  --environments flappycnn --appearances all --seeds 53121 53122 53123 \
  --steps 19922944 --checkpoint-steps 1048576 \
  --learner-recipe flappycnn=ocean/flappycnn/learner_stock.ini \
  --out build/pixel-learning/FRESH_STOCK

.venv/bin/python research/prepare_pixel_robustness.py \
  --environments flappycnn --appearances all --seeds 53121 53122 53123 \
  --steps 19922944 --checkpoint-steps 1048576 \
  --out build/pixel-learning/FRESH_SMALL

.venv/bin/python research/audit_pixel_panel.py \
  --panel build/pixel-learning/FRESH_STOCK/protocol.json

.venv/bin/python -m unittest research.tests.test_shared_learner_recipes \
  research.tests.test_pixel_robustness research.tests.test_pixel_evaluation_plan \
  research.tests.test_learner_geometry -v
```

Follow [PIXEL_EVALUATION_BINDINGS.md](PIXEL_EVALUATION_BINDINGS.md) to bind each
panel to local native builds and development suites. G240 binary paths are not
available on the separate 5090. Receipts from these existing binaries certify
their bytes/host behavior, not compilation closure against the new panel.

## Configuration and provenance contract

Explicit overlays produce `pixel-robustness-preparation-v2` with contract
`shared-native-geometry-v1`. Legacy no-overlay packets remain v1 and inspect
under their original small-batch semantics. Do not add v2 fields to old
packets or edit earlier prototype receipts to pass the stricter contract.

Before publishing the packet, preparation compiles frozen C scalar arithmetic
with the native INI parser, snapshots the reviewed learner source, and records
per-game rollout/update/budget/cadence receipts. A zero-update configuration
retains its scalar result and failed preflight; it cannot become a prepared
learning panel. Malformed/unsupported geometry rejects. Positive arithmetic
counts do not prove numerical correctness, GPU memory fit or learning adequacy.

The captured overlay can live outside the repository. `source.sha256` checks
the entire frozen source tree, including its virtual `learner-recipes/` copy;
`source.repo.sha256` checks only repository paths. The build-only script checks
both before and after normal native builds and has its own v2 digest. It
contains no train/evaluate launch. The auditor reconstructs the full recipe
and rejects rehashed one-model or all-model setting drift, changed secondary
game INIs, source/geometry/build changes and failed-preparation packets.
These are consistency checks, not signatures or malicious-tampering protection.

34 host/configuration/scalar tests pass, including archive relocation without
the host scalar executable, external-overlay removal, float32 boundaries,
zero-update failures and unchanged v1 panels. No CPU model substitution was used.

## Next execution gates

First qualify the pending exact evaluator with saved weights through
[EVAL_ACCEPTANCE.md](EVAL_ACCEPTANCE.md) in an explicitly scheduled window.
Then check memory, numerical math and train/reload behavior for **all four**
families at each chosen geometry before any timed comparison. The 16,384-decision
IMPALA minibatch is much larger than the historical 2,048-decision baseline;
neither 5060 nor 5090 fit/speed is established. Do not silently shrink one
model's batch, catch OOM as a successful score, or substitute a CPU learner.

After those gates, freeze an affordable calibration panel, caps, retention,
timing and failure rules. The prepared three seeds are development allocation,
not a powered paper confirmation or authorization to occupy either GPU.
Carry the same frozen architecture to the other native games after shared
per-game recipe calibration; pretraining remains a separate future axis.
