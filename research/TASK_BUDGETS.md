# Per-game budgets for fixed CNN transfer experiments

October 6 [mixed-training follow-up](MIXED_PIXEL_TRAINING.md) uses panel-v4,
retaining the same per-game budget/default/cadence rules and resolved maps.
`audit_pixel_panel.task_budget` handles v4 as well as v3; mixed training changes
the appearance allocation, not the matched budget contract within a game.

October 6, 2026. **Preparation only; both GPU holds remain.** No policy,
GPU query, training or dataset ran. This extends the existing fixed-graph panel,
not the native search algorithm. Python remains configuration/audit glue;
policy construction/execution/optimization remain native C/CUDA.

## What changed

`research/prepare_pixel_robustness.py` now accepts repeatable
`--task-budget ENV=STEPS:CHECKPOINT_STEPS`. `--steps`/`--checkpoint-steps`
are defaults for games without an override. Budgets may differ **between games**;
every CNN, drawing and seed **within a game** gets the same decisions, native
LR schedule and checkpoint cadence. Learner overlays remain shared per game.
Frozen quality/Nature/IMPALA/Impoola graphs, random initialization, H128/L1,
rules, drawing transforms, seed allocation and rotated job order are unchanged.

Each game's decisions and cadence must be positive exact rollout multiples
under its own resolved learner. Native scalar preflight verifies positive
updates and no dropped decisions. Partial final cadence retains the final
checkpoint; cadence greater than budget still retains that final checkpoint.
No defaults silently round a game to another game's rollout geometry.

Example: prepare Connect4 and Flappy with their different declared budgets,
without executing either learner:

```bash
.venv/bin/python research/prepare_pixel_robustness.py \
  --environments connect4cnn flappycnn --appearances all --seeds 53131 \
  --steps 13312000 --checkpoint-steps 262144 \
  --learner-recipe flappycnn=ocean/flappycnn/learner_stock_small_batch.ini \
  --task-budget flappycnn=19922944:1048576 \
  --out build/pixel-robustness/FRESH_TASK_BUDGETS

.venv/bin/python research/audit_pixel_panel.py \
  --panel build/pixel-robustness/FRESH_TASK_BUDGETS/protocol.json

.venv/bin/python -m unittest research.tests.test_task_budgets \
  research.tests.test_shared_learner_recipes research.tests.test_pixel_robustness \
  research.tests.test_pixel_evaluation_plan research.tests.test_learner_geometry -v
```

Use local native builds and [evaluation binding instructions](PIXEL_EVALUATION_BINDINGS.md)
to create dedicated development suites. No full campaign launcher is supplied.
Large declared budgets aren't authorization to occupy either GPU.

## Versioning and analysis contract

Explicit task budgets produce `pixel-robustness-preparation-v3` with
`shared-native-geometry-v2`. It records original defaults, explicit overrides
and the complete resolved `task_budgets` map. Every game gets native scalar
receipts, even when no learner overlay is supplied. No-overlay/no-budget
preparations remain v1; learner-only preparations remain v2. Existing archives
inspect under their original contracts; don't add new fields to old receipts.

Readers must use `audit_pixel_panel.task_budget(protocol, environment)` or the
resolved map for v3. Root `steps`/`checkpoint_steps` are **defaults**, not every
job's actual schedule. Evaluation bindings retain each job's complete list.
The auditor rejects malformed/duplicate/unknown task declarations, bool/float
integer substitutes, declaration/resolution drift, per-model rehashed budget
changes, unaligned geometry, altered core/world/builds and improper claim gates.
These are consistency checks, not signatures against complete malicious rewriting.

43 host/configuration/scalar tests pass (18.465 seconds), including 304
full-inventory/two-seed cells, inherited/default/override budgets, final partial
cadence, relocation without the scalar executable, CLI rejection and v1/v2
compatibility. The binding test preserves all declared checkpoints without a
GPU; its Connect4 binary metadata is explicitly synthetic. Actual host receipts
are retained separately below. No CPU neural substitute was used.

## Actual six-game preparation

[Retained packet and checks](results/task-budgets-20261006/README.md) cover all
six games, 38 drawings and 152 fixed-model cells, development seed 53131.
The following are **uncalibrated design inputs, not a launch recommendation**:

| Game | Decisions/job | Checkpoint decisions | Checkpoints/job |
|---|---:|---:|---:|
| Connect4CNN | 13,312,000 | 262,144 | 51 |
| PongCNN | 4,194,304 | 524,288 | 8 |
| FlappyCNN | 19,922,944 | 1,048,576 | 19 |
| BreakoutCNN | 54,919,168 | 1,048,576 | 53 |
| SnakeCNN | 13,312,000 | 262,144 | 51 |
| MazeCNN | 337,903,616 | 8,388,608 | 41 |

The example borrows resource scales from earlier/native controls, not validated
pixel learner recipes. Only Flappy receives the explicit stock-small-batch
overlay; the other common recipes remain untuned. Their state-control cores,
information and sometimes horizons differ. Maze stock-scale decisions with
the current common horizon **do not establish sufficient learning**.

It declares **10,496,245,760 training decisions and 5,836 checkpoint bindings**;
none were executed. Do not turn this large one-seed design into a confirmation
campaign. First freeze affordable per-game calibration under equal allowances,
qualify full-trainer memory/math/reload and the five pending exact evaluators,
then choose adequate budgets and fresh replication/stopping/inference rules.

Host-only preparation checks 1,000 assigned starts per drawing and 112 native
per-job world/raster manifests across five games. Connect4 separately retains
its declared empty-board/identity-seed manifest; it has no native host-manifest
command. Worlds/RNG/levels match across drawings, starts match across models.
Metadata/hashes don't certify current-source binary closure or GPU behavior.
Maze uses its full declared level table, not certified unseen levels; Pong/
Breakout administrative caps remain uncalibrated. No new score, SPS, robust
transfer, frontier dominance or SOTA is established by this packet.
