# Evaluate the same pixel policy across drawings

October 6 [mixed-training follow-up](MIXED_PIXEL_TRAINING.md) adds a distinct
panel-v4 / binding-v3 path: persistent native slot mixtures, null single-training
drawing labels and fixed-ID evaluation of every checkpoint. The fixed-training
v1/v2 contracts described below stay unchanged. No policy or GPU ran.

October 6, 2026. GPU-free preparation and host manifests only. Both GPU holds
remain; no policy, neural model, training, dataset or pretraining ran.

## Why this adds a different robustness check

Training a fresh policy on each drawing tests whether an architecture learns
each visual variant. It does not test whether the **same learned weights**
transfer to another drawing. `pixel_evaluation_plan.py prepare
--evaluation-appearances all` now binds every existing planned training job to
every fixed drawing of its game, retaining every declared checkpoint. It adds
evaluation targets, not training runs, model changes or parameter imports.

All four frozen encoders receive the same target image and episode identities.
World, RNG, levels, rules, rewards, core and trained graph remain fixed. The
dedicated adapters override the evaluation drawing only, retaining the full
training INI as provenance. Native manifest preparation executes game/raster
code, not policy inference. Reflection keeps action IDs fixed: a policy trained
on one view is not automatically expected to handle a reflected view.

## Prepare on the machine holding the native builds

```bash
# Six tasks, drawing 0, four models: 24 planned training jobs, no execution.
.venv/bin/python research/prepare_pixel_robustness.py \
  --appearances default --seeds 53141 \
  --out build/pixel-transfer/FRESH_TRAIN_PANEL

# Compile-only if current-source builds are needed. No GPU execution.
NVCC_ARCH=sm_120 bash build/pixel-transfer/FRESH_TRAIN_PANEL/build-only.sh

# LOCAL_REGISTRY has default/impala/impoola paths for every requested task;
# use the registry format in PIXEL_EVALUATION_BINDINGS.md.
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/pixel_evaluation_plan.py prepare \
  --panel build/pixel-transfer/FRESH_TRAIN_PANEL/protocol.json \
  --registry build/pixel-transfer/LOCAL_REGISTRY.json \
  --evaluation-appearances all \
  --out build/pixel-transfer/FRESH_TRANSFER_BINDINGS

.venv/bin/python research/pixel_evaluation_plan.py inspect \
  --plan build/pixel-transfer/FRESH_TRANSFER_BINDINGS/plan.json \
  --check-binaries
```

The default 65,536 decisions and two checkpoints are plumbing placeholders,
not adequate-learning budgets. Use [shared learners](SHARED_LEARNER_RECIPES.md)
and [per-game budgets](TASK_BUDGETS.md) after scheduled calibration. This tool
has no campaign launcher. Remote agents regenerate from committed source and
their own binaries; G240 paths and uncommitted files cannot arrive via a pull.

## Contract and allocation

The default `--evaluation-appearances matched` keeps binding-v1 and its original
one-target-per-job semantics. Explicit `all` writes binding-v2. Every binding
has `training_representation`, `evaluation_representation`, the original
`job_index`, full training `config`/hash and its original checkpoint list.
`representation` denotes the **evaluation** drawing in v2. Consumers must not
replace the training INI with a target-drawing training recipe or retrain it.

Evaluation seed is still `seed + training_seed_index`; episode IDs are
`0..N-1`. Training/evaluation drawing, model, checkpoint and observed score do
not change this allocation. All training drawings of a task/seed reuse the
same target suite, so a full train-drawing × test-drawing matrix stays paired.
The reader reconstructs the complete target ordering and rejects missing,
duplicate, reordered or rebound targets; rehashing a receipt is insufficient.
Per-binding native host comparisons verify each family's starting worlds and
target raster. Cross-drawing comparison retains all world/RNG/level columns.
Connect4's declared empty-board identity manifest remains separate from a
native host raster proof.

For one seed:

| Game | Drawings | Drawing-0 training jobs | All-drawing evaluation bindings |
|---|---:|---:|---:|
| Connect4CNN | 10 | 4 | 40 |
| PongCNN | 7 | 4 | 28 |
| FlappyCNN | 4 | 4 | 16 |
| BreakoutCNN | 5 | 4 | 20 |
| SnakeCNN | 6 | 4 | 24 |
| MazeCNN | 6 | 4 | 24 |
| Total | 38 | 24 | 152 |

Each binding retains all checkpoints: two in the plumbing example give **304
planned checkpoint evaluations, zero completed**. At 1,000 assigned episodes,
that would require 304,000 policy episodes if later scheduled. Thirty-eight
host suites retain 38,000 rows; those are starting-state records, not completed
episodes. Four-model-family checks produce 112 non-Connect4 host comparisons.

An all-drawing **training** panel instead has 152 jobs and 1,048 transfer
bindings per seed (sum of four times each game's squared drawing count).
Do not accidentally pay for that larger training design when the question is
transfer of existing drawing-0 checkpoints. Additional paired seeds multiply
both counts; this is not a replication/power recommendation.

## Evidence retained and remaining gates

[The retained packet](results/pixel-transfer-bindings-20261006/README.md) uses
actual existing native metadata/manifests for all six games: 24 unexecuted
jobs, 38 suites, 152 transfer bindings and 112 per-binding native world/raster
checks. Inputs, binary hashes and tools pass offline inspection. Existing
binding-v1 and per-game-budget packets remain inspectable unchanged. Seven
new host/configuration tests cover the transfer matrix, paired seeds, complete
checkpoint cadence, relocation, corruption/rebinding and unchanged legacy
mode. They do not validate neural outputs.

Only Connect4's dedicated GPU evaluator has accepted runtime evidence. The
other five still require scheduled [evaluator acceptance](EVAL_ACCEPTANCE.md),
whole-model memory/math/reload checks, cap/learner calibration and learning.
Captured executable hashes/metadata are not current-source compilation
certificates. Maze uses the declared full table, not certified unseen levels.
Both modes remain development-only; v2 explicitly leaves unseen-appearance
selection exclusion uncertified. No advantage or SOTA is established here.

## Analyze learning and transfer separately

Retain a per-game/per-target full score-versus-training-time curve for each
training seed, including failures, declines and censored outcomes. All target
evaluations of one checkpoint use the **same** measured training time; never
count them as independent training replications or add their training cost
multiple times. Record evaluation time separately. For a train-drawing matrix,
keep training drawing and evaluation drawing as separate axes.

Separate three questions: fresh learning on each fixed drawing, transfer of a
fixed-drawing policy, and mixed-drawing training followed by fixed-ID tests.
The v2 extension prepares the second; mixed training uses the separate v4/v3
panel/binding path linked above.
Poor zero-shot transfer does not erase a from-scratch learning advantage, and
success on some drawings does not establish robustness to all of them. Do not
pool easy targets into a favorable mean or select a best target/checkpoint.
Fix appearance/seed/architecture-selection boundaries before confirmation and
account for correlated targets in frontier uncertainty. Existing statistical
coverage gates remain open.
