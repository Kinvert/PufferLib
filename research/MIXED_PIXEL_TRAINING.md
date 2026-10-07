# Deterministic mixed-drawing training panels

October 6, 2026. Preparation and native scalar/host manifests only. **No policy,
training, neural model, GPU query, dataset or pretraining executed.** Both GPU
holds remain; this document does not schedule a campaign.

## What this prepares

The six native pixel environments already assign a persistent drawing to each
environment slot when `env.representation_mode=1`. Fixed-model preparation now
exposes that existing behavior with `--appearances mixed --appearance-seed N`.
Every CNN in a task receives the same assignments, world, learner, decisions,
core and checkpoint cadence. The quality graph remains frozen across tasks;
Nature/IMPALA/Impoola stay their existing adapted architectures. All start with
random weights. No production game, encoder, learner or RNG code was changed.

Assignment uses the existing native unsigned hash and rejection sampling on
`(appearance seed, global environment slot)`. It runs once at initialization,
consumes no game RNG and persists through resets. This is **mixed per slot**,
not a new random drawing on every episode or frame. Thread/buffer ordering is
not an assignment input; slot count/order must be preserved on reload. The
learner's `base.seed` is separate and does not select the drawing mixture.
Maze's native vector path explicitly uses the global slot for appearances,
while its world RNG/level table retain their original initialization.

Pong mixes all seven current drawings through explicit
`representation_mix_catalog=1`. Its historical default mixed catalog remains
five drawings. Fixed/default/all panels retain their previous settings and
v1/v2/v3 contracts; old evidence is unchanged.

## GPU-free preparation

```bash
# 6 tasks x 4 CNNs x 2 paired training seeds = 48 planned jobs.
.venv/bin/python research/prepare_pixel_robustness.py \
  --appearances mixed --appearance-seed 12345 --seeds 53151 53152 \
  --out build/pixel-mixed/FRESH_PANEL

.venv/bin/python research/audit_pixel_panel.py \
  --panel build/pixel-mixed/FRESH_PANEL/protocol.json

# Compile-only if needed, using the usual build and existing dependencies.
NVCC_ARCH=sm_120 bash build/pixel-mixed/FRESH_PANEL/build-only.sh

# Use a registry of this machine's three float32 builds per game.
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/pixel_evaluation_plan.py prepare \
  --panel build/pixel-mixed/FRESH_PANEL/protocol.json \
  --registry build/pixel-mixed/LOCAL_REGISTRY.json \
  --evaluation-appearances all \
  --out build/pixel-mixed/FRESH_FIXED_EVALUATION_BINDINGS

.venv/bin/python research/pixel_evaluation_plan.py inspect \
  --plan build/pixel-mixed/FRESH_FIXED_EVALUATION_BINDINGS/plan.json \
  --check-binaries
```

Registry format is in [PIXEL_EVALUATION_BINDINGS.md](PIXEL_EVALUATION_BINDINGS.md).
Do not use G240 paths on the 5090; regenerate after committed source arrives.
Default budgets are 65,536 decisions/two checkpoints: plumbing, not sufficient
learning evidence. [Shared learner overlays](SHARED_LEARNER_RECIPES.md) and
[per-game budgets](TASK_BUDGETS.md) work with mixed panels, including the
equally applied Flappy smaller-minibatch candidate. Calibrate adequate budgets
and full-model memory/runtime before scheduling learning. No launch controller
is supplied by this preparation.

## Contracts and actual assignment receipts

Mixed panels use preparation-v4 / `shared-native-geometry-v3`, full resolved
per-game budget maps, native scalar learner receipts and a versioned
`native-slot-fixed-full-catalog-v1` appearance contract. The additional native
C tool parses the captured full INI and calls the existing shared assignment
helper. It never initializes a game, image, policy or CUDA. Each game gets a
`validation/*-appearances.csv` with slot/ID rows, hashes, catalog/seed/slot
metadata and per-drawing counts. Counts match an independent unsigned-arithmetic
translation, including rejection sampling; catalog sizes match captured native
header definitions. Full recipes/source/build ledgers remain checked.

`native_vector_initialization_validated=false` is intentional: scalar assignment
receipts are not acceptance of the live GPU learner/vector/reload path. Existing
environment/raster tests remain separate evidence. Optional scalar binaries
may be omitted from a portable copy before binding; their hashes/source/compiler
receipts remain. If present, changed helper executables are rejected.

At the declared seed 12345 and 64 slots, actual native counts were:

| Game | Counts for IDs in increasing order |
|---|---|
| Connect4CNN, 0–9 | 9, 7, 3, 4, 9, 5, 6, 8, 7, 6 |
| PongCNN, 0–6 | 8, 9, 10, 15, 9, 10, 3 |
| FlappyCNN, 0–3 | 9, 14, 25, 16 |
| BreakoutCNN, 0–4 | 14, 13, 11, 11, 15 |
| SnakeCNN, 0–5 | 8, 7, 14, 12, 12, 11 |
| MazeCNN, 0–5 | 8, 7, 14, 12, 12, 11 |

Every ID appears in this declared realization; it is **not balanced**. Future
seeds/slot counts may leave IDs absent, which is reported honestly through
`all_drawings_assigned`, not silently repaired. Do not search appearance seeds
for winning policies or assume equal experience per drawing. The two training
seeds above share the same mixture, so they replicate learner/action randomness,
not different mixture realizations. A future exposure study must predeclare
additional appearance seeds and match them across families.

## Fixed-ID evaluation of mixed-trained weights

Mixed panels require explicit `--evaluation-appearances all`; matched mode is
rejected before output creation. Binding-v3 records the original mixed training
seed/catalog/mode while every evaluation suite selects one fixed drawing.
`training_representation` is **null**, because a mixture is not one drawing.
The training INI's `representation=0` is an unused fallback, never the correct
label for this mixed experiment. `evaluation_representation` and the binding's
`representation` identify the actual target drawing. All checkpoint steps are
retained, and no extra training job is invented.

The actual two-seed six-game host packet contains **48 planned mixed jobs, 76
paired suites, 304 target bindings and 224 non-Connect4 family/world/raster
checks**. All suites have 1,000 assigned IDs. Every target of one checkpoint
shares the same training time and weights; two plumbing checkpoint steps mean
608 planned evaluations, zero completed. Host starts are not policy episodes.
Connect4 retains its declared empty-board identity proof separately from a
native raster manifest. Evaluating a policy on every appearance is not evidence
that all appearance families were withheld from training or selection.

[Retained sources/configs/receipts](results/pixel-mixed-20261006/README.md) include
native metadata/manifests, independent scalar/UBSan tests and successful legacy
packet inspection. Tests cover complete mixed/fixed target allocation, source/
recipe/seed/catalog closure, relocation, rejected ambiguous labels, invalid
inputs and unchanged legacy contracts. They do not run a neural model.

## Remaining scientific and runtime work

First accept pending exact evaluators using existing saved policies in a
scheduled GPU window. Then qualify whole-model math/memory/reload for all four
families under the same recipe, calibrate learning budgets/caps and compare
fixed-drawing versus mixed-training full frontiers across games. Preserve
per-target seed/checkpoint curves, failures and censored results. Keep costs
and timings separate for training, evaluation and tuning. Do not pool drawings
as independent seed replications or count one training run's cost repeatedly.

All current catalogs in the mixed example are exposed during training: this
would test robustness **within the trained catalog**, not unseen-drawing
generalization. A later explicitly withheld catalog/appearance-family design
is different work. Only Connect4's dedicated GPU evaluator is accepted;
other games' runtime gates, cap calibration, full memory/model math, encoder-5,
frontier uncertainty and independent replication remain open. No general
architecture advantage, Pareto dominance or SOTA follows from preparation.
