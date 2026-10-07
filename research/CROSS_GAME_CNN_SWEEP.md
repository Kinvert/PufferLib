# One CNN candidate across games and drawings

October 6, 2026. Kinvert's scope: find a CNN architecture that generalizes across
the six existing native pixel games. LR, clipping and other learner settings
belong to each game and are fixed across CNN candidates within that game.
Do not pick a different winning architecture per environment.

`candidate_panel.py` is external configuration/process/audit glue. Construction,
random initialization, learning and exact policy inference stay native C/CUDA.
It does not implement a Python optimizer. Native PROTEIN discovery is currently
single-game; this first version expands its candidates into a full multi-game
validation panel. Feeding a cross-game objective back to PROTEIN is a later
native change, requiring a declared normalization across unlike score units.

That description applies to this generic panel. A separate local research
[native feedback prototype](CROSS_GAME_PROTEIN_FEEDBACK.md) now implements the
outer adaptive loop with fixed normalization. It is compiled/prepared, not GPU
executed, and must not be pushed as the eventual clean architecture PR.

**Completed local smoke:** [retained evidence](results/candidate-panel-20261006/README.md)
passes 12 training jobs/82 all-drawing evaluations/24 graph-repeat/eager exact-byte
checks. One retained initial parser failure was fixed with a fresh packet.
Nine host/config/scalar tests and 20 reporting regressions pass. A real old
PROTEIN campaign imports all 12 completed proposals, preparing 72 jobs/492
targets without executing them. No quality ranking follows from the smoke.
The commands below are reference instructions, not a request to relaunch it.

## First bounded 5060 smoke

The user's latest message explicitly authorizes this local smoke. It does not
authorize encoder 5, a long local search or a 5090 campaign now. Use the existing
venv/toolkit/NCCL, never install/change global CUDA or Torch. No CPU neural tests.

Two encoder-4 candidates: the frozen quality architecture (16 channels, 7x7
stride-4 convolution, projection 64), and the same graph with 8 channels. Both
retain H128/L1. Six independently initialized/trained policies per candidate;
this is architecture generalization, not one transferable trained policy.

Per game/candidate: 65,536 decisions, one final checkpoint, 64 actor slots,
horizon 32, minibatch 2048 and the game's unchanged `compare.ini` recipe.
Training uses deterministic fixed-per-slot drawing assignments, seed 35173,
full catalogs (Pong/Flappy explicitly catalog 1). Counts aren't forced balanced.
Independent scalar/native assignment CSVs are retained; this does not by itself
certify the live vector's initialization.

Each checkpoint receives all fixed drawings: Connect4 10, Pong 7, Flappy 7,
Breakout 5, Snake 6, Maze 6. Evaluation fixes the same 17 episode identities,
worlds/RNG and 16-slot batch across candidates/checkpoints/drawings; worlds are
checked across drawings during preparation. The second wave has a partial tail.
Pong's 512-decision administrative cap retains censoring bounds; Breakout uses
2,048 physics frames. These small caps are for plumbing, not quality conclusions.
Flappy/Snake/Maze retain their native horizons. Maze uses the original 8192-level
table and makes no held-out level exclusion claim.

Allocation: **12 training jobs, 82 drawing evaluations, 1,394 assigned episodes**,
plus 24 drawing-0 graph-repeat/eager evaluations (408 additional episodes).
The smoke compares exact episode CSV bytes for both modes for every candidate/
game, but that isn't an independent numerical/gradient or full acceptance test.
One training job is charged once, irrespective of its drawings/repeats.

Fresh paths are required throughout. Build/prepare never query a GPU; host native
metadata/manifests and scalar arithmetic are permitted. `run` reserves the shared
hardware lock for each training process; adapters reserve it for evaluation.
Busy/query failure stops and retains partial evidence without killing other work.
Process groups and campaign/per-process deadlines bound execution. Never actively
watch a training run; it writes progress/results itself.

```bash
source ocean/connect4cnn/runtime_env.sh

.venv/bin/python research/candidate_panel.py build \
  --out build/candidate-panel/BUILD_FRESH

.venv/bin/python research/candidate_panel.py prepare \
  --registry build/candidate-panel/BUILD_FRESH/registry.json \
  --candidate happy-cat-1=research/recipes/panel_smoke_quality.ini \
  --candidate mystic-tree-2=research/recipes/panel_smoke_small.ini \
  --out build/candidate-panel/SMOKE_FRESH

.venv/bin/python research/candidate_panel.py inspect \
  --plan build/candidate-panel/SMOKE_FRESH/plan.json

# Only in the explicitly authorized, idle 5060 window.
.venv/bin/python research/candidate_panel.py run --mode smoke --allow-gpu \
  --plan build/candidate-panel/SMOKE_FRESH/plan.json \
  --timeout 900 --train-timeout 60 --eval-timeout 30

# Offline: retained clocks/checkpoints/configs/assigned CSVs, no policy or query.
.venv/bin/python research/candidate_panel.py audit \
  --plan build/candidate-panel/SMOKE_FRESH/plan.json \
  --out build/candidate-panel/SMOKE_REVIEW_FRESH
```

Native SPS/uptime/performance arrays remain in each job's original metrics INI.
`training.csv` reports whole-process SPS separately; no invented native average.
`observations.csv`/`analysis.json` retain every game/drawing's actual score units,
Pong lower/upper bounds and missing cells. Short scores do not select a winner,
establish a frontier or qualify a paper claim. Existing four-baseline reporting
packets remain unchanged; this is a distinct arbitrary-candidate protocol.

## Larger 5090 panel after delivery and scheduling

For direct baseline comparisons, use the opt-in
[shared candidate/reference panel-v2](CANDIDATE_BASELINE_COMPARISON.md).
It adds fixed Nature/IMPALA/Impoola, three normal builds per game, complete
recipe reconstruction and native family/config start comparisons. Default v1
packets and the successful smoke remain unchanged. The new panel is preparation
only until qualified and separately scheduled; no extra local smoke is authorized.

The architecture-only discovery recipe is
`research/recipes/general_cnn_discovery.ini`: 18 numeric CNN dimensions,
H128/L1 and 13.312M decisions fixed. The native preparation helper discards
inherited `[sweep.*]` sections first; the recipe adds only `[sweep.policy.cnn_*]`.
There is no literal `sweep_only` switch in this checkout. LR/clip/gamma/etc.
come from each game's `compare.ini`, unchanged by the discovery recipe. The
same broad CNN grammar can be prepared for all six games, but broader-shape
math/runtime qualification and meaningful learner/cap calibration precede launch.

Explicit `sweep.py --appearance-seed N` now prepares full-catalog persistent
slot mixtures for all six discovery games, including Flappy's seven-ID catalog.
Seed/catalog are fixed experiment settings, not search coordinates. See
[DISCOVERY_PIXEL_MIXTURES.md](DISCOVERY_PIXEL_MIXTURES.md) for counts, legacy
behavior and the same-seed validation handoff. Omitting the option preserves
previous recipe settings. This doesn't schedule discovery or add an adaptive
cross-game objective.

```bash
# GPU-free preparation only; does not instantiate the PROTEIN GP or a policy.
.venv/bin/python ocean/connect4cnn/sweep.py \
  --environment connect4cnn --recipe research/recipes/general_cnn_discovery.ini \
  --max-runs 128 --prepare-only --wandb disabled
```

When discovery is separately scheduled, use the shared reservation for the
entire existing native sweep runner (`flock -n build/connect4cnn/hardware-benchmark.lock ...`),
not parallel single-game campaigns. Preserve every original discovery record,
including failures and consumed process/native cost, before importing completed
proposals. A single-game optimizer still proposes against that game's native
training metric; the deterministic full panel is the generalization check.

The 5090 agent must pull **committed/pushed Kinvert source**, then build and prepare
fresh local paths; G240 paths/binaries aren't shared. Nothing here has been
pushed by this task. Do not launch a long campaign merely because these commands
exist. First review the local smoke/failures and qualify the relevant model math,
memory, evaluator/reload and learning recipes/caps on the scheduled hardware.

An existing native encoder-4 PROTEIN campaign can supply **every completed
candidate**, including duplicate and dominated configurations. Imports retain
actual saved trial INIs, proposal task, discovery cost/score, log/checkpoint hashes;
no score filtering, pruning or claimed cross-game adaptive feedback. Incomplete/
failed discovery trials remain in their original campaign, not counted as completed
proposals. Unsupported encoder-5 imports fail, rather than being silently skipped.
Broader encoder-5 support follows its separate numerical/runtime gates.

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/candidate_panel.py build \
  --out build/candidate-panel/BUILD_5090_FRESH

# Replace budgets/cadences with calibrated, declared per-game values.
# Repeat --native-campaign to pool independent native proposal campaigns.
.venv/bin/python research/candidate_panel.py prepare \
  --registry build/candidate-panel/BUILD_5090_FRESH/registry.json \
  --native-campaign build/connect4cnn/LOCAL_NATIVE_PROTEIN_CAMPAIGN \
  --seeds 57173 57174 57175 57176 57177 \
  --steps 13312000 --checkpoint-steps 262144 \
  --episodes 1000 --slots 64 \
  --pong-max-decisions 131072 --breakout-max-frames 131072 \
  --out build/candidate-panel/DEVELOPMENT_5090_FRESH
```

The example budgets/caps are **not calibrated launch recommendations**. CLI
`--task-budget ENV=STEPS:CHECKPOINT_DECISIONS` records a matched per-game schedule;
`--learner-recipe ENV=INI` applies a numeric per-game learner overlay identically
to all candidates (world/architecture/seed/budget/output edits forbidden). This
first launcher holds geometry at 64 slots/H32/M2048/H128-L1; geometry expansion
needs explicit memory/math/runtime acceptance. Budget curves are retained in full.

Before scheduling, inspect `planned_training_jobs` and `planned_evaluations`.
For C candidates/S seeds/K checkpoints per game, cost is 6*C*S training runs
and 41*C*S*K drawing evaluations. For example 100*5*51 produces 104,550 evals:
this can dominate cost. Progressive discovery/validation is useful, but final
confirmatory candidates/seeds/episodes must be allocated before seeing outcomes;
retain every rejection and all discovery/selection costs.

Scheduled 5090 execution uses `run --mode development --allow-gpu`, explicit
whole-campaign/per-process limits, then offline `audit` with a fresh report path.
The development mode verifies the GPU name contains 5090. It does not lift the
current 5090 scheduling hold or perform preliminary numerical acceptance.

Keep score/time fronts separate per game and drawing; compare one fixed candidate
across them. The opt-in [candidate/reference panel-v2](CANDIDATE_BASELINE_COMPARISON.md)
now includes Nature/IMPALA/Impoola with identical per-game learner, seeds, budgets,
backend and cap treatment; its native preparation is complete, not executed.
Do not average raw pipes, wins, score,
length and goal rates into a fabricated objective. Confirm held-out performance,
paired uncertainty/full-curve inference and deployment speed independently before
calling an architecture generally better or a region of the frontier SOTA.
