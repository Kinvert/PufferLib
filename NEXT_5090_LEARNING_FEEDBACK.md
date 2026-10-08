# Real staged CNN discovery run on the RTX 5090

Read this instead of repeating the completed 65K-step feedback canary. Work is
temporary research on Kinvert's fork, not an upstream PufferLib PR. No code
editing, CUDA installation, Torch installation or CPU CNN reference is needed.

From the existing checkout, update without discarding local work:

```bash
cd ~/Git/ml/cnn-5090
git pull --ff-only origin cnn-research
bash research/run_learning_feedback_5090.sh prepare
```

`prepare` is GPU-free. It runs host checks; builds the native optimizer bridge,
six normal float32 game targets, three metadata tools and two CUDA test libraries;
and registers both calibration and paired-reference panels. It freezes exact
recipes, seeds, all 41 drawing suites, update geometry and build/source identities.
It prints a fresh directory and the exact `run` command. Use that printed path:

```bash
bash research/run_learning_feedback_5090.sh run build/learning-feedback/5090-TIMESTAMP
```

Launch only when the 5090 is available. Shared reservation and idle checks refuse
competing work; there is no automatic retry or continuation of an old allocation.
Do not actively poll. The launcher saves all logs and creates a full evidence
archive/checksum on success, failure or insufficient learning.

If `git pull --ff-only` fails because the checkout has divergent local commits,
preserve them and the old artifacts. Create a clean sibling worktree instead of
resetting the checkout:

```bash
git fetch origin cnn-research
git worktree add ../cnn-5090-learning origin/cnn-research
source build/connect4cnn/runtime-5090.sh
cd ../cnn-5090-learning
bash research/run_learning_feedback_5090.sh prepare
```

Source that existing runtime helper only if it exists; otherwise export
`CUDA_HOME` and `NCCL_ROOT` for the existing toolkit/NCCL. The launcher uses
process-local paths and explicit `sm_120`. An absent venv is created with
`uv venv --python 3.12 .venv` and only NumPy 2.5.3 is installed there. An existing
incompatible venv is rejected, never replaced silently.

## What the real run does

```text
CUDA numerical checks: quality and Nature, B1/B64/B2048
                         |
                         v
Paired learning controls: quality + Nature
six games x two calibration seeds x four checkpoints x every drawing
                         |
                         v
Native receipts audited; learning eligibility checked for EVERY game
                 |                         |
           inadequate learning        all games eligible
                 |                         |
       save report and STOP                v
                               native PROTEIN proposes a CNN
                                            |
                               independent CUDA graph checks
                                            |
                               train on all six games/two seeds
                                            |
                               exact evaluation of all 41 drawings
                                            |
                               equal-game quality + training cost
                                            |
                               native PROTEIN observes, repeat
                                            |
                               12 suggestions completed/audited
                                            |
                               quality + Nature references trained
                               on the SAME search seeds/recipes
                                            |
                               complete paired per-game/drawing curves
```

All models use float32, H128/L1 and fresh random weights independently per
game/seed. Only the CNN's 18 shape coordinates vary in the search. Learners,
budgets, training mixtures and evaluation assignments are fixed per game across
architectures. Training appearance uses seed35173 and the full native catalogs.
Each checkpoint is evaluated on every fixed drawing, with 33 assigned episodes
and 16 slots. Drawing-zero graph/repeat/eager CSVs must agree exactly for every
job. Administrative Pong/Breakout caps are recorded and censoring remains visible.

## Learning budgets and the stopping rule

The control runs use the full budgets below. Search and reference runs use the
same full budgets and checkpoint schedule, including the same LR annealing.
We do not shorten a budget based on an early checkpoint under a different LR
schedule, tune one architecture preferentially, or drop a difficult game.

| Game | Decisions per model/seed | Checkpoint cadence | Learner source |
|---|---:|---:|---|
| Connect4 | 13,312,000 | 3,328,000 | Existing matched comparison recipe |
| Pong | 8,388,608 | 2,097,152 | Historical recipe A scalar overlay |
| Flappy | 19,922,944 | 4,980,736 | Stock scalar settings, declared smaller shared rollout |
| Breakout | 16,777,216 | 4,194,304 | Existing comparison recipe |
| Snake | 8,388,608 | 2,097,152 | Existing comparison recipe |
| Maze | 33,554,432 | 8,388,608 | State-Maze scalar overlay, horizon256 |

All use 64 actors and minibatch2048; Flappy horizon64, Maze horizon256, others
horizon32. Native effective replay/update counts are frozen, not inferred from
the decimal replay ratio. Those adapted learners are development choices,
not claims of stock-equivalent tuning or guaranteed learning.

For each game, at least one control must have final-checkpoint mean normalized
score >=0.1 across both seeds/all drawings, with each seed's drawing mean >=0.025.
Normalization uses the fixed anchors declared before calibration: Connect4,
Pong and Maze0→1; Flappy0→100; Breakout0→20; Snake1→20. Pong uses its lower
censoring bound. This is an explicit development eligibility rule, not a
statistical superiority test or a claim that these anchors are universally best.
An unresolved game stops the search and leaves `execution/calibration-gate.json`
with every control/seed score. Report it; do not lower thresholds, increase
budgets, choose a best seed, rerun failed evaluations or patch code locally.

Calibration and references each allow24 jobs/656 drawing evaluations/48 repeated
checks. Search allows12 proposals, each12 jobs/328 drawing evaluations/24 repeated
checks, retaining architecture duplicates and their compute. Native exploration
is one default proposal, four random suggestions, then GP suggestions. Every
candidate is numerically checked before training; a final unused proposal proves
the last score/cost was observed. These are maximum allocations, not completed
results. GP search seeds64173/64174 differ from calibration seeds63173/63174.

## Deadlines, output and interpretation

Calibration deadline2h; search6h; paired references2h, excluding builds/preparation
and archive compression. Each training process caps900s, evaluation120s and
proposal120s; each numerical worker caps120s. These are ceilings, not runtime
predictions. Do not silently weaken or extend them on a timeout. Evaluation and
validation overhead are recorded separately from the training cost sent to
PROTEIN. Process SPS includes training startup/writes. Native-average SPS is not
substituted from the older inconsistent realtime timer.

On complete success, open `prepared/review/combined/curves.html` and retain its
JSON/CSV data. It contains every searched candidate and both paired references,
all games/drawings, seeds, scheduled checkpoints, declines and training costs.
Calibration, proposal, raw evaluation, numerical and checkpoint proofs remain in
the packet. `prepared/execution/result.json` gives the final stage/status/time.

Return the archive path, SHA256, source revision, hardware/compiler, exit status,
calibration gate and complete-run status. Full archives live under
`build/hardware-artifacts/learning-feedback-5090.*.tar.gz` with `.sha256` sidecars.
Keep failed/blocked packets too. Do not SCP only the terminal summary.

This is a real exploratory learning/search run. Adaptive discovery seeds and two
seeds per model do not establish a paper-ready frontier advantage or SOTA. Fresh
held-out confirmation and stronger inference follow any promising point. The
new conditioned backward check does not erase the historical unconditional
H256/B2048 numerical failure or independently qualify the entire learner.
