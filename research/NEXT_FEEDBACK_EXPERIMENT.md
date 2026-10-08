# After the successful 5090 feedback canary

October 7, 2026. The native cross-game loop executes and audits on the 5090.
[Received evidence](results/feedback-canary-5090-20261007/README.md) covers
three trials, but every trial constructs the same C16 architecture and nearly
all final game scores are zero. That is plumbing evidence, not a useful search
or a comparison against Nature. Do not restart the completed canary.

## Implemented: per-game calibration and broader architecture controls

The feedback wrapper now accepts `[task.ENV]` sections containing `steps` and
`checkpoint_steps`. `[search] learner_mode=per_game` permits fixed per-game
rollout/minibatch geometry through `--learner-recipe ENV=INI`. Both are forwarded
into every candidate panel, with exact native rollout alignment and positive
optimizer updates checked before the first proposal. LR/clip and learner geometry
remain fixed per game across candidates; they are never CNN search coordinates.
The CNN remains common across all six games, with H128/L1 and random weights.

The opt-in feedback-v2 / panel-v3 preparation contracts:

1. Freeze explicit positive per-game decision budgets/checkpoint schedules and
   forward them into the existing panel's resolved `task_budgets` map. Require
   exact native rollout alignment and positive effective optimizer updates.
2. Permit declared per-game learner/vector geometry in that route, preserving
   identical settings for every compared architecture within each game. Keep
   all existing canary/smoke/mini allocation bounds and old packet meanings.
3. Freeze native scalar geometry/configuration proofs and full-policy
   registration. Separate memory estimates and actual shape/batch numerical
   qualification remain required before a serious search; registration is not
   full-trainer fit or neural correctness acceptance.
4. Record every resolved learner, update count, schedule, cost and failure;
   separate calibration choices from subsequent architecture search and
   independent confirmation. Build/test/prepare locally without new GPU work.

Fifty-one host/config/scalar/mock regression checks pass, including a three-trial
six-game feedback loop with different per-game budgets and rejection of rehashed
learner edits, invalid cadence and zero effective updates. These tests do not
execute a new GPU feedback loop. Existing v1/v2 packet meanings and bounded
canary/smoke/mini routes stay unchanged; generalized learners require development
mode and cannot be launched through the bounded canary launcher.

[Actual native preparation](results/feedback-development-preparation-20261007/README.md)
also passes four native host checks and fresh optimizer/six-game/metadata builds.
Two distinct one-stage/three-stage architectures register all 24 policies across
two seeds/six games, with 82 suites and 656 evaluations allocated, none executed.

GPU-free preparation example, without starting another canary or a search:

```bash
bash research/run_feedback_5090.sh prepare-development
```

This builds and prepares all 18 CNN coordinates, an explicit Flappy budget
override and a Maze horizon override. The recipe, budget, anchors and learner
override are **uncalibrated examples**, not a recommendation to launch its eight
trials. The launcher prints no GPU run command for this preparation. Do not use
`run` on that packet; that command remains the old bounded canary route.

This is temporary discovery infrastructure for Kinvert's research fork. The
selected architecture's eventual upstream implementation remains a separate
minimal native PufferLib PR. No system/CUDA/Torch changes or CPU CNN testing.

## Numerical gate at the shapes and batches actually used

Before allocating a serious search, qualify the allowed C8/C16 graphs and
Nature at the actual rollout/learner batch sizes. H128/B1/3/64 success doesn't
certify arbitrary convolution widths or full learner batches. The H256/B2048
strict reference comparison remains failed; preserve it. Its proposed separate
layer-forward / backward-on-actual-float32-branches protocol is not implemented.
Don't waive tolerances or turn a successful training run into math acceptance.

Use independent CUDA forward/backward references and retained arrays/process
proofs. A fresh bounded GPU allocation must declare its complete case matrix;
do not rerun old completed/failed packets. Full head/core/optimizer/reload and
concurrency checks remain separate from encoder checks.

## First informative experiment: frozen quality versus Nature calibration

Start with two frozen architectures: current C16 quality and adapted Nature,
H128/L1, float32, random weights separately for each game/seed. Use paired
development seeds and the same candidate learner/budget/drawings/evaluation
within each game. Give both architectures the same declared calibration
allowance; avoid tuning exclusively for one architecture and presenting the
other's loss as an architectural result.

Progressively test longer decision budgets, with all scheduled checkpoints and
losses retained. Historical Connect4 evidence extends to13.312M decisions; Pong
has learning around4.194M under some recipes; stock Flappy learned at19.923M.
Those are candidate resource scales from different protocols, not sufficient
budgets guaranteed for the current mixed-drawing/common-core setting. Breakout,
Snake and Maze need measured pixel-learning calibration, especially Maze's
memory horizon. Do not multiply a large speculative six-game budget into a
hundred-candidate campaign.

Calibrate Pong/Breakout administrative evaluation caps from retained policies,
preserving censoring/partial-score data. Use exact assigned episode identities,
all fixed drawings, and deterministic paired graph/repeat/eager checks. Pilot
evaluation cost separately; all-drawing evaluation can dominate training time.

Choose and freeze each game's learner, adequate budget and normalization anchors
using a declared development rule across both controls. Preserve unsuccessful
recipes. If a game remains unlearned, repair/calibrate its recipe or record the
unresolved task; do not silently drop it or optimize an all-zero aggregate.

## Then make the architecture search genuinely varied

Use an explicit initial allocation of distinct registered architectures, or a
declared native exploration warmup, to exercise small and larger choices before
adaptive GP suggestions. Retain architecture duplicates and their compute; do
not report different normalized coordinates as different constructed graphs.
An initial fixed C8/C16 panel is transparent; broader depth/channel/kernel/
stride/pool/residual/projection choices require their own construction/math gates.

The validator now uses the maximum explicitly swept depth, so a depth-one default
can expose all 18 coordinates across up to three stages. Fixed depth still rejects
inactive-stage search dimensions. Audits bind every proposed INI coordinate,
including temporarily inactive stages, while duplicate detection uses the
effective constructed architecture. Random warmup can explore before GP feedback;
the preparation example declares four random suggestions. Actual diverse GPU
proposals, learning and numerical qualification remain pending.

Keep the CNN architecture common across all six games. Other hyperparameters,
budget/cadence and appearance/evaluation assignments stay frozen per game across
suggestions. Every completed candidate gets all-game/all-drawing feedback;
retain complete per-game curves alongside the normalized development objective.

## Evidence after discovery

Evaluate selected common architectures against Nature under the same per-game
recipes and full budget/checkpoint schedules, using fresh paired training seeds
and independent evaluation suites. Report every drawing/game and the complete
observed score/time fronts with uncertainty, retaining weak cases. A favorable
point needs held-out replication; the search's best result is not confirmation.
Add IMPALA/Impoola only with equally declared learner/memory/math/backend gates;
don't claim general SOTA from a Nature comparison alone.

This document schedules no GPU work or large sweep. Next prepare a measured,
bounded paired quality/Nature calibration allocation, then freeze adequate
per-game learners/budgets before adaptive architecture discovery. The existing
three-trial canary is complete and must not be restarted as the next experiment.
