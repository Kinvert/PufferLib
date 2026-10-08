# Actual native development preparation, October 7

The real all-six-game/all-41-drawing score and PROTEIN feedback already worked in
[the completed 5090 canary](../feedback-canary-5090-20261007/README.md). This new
work removes preparation restrictions that prevented a useful broader search:
per-game decision budgets/cadence, per-game rollout/minibatch settings and
stage-two/three search controls with a depth-one initial architecture.

Actual G240 GPU-free command:

```bash
bash research/run_feedback_5090.sh prepare-development \
  build/cross-game-feedback/development-preparation-20261007
```

Passed 51 host/config/scalar/mock regression checks and four actual native
descriptor/input-rejection checks. Fresh native optimizer bridge, six normal
float32 game targets and three policy metadata tools compile. The descriptor
registers all 18 CNN coordinates and explicitly reports no GPU query or policy
execution. Feedback-v2 preparation freezes six native scalar learner proofs;
Maze's example horizon64 gives rollout4096, other games rollout2048. A Flappy
budget override is recorded independently. All update counts are positive,
budgets exact, and checkpoint cadences aligned.

Separately prepared and inspected a real panel-v3 with two distinct architectures
(one-stage C8 and a three-stage residual/pooling/global-pooling fixture), two
seeds, all six games and all 41 drawings: **24 full-policy registrations, 82
shared suites, 656 checkpoint evaluations allocated, zero executed**. Native
registration includes the encoder, head and recurrent core parameter layouts.
These are constructor/configuration and environment-manifest checks, not neural
forward/backward checks, live vector initialization or full-trainer fit tests.

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/candidate_panel.py prepare \
  --registry build/cross-game-feedback/development-preparation-20261007/games/registry.json \
  --policy-metadata build/cross-game-feedback/development-preparation-20261007/metadata \
  --candidate happy-cat-1=research/recipes/cross_game_feedback_prepare.ini \
  --candidate mystic-tree-2=research/recipes/panel_prepare_three_stage.ini \
  --steps 1048576 --checkpoint-steps 262144 \
  --task-budget flappycnn=2097152:524288 \
  --learner-recipe mazecnn=research/recipes/feedback_maze_prepare.ini \
  --per-game-learners --seeds 61173 61174 \
  --out build/cross-game-feedback/development-preparation-20261007/panel
```

Both new packets pass offline inspection. The received original 5090 feedback-v1
packet also remains readable under the updated inspector without modifying its
bytes or absolute paths. Generalized learners are rejected from all bounded
smoke/mini/canary routes; the existing launcher `run` refuses feedback-v2 before
GPU work. No GPU query, model computation, training, dataset, dependency or system
change was performed for this work.

The recipes are **uncalibrated preparation examples**, not selected learners or a
scheduled eight-trial learning experiment. Normalization anchors, evaluation
caps, informative training budgets and actual architecture/batch math remain
gates. Read [the next experiment](../../NEXT_FEEDBACK_EXPERIMENT.md).

Small receipts retain test output, native descriptor, scalar configs/results,
plan hashes and all 24 registration records. Original native binaries, full
source/config/manifests and complete input ledgers remain under the ignored build
directory above. This compact receipt set is not a standalone reconstructed
packet or complete vendor/system/toolchain closure certificate. Build revision
and dirty-worktree records describe the actual pre-commit preparation state.
