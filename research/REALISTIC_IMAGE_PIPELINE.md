# Shared larger-image discovery pipeline

This extends the existing native PROTEIN/panel/evaluator path. It is temporary
Kinvert research, not the eventual upstream PR. The overnight handoff remains
unfinished; current small allocations establish plumbing, not learning.

```text
Native PROTEIN suggests numeric CNN coordinates
                   |
                   v
Check the same architecture on each distinct image shape with CUDA
                   |
                   v
Freeze one complete panel: SAME architecture across all six games
FIXED learners, budgets, training seeds and appearance seed per game
                   |
                   v
Native training -> scheduled checkpoints -> every drawing of each game
                   |
                   v
Exact assigned episodes / RNG + graph-repeat-eager checks + raw receipt audit
                   |
                   v
One complete scalar objective + training cost counted once per job
                   |
                   v
Native PROTEIN observes it and proposes the next CNN
```

`candidate_panel.py` v4/v5 require an explicit `native-cnn-candidate-image-build-v1`
registry, per-game learner overlays and per-game native metadata. This binds
trainer, checkpoint registration, storage geometry and suite image contracts.
Legacy v1/v2/v3 packets retain their existing contracts.

V5 adds a measured initial checkpoint and eight trained checkpoints. Preparation
declares all nine evaluations per drawing; execution explicitly enables the
research initial-save option, and the auditor checks that environment receipt.
V4 remains unchanged when this option is omitted. Both support explicit paired
episode-seed lists and immutable development/confirmation roles. The eight-point
curve objective uses the initial policy only when no trained checkpoint is
affordable; it is measured separately and never substitutes another model's
random return. Native v5 execution remains an acceptance task.

Feedback-v4 integrates the curve objective and fixed seed roles into the native
suggest/observe loop. It rejects mismatched Nature inputs before proposal CUDA
work and independently replays the reference and candidate raw audits. A separate
completed runtime pilot is required for overnight development. Its admission
receipts and clean `budget_stop` are replayed by the auditor. Controller tests
mock policy/search processes; they do not certify real calibration or learning.

`import_image_builds.py` registers existing trainers produced by
`build_image_policy.sh`. It checks actual native info, owned compiled sources,
compiler image flags and binary hashes. Host metadata is separately frozen with
`checkpoint_preflight.freeze_images`; Nature shares the default training target,
while IMPALA/Impoola require their own per-game image builds. No Python model or
GPU query occurs during registration.

`image_validation_catalog.py build` compiles the independent CUDA/native test
libraries once per distinct H/W/channel tuple. `run` executes the existing
numerical gate for an architecture on all shapes, retaining raw arrays and
reports. `image_encoder_checks.py` binds complete coverage to the panel, checks
the actual effective encoder topology and actor/learner batches, and rejects
different hardware/toolchain receipts before training. Matching encoder checks
do not qualify the recurrent core, head, optimizer or full policy numerics.

For expensive full-resolution checks, `image_validation_collection.py` combines
individually completed shape allocations after their independent raw audits.
All shapes must use the same frozen catalog, configuration and hardware. Equal
Snake/Maze input shapes share one proof. Failed/partial allocations are rejected;
the collector does not launch CUDA or repeat a completed allocation.

`cross_game_feedback.py` v3 selects this path when `[search]` explicitly declares
`image_contract=human-geometry-v1` and `learner_mode=per_game`. It requires the
all-shape catalog, checks each native proposal before preparing its all-game
panel, and reuses the original native suggest/observe bridge. The current
plumbing objective is still the equal-game final-checkpoint fixed-anchor mean
and final monotonic training cost; the calibrated curve/coverage objective is
remaining work, not implemented by renaming this protocol.

The bounded `image5060` modes admit short jobs only: one training seed, four
evaluation slots, at most five assigned episodes, at most1024 decisions/game,
at most eight checkpoints,30s training and90s maximum evaluation deadlines. Panels
also enforce their declared cross products and memory rejection floors. These
caps do not extend historical completed allocations. Independent numerical
workers retain their bounded120s limit and reservation, clipped to the remaining
outer campaign deadline. Native all-game layout/storage/index rejection now runs
before CUDA workers; unsafe proposals stop with receipts rather than retries.

`prepare_image_inputs.py` v2 emits numeric per-game learner overlays plus the
architecture fixtures. They use four actors/H16/B16 and remain explicitly
uncalibrated. Never substitute them for the overnight learning recipes.

Training, evaluator, appearance and optimizer seeds are fixed manifest inputs,
never CNN search coordinates. Current smoke seed7/episode seed7007 are
development-only. Confirmation training replicas and evaluator lists must be
frozen separately before selection. Seen drawings and games remain development
conditions even when their episode seeds change.

The complete report retains every scheduled checkpoint/drawing, duplicate,
decline, failure and missing cell; training costs are charged once per job.
Both host storage floors and short process SPS are limited evidence. Full
learning calibration, baseline math, targeted Nature comparisons, approximately
12-hour admission/stopping and the final5090 instructions are still required.

The first eight-checkpoint image packet actually failed a30s Maze evaluation,
after six trainings/280 evaluations/ten repeat checks. Its audit retains that
failure. A separate final-Maze graph/repeat/eager probe completes all five
assigned episodes with exact rows in36.6–36.8s/process. The three-proposal
feedback smoke now passes in600.494s:18 trainings/123 all-drawing evaluations/
36 repeat-eager checks/270 numerical instances/1080 calls, final native replay3
and complete raw receipt audit. It uses one episode/drawing within900s, not a
changed Maze horizon. The rounded duplicate remains; no quality claim follows.
Read[the overnight protocol](IMAGE_OVERNIGHT_PROTOCOL.md) for remaining work.
