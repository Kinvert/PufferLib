# Paper plan: practical compute-efficient visual RL encoders

Status: research plan, September 14, 2026. No additional training, new architecture search, installation, or publication is triggered by this document. Kinvert wants the project treated as a paper backed by usable software and verifiable comparisons. Architecture innovation is paused while measurement, portability, and confirmation are established.

Working title: **Practical Compute-Efficient Visual Encoders for Reinforcement Learning**. Keep the title neutral until the experiments establish the contribution. The main question is: **Can a fixed compact encoder, trained from scratch on each task, improve the score-versus-training-time tradeoff across diverse pixel environments under fair tuning and reliable evaluation?** A useful result could be a small set of budget-specific encoders; one universal winner is not a required conclusion.

## Action checklist

1. **Confirm current results — completed:** all 30 jobs and 390 evaluations passed; checkpoint/config/source audit and full paired-seed frontier analysis are in [CONFIRMATION_RESULTS.md](CONFIRMATION_RESULTS.md). Our models occupy the low-cost mean frontier and IMPALA the high-performance region; the quality-versus-Nature score advantage remains uncertain with five seeds.
2. **Separate fairness questions:** controlled identical-learner comparison versus comparable per-family tuning allowance; see the experimental-design section.
3. **Test architecture generalization:** select on development environments and freeze before retraining on untouched game identities; see generalization and environment strategy.
4. **Integrate an established external benchmark:** proposed original Procgen; its native integration remains unverified here.
5. **Deliver a usable native PufferLib component:** standard configuration, portable pixel contract, reproducible build/train/eval commands; see practical deliverable.
6. **Report the whole tradeoff:** all curves, uncertainty, failures, search cost, and tasks where a method loses; see measurement and analysis requirements.

Only item 1 is newly authorized for execution by the current request. The remaining items stay in the plan; no unseen-environment experiment has been launched.

## Claims and evidence

| Claim | Required evidence | Current status |
|---|---|---|
| Better observed Connect4 time/score tradeoff | Matched protocol, whole curves, repeated independent training seeds | Five-seed confirmation complete: low-cost mean advantage for ours; broad score uncertainty; IMPALA leads sample efficiency/high scores |
| Encoder architecture generalizes across environments | Shape selected on development tasks, frozen before training on unseen tasks, fresh weights per task | Not tested |
| Efficient practical PufferLib component | Standard configuration/build path, documented pixel contract, multiple actual environments, reproducible commands | Native trainer/encoders work for Connect4CNN; input assumptions and temporary sweep glue remain |
| Faster implementation of a given architecture | Same model/math/precision/data across implementations with numerical checks and realistic timing | Native correctness checks exist; controlled implementation-speed comparison not established |
| Optimal encoder/recurrent-core allocation | Separate controlled sweep of both components across visual and memory demands | Not tested; current core is fixed at 128 |

The first paper should center on the first three claims. Do not imply optimal encoder/core allocation from experiments that never vary the core. Do not attribute an architecture speedup to new CUDA kernels without an implementation-controlled comparison.

## What generalization means here

Implemented development control: Connect4 now has ten fixed, numerically selectable [appearance presets](../ocean/connect4cnn/REPRESENTATIONS.md). These make within-game representation tests practical in the existing native sweep workflow. Jointly selecting rendering and architecture is an optimization experiment; robustness needs a fixed architecture tested across declared representations with paired seeds/budgets. This does not replace held-out game identities for the broader paper claim.

1. **Training-seed robustness:** the same design learns reliably from different random initializations and trajectories.
2. **Within-environment policy generalization:** a trained policy works on levels or initial conditions excluded from training and validation.
3. **Across-environment architecture transfer:** the frozen encoder design is retrained from scratch on games excluded from architecture selection. All structural adaptation rules, including output projection for changing image dimensions, are fixed in advance.
4. **Policy transfer without retraining:** the same weights act in a different game. This is a different, much stronger claim and is not the initial target.

Connect4 is a development case study and integration test. Its existing evaluation set has been inspected repeatedly and is no longer a pristine final test. Keep those results, label their role honestly, and use fresh confirmation seeds and untouched environments for the paper's stronger claims. Repackaging Connect4 with different colors or board renderings tests a narrower distribution shift than transfer to a different game.

## Apples-to-apples experimental design

Use two explicitly separate comparisons:

- **Controlled architecture comparison:** same learner settings, core, observations, preprocessing, precision, environment version/opponent, rollout/replay semantics, hardware, and evaluator; vary the encoder. Match model-independent backend capabilities. Parameter and FLOP counts are reported rather than silently assumed equal.
- **Practical best-achievable comparison:** each family receives a comparable tuning resource allowance on the development set. Freeze its selected recipe before final testing. Report the allowance, consumed accelerator/CPU time, failures, and search history. Equal trial counts alone do not mean equal compute. A common recipe is useful for isolation but does not establish each baseline's best attainable performance.

Keep Nature, IMPALA, and Impoola reference definitions and every adaptation explicit. Their current native implementations are not automatically the fastest possible implementation of those models. No large universal speedup claim follows from comparing our implementation with an arbitrary slow reference. If claiming advantage over another training system, compare the same environment and model with its reasonably optimized supported backend, and separate that systems experiment from the architecture experiment.

For the initial confirmation, freeze 2–3 existing Flex configurations representing distinct frontier tradeoffs and the reference architectures. Five new paired training seeds is the proposed starting point, with the same evaluation seeds across candidates within each pair. Increase replication if uncertainty is too wide; five is not a guarantee of adequate power. Do not treat evaluation episodes, checkpoints, repeated identical trials, or games as independent training seeds.

Use identical predetermined checkpoint schedules and evaluation budgets across models. Predeclare training horizons and learning-rate schedules. A checkpoint from a 13M-step schedule is not an independent 8M-step-budget run. For budget sweeps, use the same budget and annealing rules for every family and report the distinction. Avoid giving only our winner a dense curve while giving references four checkpoints in the definitive comparison.

## What to measure and publish

- **Primary:** held-out return against measured training wall time, with all per-seed curves, uncertainty, and observed Pareto frontiers over a declared budget interval. Plot all points and failures, not just selected winners. Compute selection on development/validation data, then evaluate the frozen choices once on final tests.
- **Secondary:** return against agent decisions; native SPS; rollout, encoder, learner and environment timings; total model and encoder parameters; estimated FLOPs with conventions; peak VRAM; CPU utilization and resource allocation.
- **Practical thresholds:** time and decisions to predeclared per-task performance targets. Include non-achievement and cap information; do not average times only over successful runs. Treat measured crossings as checkpoint-resolution estimates.
- **Across-task summaries:** per-task tables first, followed by normalized aggregate curves and uncertainty. Specify normalization anchors from published protocols or frozen development rules, not the eventual winner. Use equal declared task weights. Never average raw Connect4 wins and unrelated reward scales into one objective.
- **Accounting:** separate complete experiment/search cost from the cost to train the delivered frozen configuration. Report startup, graph compilation/capture, logging, checkpoint, and evaluation timing boundaries. The published practical path should expose end-to-end wall time as well as steady-state diagnostics.

Use training-seed-aware uncertainty and appropriate task/seed aggregation rather than point estimates alone. The rliable methodology provides useful aggregate metrics, performance profiles, and uncertainty tools; it does not turn one training seed into reliable evidence. [Agarwal et al., 2021](https://arxiv.org/abs/2108.13264), [official implementation](https://github.com/google-research/rliable).

Rotate/interleave model order while running serially on an otherwise available GPU. Pin software, environment builds, and precision policy. Record hardware and competing load. Bitwise same-machine repeatability, exact restart reproducibility, and statistically similar behavior on other GPU models are distinct tests; claim only those performed.

## Environment strategy

September 14 implementation: user authorized adding native **PongCNN** while Connect4 confirmation runs. See [PongCNN](../ocean/pongcnn/README.md). It is an additional development/integration task, not ALE Pong or an untouched test game. CPU transition/pixel checks pass; shared-encoder GPU training remains to be validated after the active confirmation. State/pixel information differences and Pong's point-fraction metric are documented there. This does not replace the proposed external benchmark or establish across-environment learning transfer.

**Stage A — Connect4 confirmation:** validate the measurement pipeline, train the frozen panel across fresh seeds, and reproduce the full-frontier analysis with equal checkpoint density. Preserve the existing sweep as development evidence. Keep the stock state baseline as a separately tuned practical reference; a pixel/state comparison alone is not a CNN-architecture comparison.

**Stage B — external pixel benchmark integration:** original Procgen is the proposed primary suite because its 16 games share 64×64 RGB observations and support procedural level variation. Use the original environment implementation with pinned semantics; a custom native remake cannot inherit its scores or name as an equivalent benchmark. Original Procgen's packaged Python support is older than this project's Python 3.12 environment, so investigate its native interface/build route before committing to integration cost. It is not currently a verified native PufferLib 5.0 path here. [Procgen paper](https://arxiv.org/abs/1912.01588), [official implementation and input contract](https://github.com/openai/procgen).

Proposed development panel: CoinRun, StarPilot, BigFish, Maze, building on existing benchmark notes. Reserve the remaining 12 game identities for architecture-transfer confirmation if feasible. Freeze the split and level/seed protocol before tuning on these tasks; do not silently move a disappointing test task into development. If the full suite is unaffordable, freeze a smaller explicit split and narrow the claim accordingly. The selection is a proposal, not established representativeness.

Within each game, distinguish training levels, development-validation levels, and untouched final-test levels. Calibration that changes architecture or hypers belongs on the development games. Test-time per-environment adaptation, if permitted, must be predeclared, limited, equal across families, and reported separately from strict frozen-recipe transfer. Generalization is not demonstrated by picking a different winning CNN after inspecting each test game.

**Stage C — memory and an independent suite:** after the primary study, test a controlled memory-demand contrast or actual ALE games using an exact published observation/action-repeat/sticky-action protocol. POPGym Arcade's paired observability tasks are a candidate for isolating memory demand; integrating its accelerator environment is separate work. This extension becomes necessary if the paper claims a general encoder/core allocation result. [POPGym Arcade](https://arxiv.org/abs/2503.01450).

Any cheaper native PufferLib pixel task can help engineering validation in parallel. Label it as its own task. Native state-based Breakout is not ALE pixel Breakout, and rendering a viewer does not establish that the policy receives pixels.

## Cross-environment architecture search

The eventual user objective is to sample environments across individual architecture-search trials. Preserve that direction, but do not feed arbitrary raw task returns into the current single-task PROTEIN objective: environment difficulty would become a confounder.

Begin with frozen architectures evaluated on a balanced environment × seed × budget matrix. Then, if search resumes, use a documented balanced or stratified task assignment, task-aware observations, and a candidate-level aggregate or task-conditioned search model. Ensure each finalist has adequate coverage across development tasks. Record the sampled environment and all consumed resources. Compare against random architecture search under the same resource allowance if claiming that PROTEIN itself is responsible for the improvement. Test environments remain excluded from optimizer observations.

This likely needs search tooling beyond randomly changing the environment name in an INI. The first defensible transfer experiment does not require implementing that entire search system.

## Practical deliverable and real-world use

The first real-world application is a usable component for people training pixel agents with PufferLib. A robotics or industrial deployment is not implied by game results. If we later want a claim about a specific application, run that application's workload and report its operational constraints.

Acceptance criteria for the software artifact:

1. Configure an existing frozen encoder through normal numeric policy settings and train with the ordinary native PufferLib path. Core/search/training stay C/CUDA; any remaining Python preparation is identified as unfinished delivery work. An optional external logging/reporting sidecar is acceptable.
2. Support a documented pixel input contract beyond Connect4's hardcoded 1×36×44 layout, including the selected benchmark's RGB dimensions. Revalidate reference and candidate encoders for each supported contract. No new architecture features are required for this portability work.
3. Reproduce the paper's curves and tables from small archived receipts; obtain/load the corresponding checkpoints with hashes. Include exact commands, manifests, versions, search histories, failures, and implementation adaptations.
4. Verify one ordinary fresh-checkout use case on an environment excluded from encoder selection. Measure score, latency/throughput, resource use, and training time. A second GPU is useful external validation if available, not a substitute for seed replication.
5. Keep changes in `src/` small and idiomatic. Experimental encoders can remain outside the core until clean. No upstream push/publication without the user's authorization; no system CUDA, driver, global environment, or external venv changes.

## Paper structure

1. Problem and contribution: actual training cost versus visual-policy quality, practical constraints, bounded claims.
2. Related work: Nature/IMPALA/Impoola encoders, RL model scaling, efficient training systems, generalization, evaluation reliability. Establish novelty after the literature and experiments, not from the absence of an obvious answer.
3. Method: frozen encoder family, search space/objective, task assignment, native integration, and determinism boundaries.
4. Experimental protocol: task/level/seed splits, fairness matrix, equal tuning allowance, timing/precision, statistics, and failure accounting.
5. Results: complete task-wise and aggregate frontiers; architecture transfer; practical PufferLib use; relevant ablations. Connect4 is the development case study.
6. Limitations: task coverage, selection bias, framework/kernel sensitivity, memory demand, and tested hardware scope.
7. Reproduction artifact: runnable commands, configurations, raw results, checkpoints, and clean integration.

Do not write a successful cross-environment abstract or populate final-result tables before running the corresponding experiments. A negative transfer result is publishable evidence only if measured rigorously; it should narrow or change the claim rather than prompt hidden selection of favorable tasks.

## Immediate next work

Freeze the current experimental snapshot and candidate selection; specify a machine-readable confirmation protocol; finish the common evaluator/timing/curve workflow; remove duplicate active configuration/budget/seed trials; and audit portability to one actual external pixel environment. Start with the Connect4 confirmation panel once the protocol is concrete. Broader experiments follow measured learning and runtime calibration, not an assumed tiny timestep budget. This plan does not authorize a new long sweep by itself.

Current evidence: [complete CNN2 analysis](CNN2_RESULTS.md), [experiment history](EXPERIMENT_LOG.md), [benchmark research](BENCHMARKS_AND_SWEEPS.md).
