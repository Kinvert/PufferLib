# Paper plan: practical compute-efficient visual RL encoders

Status: research plan, updated September 15, 2026 after the 5090 Pong evaluation/backend audit. Kinvert wants usable native PufferLib software and an arXiv paper with defensible evidence. Architecture innovation is paused while measurement, portability, and confirmation are established. The ordered implementation milestones below supersede older proposed sequencing; editing this plan does not launch experiments.

Working title: **Practical Compute-Efficient Visual Encoders for Reinforcement Learning**. Keep the title neutral until the experiments establish the contribution. The main question is: **Can a fixed compact encoder, trained from scratch on each task, improve the score-versus-training-time tradeoff across diverse pixel environments under fair tuning and reliable evaluation?** A useful result could be a small set of budget-specific encoders; one universal winner is not a required conclusion.

The practical delivery target is the [native CNN constructor and search workflow](CNN_CONSTRUCTOR_PLAN.md). There are two distinguishable research claims: a frozen encoder's tradeoff advantage, and a search procedure finding better configurations for a declared resource budget. Neither is established yet. The software should remain useful when the existing compact model loses; the paper must not depend on forcing that model to win.

## Next actions — ordered implementation milestones

**Current priority after Kinvert's full-frontier clarification:** execute the gates in [CONNECT4_CLAIM_PROTOCOL.md](CONNECT4_CLAIM_PROTOCOL.md). Compare all four complete curves under one identical learner recipe, 13.312M-decision budget and checkpoint cadence. No three-time restriction, short wall cap or per-family recipe selection. Native checkpoint completion receipts, exact evaluation allocation and calibrated simultaneous frontier analysis precede fresh-seed confirmation; the replication count remains to be justified and frozen. Pong v2 remains supporting work and must precede further Pong quality claims.

**Current evidence:** Connect4 five-seed confirmation and the 5090 Pong replication/profile work are complete. The [received audit](PONG_EVALUATION_AUDIT_RESULTS.md) establishes sparse Pong match completion despite high throughput; telemetry cleanup preserves results but has no measured speed advantage. Nature beats ours under Pong recipe B; a quality advantage is not established. Original missing outcomes remain missing. Full 5090 trace/checkpoint bytes are retained remotely; transferred summary/diagnostic receipt checks are separately identified.

| Order | Work | Concrete result and completion check |
|---|---|---|
| 1 | Implement Connect4 timing and evaluation gates | Native fully-written checkpoint times, identical dense step cadence, deterministic exact episode allocation and saved-model tests |
| 2 | Audit fairness and freeze the full-frontier protocol | Review backend/architecture adaptations; freeze one shared resolved learner recipe/source; calibrate simultaneous analysis and independent training-seed count |
| 3 | Measure the complete Connect4 frontier | All four models finish the same 13.312M decisions; retain every checkpoint/seed, declines and failures; report supported and unresolved regions without significance-driven additions |
| 4 | Deliver a portable constructor | Ordinary numeric INIs, configurable image channels/dimensions, validated shapes, native build/train/eval/reload and export; another clean checkout reproduces the workflow |
| 5 | Integrate an established benchmark | Original environment semantics and train/validation/test splits preserved; first native original-Procgen feasibility result before committing to a broader suite; shared 64x64 RGB input tested |
| 6 | Run the search-method study | Small-first PROTEIN versus random/unrestricted search and competitive references under declared tuning resources; full search costs, failures, recipe effects and exported configurations retained |
| 7 | Freeze and run final paper confirmation | Candidate selection, budget schedules, test tasks, seed count/stopping and analysis fixed in advance; paired uncertainty and complete results support only the claims actually demonstrated |

### Supporting task: Pong evaluation v2, with saved models

Use the [received proposal](results/pongcnn/eval-profile-audit.VjfOaDBg/audit/PONG_EVALUATION_POLICY_PROPOSAL.md) as design input, not an already implemented specification. Finish a short v2 protocol before writing the implementation:

- Assign exactly 512 episode IDs, eight per each of 64 slots, and an explicit deterministic seed mapping. Fast slots cannot contribute extra sampled episodes. Finished-slot masking must preserve active-slot behavior and recurrent state/reset semantics; test interactions with action RNG and vector execution rather than assuming isolation.
- Record whole-match decisions, current-rally decisions, score, seed/ID, completion/truncation status and slot contribution. Current Pong `tick` resets at each point, so it is insufficient as a match counter.
- Choose the numeric per-match decision cap from a declared duration-only calibration on existing development checkpoints, with an explicit compute ceiling; freeze it before new comparative quality evaluation. Record the cap rationale and check how truncation affects the usefulness of the result. A short cap that leaves broad score bounds is not a solved quality benchmark.
- At a cap, record truncation without awarding a win or inventing a final score. Preserve current environment physics, scoring and observations. Report completion fraction plus identification bounds for the complete-match estimand; do not claim an unbiased complete-case mean. Keep a separate infrastructure/process timeout and distinguish it from planned episode truncation.
- Specify reset/seeding behavior and a fixture-based comparison of genuinely matched completed trajectories. New episode allocation is a new estimator; it need not reproduce v1's pooled score, and its scores must not replace v1 entries.

Implement native accounting/masking using existing interfaces, with minimal changes to `src/`; inspect the 5090 runner changes before duplicating or merging them. Preserve archived source as evidence instead of blindly applying a whole archive. Keep ordinary training and unrelated environments on their existing paths. Package the tested telemetry/portability fixes and v2 work as reviewable changes with exact source identity; reconcile both machines before their next shared experiment.

Acceptance tests: unique/exact episode allocation; repeatable ID/seed mapping and results; cap/terminal boundary precedence; recurrent reset correctness; inactive-slot isolation; complete/truncated/error distinction; correct bounds; no loss of assigned episodes on process failure; deterministic repeated checkpoint evaluation. Use fixtures and bounded saved-model checks first. Then freeze one balanced diagnostic checkpoint panel with common rules for every family, including previously slow examples. **This milestone requires no full training rerun.** Store v2 results in new directories and preserve all v1 failures/declines.

Only after v2 is dependable should it be used for further quality comparisons. Baseline layer attribution can proceed independently; kernel rewrites should follow measured attribution. Connect4 rendering variants are useful regression/appearance tests but are secondary to this evaluation milestone and do not substitute for held-out game identities.

### Paper writing alongside implementation

Draft motivation, contribution alternatives, native software design, experimental protocol and limitations now. Keep current Connect4/Pong findings as development evidence with explicit source and evaluation versions. Add final results after the method/protocol is frozen; do not retrofit the primary claim to a favorable test checkpoint. Refresh related work before making a novelty/SOTA claim. A useful software artifact and a scientifically novel search result are separate requirements; neither guarantees the other.

## Action checklist

Additional potential software contribution: [native CNN constructor and per-environment frontier search](CNN_CONSTRUCTOR_PLAN.md). This can complement the fixed-CNN result; its search-budget comparisons and practical-delivery requirements are distinct from claiming that one architecture generalizes everywhere.

1. **Confirm current results — completed:** all 30 jobs and 390 evaluations passed; checkpoint/config/source audit and full paired-seed frontier analysis are in [CONFIRMATION_RESULTS.md](CONFIRMATION_RESULTS.md). Our models occupy the low-cost mean frontier and IMPALA the high-performance region; the quality-versus-Nature score advantage remains uncertain with five seeds.
2. **Establish a defensible advantage over Nature — open:** substantially reduce uncertainty using the thesis-defense standard below, and separate controlled identical-learner comparisons from comparable per-family tuning allowances. Completing item 1 did not establish this claim.
3. **Test architecture generalization — partial development evidence:** the Connect4-selected architecture has been retrained on Pong, with strong recipe dependence and no demonstrated advantage over Nature. Untouched-game confirmation remains open; see generalization and environment strategy.
4. **Integrate an established external benchmark:** proposed original Procgen; its native integration remains unverified here.
5. **Deliver a usable native PufferLib component:** standard configuration, portable pixel contract, reproducible build/train/eval commands; see practical deliverable.
6. **Report the whole tradeoff:** all curves, uncertainty, failures, search cost, and tasks where a method loses; see measurement and analysis requirements.

Item 1 was the initial execution authorization and is complete. The user subsequently authorized Pong development experiments with the existing CNN locked and training-hyperparameter search; see [PONG_HYPER_SWEEP.md](PONG_HYPER_SWEEP.md). That bounded development search does not authorize all remaining paper experiments or replace fresh-seed confirmation. No untouched-game final test has been launched.

## Required evidence standard: defend the claim like a thesis

Explicit user requirement, September 14, 2026: **make our advantage over Nature much more certain, and prepare the arXiv paper as though every claim must survive a thesis defense.** Cover plausible alternative explanations and retain evidence a reviewer can independently check. The objective is to determine whether the advantage holds; an inconclusive or negative result must narrow the claim rather than trigger selective reporting.

Current evidence is insufficient for a firm score-superiority claim: quality's final paired advantage over Nature is +6.31 percentage points, with a pointwise 95% bootstrap interval of −0.07 to +13.00. The observed mean wall-time frontier is encouraging, but uncertainty in a final score difference is not a test of the whole frontier. See [confirmation results](CONFIRMATION_RESULTS.md) and [measured source revision](BENCHMARK_STATE.md).

Before presenting an advantage over Nature as established:

1. **Specify the claim before the next confirmation.** Freeze candidate selection, baseline recipes, tasks, budget region, primary metric, practically meaningful improvement, and analysis. Distinguish higher score at a fixed wall budget from less time to a fixed score, sample efficiency, and implementation throughput. A favorable checkpoint discovered afterward is exploratory evidence.
2. **Plan adequate independent replication.** Use development variability to choose a justified seed count or interval-precision target; document the calculation and assumptions. Fix the sample size or a statistically valid sequential stopping rule before inspecting new results. Do not keep adding seeds until a confidence interval happens to exclude zero. Pair seeds across candidates and account for both training variability and evaluation noise.
3. **Quantify uncertainty for the actual claim.** Retain complete curves and failed/non-achieving runs. Use an appropriate predeclared comparison for the selected budget region, with simultaneous inference or multiplicity control when claiming advantages across many budgets/models/tasks. Pointwise bars alone do not establish frontier-wide dominance. Report effect sizes and practical relevance, not just statistical significance.
4. **Challenge baseline fairness.** Verify Nature's architecture, adaptations, numerical correctness, initialization, learning schedule and effective settings. Report the common-learner comparison separately from a comparison with comparable tuning resources, including competitive smaller baseline variants. Audit backend efficiency so a slow reference implementation cannot masquerade as an architectural advance.
5. **Test alternative explanations and scope.** Check observation/preprocessing equivalence, timing boundaries, hardware contention and parameter/core differences. Use fixed-design appearance tests and additional games to test whether the result depends on Connect4's rendering or rules. A second GPU needs all competing methods rerun on it. Generality requires untouched tasks; a narrow Connect4 claim must remain labeled as such.
6. **Make every published claim traceable.** Link claims to exact code revisions, resolved configs, seeds, raw measurements, analysis and reproducible commands. Reproduce the reported comparison from a clean checkout and independently audit the analysis before submission. Preserve search costs, exclusions, failures and contradictory results. A SOTA claim additionally requires a current relevant-baseline review and a clearly defined benchmark/protocol; beating Nature alone does not establish it.

The paper may claim a reliable improvement in a useful, explicitly bounded portion of the Pareto frontier if the evidence supports it. It need not claim superiority everywhere. These requirements are open research work, not a declaration that the present results satisfy them or authorization for an unspecified training campaign.

## Claims and evidence

| Claim | Required evidence | Current status |
|---|---|---|
| Better observed Connect4 time/score tradeoff | Matched protocol, whole curves, repeated independent training seeds | Five-seed confirmation complete: low-cost mean advantage for ours; broad score uncertainty; IMPALA leads sample efficiency/high scores |
| Encoder architecture generalizes across environments | Shape selected on development tasks, frozen before training on unseen tasks, fresh weights per task | Connect4-selected shape retrained on Pong; recipe dependence and seed failures measured, advantage unproven; further untouched-task confirmation open |
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

September 14–15 implementation: native **PongCNN**, fixed-architecture learner search and the 5090 two-recipe/five-seed replication are complete. See [PongCNN](../ocean/pongcnn/README.md), [search protocol](PONG_HYPER_SWEEP.md), [replication](PONG_5090_RESULTS.md) and [evaluation audit](PONG_EVALUATION_AUDIT_RESULTS.md). The initial common-recipe quality pilot scored zero after 13.312M decisions; subsequent learning depends strongly on recipe and seed, and eight replication evaluations remain incomplete. Pong is a development/integration task, not ALE Pong or a pristine test game. State/pixel information differences and the point-fraction metric remain documented. This does not replace the proposed external benchmark.

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
