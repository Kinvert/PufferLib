# Connect4 full-frontier comparison: design and execution gates

October 2 groundwork update: [CLAIM_PIPELINE.md](CLAIM_PIPELINE.md) documents the
implemented opt-in native evaluator, monotonic completed-checkpoint receipts,
matched configuration/audit tooling and full-curve reporting. All four binaries
compile, but GPU measurement acceptance remains pending. Candidate joint bands
undercover in preliminary synthetic diagnostics; no inferential gate is closed
and no new confirmation run has started. Preserve the protocol below as the
target, rather than treating implementation or canary preparation as acceptance.

September 15, 2026. **Kinvert's correction: compare the full observed Pareto frontier with the same training regime for every encoder.** This supersedes the earlier 60-second target, three reference times, 80-second training cap and per-family learner selection. No new confirmation training has started. [Machine-readable design](connect4_claim_design.json) records what is fixed and what still needs validation before launch.

## Question and scope

Under one shared training recipe, where does the frozen compact encoder improve held-out Connect4 win rate versus training time compared with Nature, IMPALA and Impoola, across their complete measured learning curves?

Measure from the first checkpoint through each model's full training budget. Keep the high-score region where a larger model may win. No three time ranges, preferred 60-second point or post-result restriction to our favorable region. Report score versus decisions alongside score versus elapsed time so sample efficiency and execution speed remain distinguishable.

The result is a full **observed** frontier for these models, checkpoint spacing, recipe, task and hardware. Finite checkpoints do not establish a continuous/global optimum. A robust advantage in part of this frontier would be useful; beating Nature on Connect4 alone is insufficient for a SOTA or cross-task claim. Preserve Pong's unfavorable and unresolved results in the paper.

## Identical training regime

Freeze quality, Nature, IMPALA and Impoola as defined in the previous [confirmation](CONFIRMATION_PROTOCOL.md). Quality is one 16-channel SAME 7x7/stride-4 stage, no pooling/residual, flatten and projection 64. All use H128/L1. Expected total parameters are 160736, 138528, 270496 and 151712 respectively; encoder/projection parameter differences are documented, not silently described as parameter matching.

Use the original 7x6 Connect4 game, representation 0, 1x36x44 pixels, seven actions, the same scripted opponent and float32. The primary learner recipe is `ocean/connect4cnn/compare.ini`, including LR .001, gamma .8, replay ratio 1, 64 agents, horizon 32 and minibatch 2048. Capture and compare every resolved non-encoder setting, including inherited defaults. Only encoder construction and its necessary per-build selection differ.

Every model receives **13,312,000 decisions**, identical step-based LR annealing over that horizon, optimizer/loss settings, rollout/update counts, checkpoint steps, paired training seeds and evaluation allocation. Use balanced predetermined model order within training-seed blocks on the same idle 5090 host. Record hardware, clocks/load, compiler, source and binary identity. Equal work gives different elapsed times; that difference is an outcome. No short wall cap may intentionally prevent IMPALA/Impoola from completing this budget. Infrastructure timeout limits must accommodate all families and retain failures.

Do not select different learner recipes for different models in this comparison. One shared recipe establishes a controlled result conditional on that recipe, not best-possible tuning for every architecture. A later sensitivity study can cross the same additional recipes with **all** models and report each matched panel separately. It must not replace the primary panel with each family's favorite recipe.

## Complete checkpoint curves

Proposed cadence: save every **262,144 decisions** (128 updates at 2048 decisions/update), plus the final 13,312,000-decision checkpoint: **51 checkpoints per training run**. This is denser than the previous 13-checkpoint panel. Validate storage, timing overhead and evaluation cost before freezing this cadence; any revision must apply to all models before confirmation. An initial untrained checkpoint is optional diagnostic work outside this count, not an assumed zero score.

Reuse native step-based checkpointing. Record a monotonic time origin immediately before native process launch and the completion time of each fully written checkpoint. Include startup, model construction, graph capture, rollouts, updates and checkpoint writes. Publication receipts must never mistake the beginning of a write for model availability. Keep whole-process SPS and native adjusted SPS distinctly labeled. Report build, evaluation and total experiment costs separately; evaluation runs outside the timed training trajectory.

Keep every chronological checkpoint and every seed curve, including declines. A checkpoint is a predeclared fixed-step training option; an early checkpoint uses the full-horizon LR schedule and is not an independently annealed short-budget run. If independently trained budget variants are later added, cross the same budget grid with every family and preserve those separate trajectories.

Plot mean elapsed time and mean held-out score at each predeclared checkpoint, over the complete paired training-seed panel. Calculate the nondominated envelope of those mean points, while retaining dominated points and raw curves. Never construct a frontier by taking the best seed at each budget. A descriptive test-score envelope is not a deployable checkpoint-selection rule: exporting one selected checkpoint requires independent validation selection, with test evaluation reserved for assessment.

For any supplementary fixed-time summary, choose the latest fully available checkpoint by time, independently of test score. Missing early checkpoints remain missing. Do not interpolate a learned policy between checkpoints, erase a later decline, or carry the best test score forward as though a user could know it during training.

## Replication and full-frontier uncertainty

Independent **training seeds** are the primary replication unit. Pair the seed list across all four models, retaining complete model/checkpoint curves within each block. Multiple held-out evaluation seeds and assigned episode IDs measure evaluation noise; they do not turn one trained network into multiple independent training runs.

The earlier suggestion of 40 training seeds is **provisional resource planning**, not a justified final sample size for whole-frontier inference. The old nine-contrast normal-power calculation does not size this new design. Choose and freeze the sample size, seed lists, practical effect threshold, analysis and stopping rule using development-only variance/precision calibration before any confirmation outcomes. No significance-driven additions or seed replacements.

The analysis must address simultaneous uncertainty across the full predeclared model/checkpoint panel, including uncertainty in both elapsed time and score. Candidate implementation: paired whole-seed resampling that retains time/score correlation, with a simultaneous confidence construction and frontier membership recomputed for each resample. Calibrate coverage on synthetic null, crossing, tied and declining curves before adopting the method. Ordinary pointwise error bars or frontier-membership frequencies alone are not proof of simultaneous dominance. Exact inference details and finite-sample limitations remain an explicit prelaunch gate.

Report the empirical frontier and the regions with supported, unresolved or contrary comparisons. A claim about a region discovered in the final plot requires inference that already covers selection over the full panel; otherwise label it exploratory. No requirement to beat all references everywhere, and no permission to ignore a cheaper reference checkpoint that achieves higher quality. Missing outcomes must remain visible with appropriate bounds or an explicit unresolved comparison; never silently drop failed seeds or impute zero quality.

## Exact held-out evaluation

Use a proposed **1,024 assigned episodes per checkpoint**, across multiple predeclared evaluation seed blocks, with the same allocation for every model/checkpoint within each training-seed block. Freeze the seed-block mapping before launch. Assign quotas/unique episode IDs so faster slots cannot contribute extra games. Record outcomes, decisions, slot/episode IDs and deterministic environment/policy RNG and recurrent-reset behavior. Completed wins/losses/draws/invalid actions receive their actual outcomes; missing records are errors, not losses.

Connect4 matches should finish within 21 agent decisions from an empty board. The native environment tests now check this bound, two-piece progress for nonterminal moves, reset boards/rewards and exact terminal log counts. Original/pixel parity, repeats and ASan/UBSan passed for all ten representations with 4,096 transitions each ([receipt](results/connect4cnn/claim-design-20260915/environment-tests.txt)). This validates game invariants, not evaluator quotas or GPU recurrent reset. Test exact allocation, unique IDs, reset/RNG behavior, inactive-slot isolation, terminal boundaries and repeated saved-model evaluation before use. Preserve the pooled v1 evaluator's historical results separately from this proposed evaluator.

## Execution gates and resources

1. Reconcile local/5090 source and audit baseline architecture adaptations, float32 numerical/all-gradient correctness, repeatability, projection/core and backend efficiency. Investigate avoidable implementation cost before interpreting it as architectural inferiority. Freeze the same measured source for all models; retain known limitations.
2. Validate native checkpoint completion receipts and the identical step cadence with bounded canaries; measure overhead for all families. Finish exact episode allocation and saved-model tests. No long retraining is needed to validate the evaluator.
3. Implement/calibrate the full-frontier analysis using development and synthetic data. Freeze sample size, evaluation seeds, cadence, complete resolved configs, inference and failure/stopping rules in an immutable launch manifest.
4. Run the complete common-regime panel, save all receipts and generate reproducible full curves, uncertainty, tables and failures. Independently audit the report before claiming an advantage.

At the historical single-seed 5090 speed, four full trajectories total **18m19s** of training. Forty paired seeds would therefore be roughly **12h13m training alone**, before denser-checkpoint overhead, evaluation and builds. At the proposed cadence this would mean 160 training jobs and 8,160 checkpoint evaluations. These are planning estimates, not a launch commitment or a measured ETA. Measure canary costs before final scheduling. No long run is launched by this document.

The previous [time-slice planning report](results/connect4cnn/claim-design-20260915/REPORT.md) and its generator remain historical development artifacts. Their snapshot calculations and game tests are useful, but their nine-contrast power table and old target do not govern this protocol. The [paper plan](PAPER_PLAN.md) retains appearance tests, untouched tasks, broader references and native-tool delivery as subsequent evidence requirements.
