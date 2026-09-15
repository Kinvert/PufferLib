# Connect4 low-time claim: design and execution gates

**SUPERSEDED, not executed.** Retained to document the abandoned design. The authoritative replacement is [the full-frontier protocol](../../../CONNECT4_CLAIM_PROTOCOL.md); relative links below preserve their original text and may no longer resolve from this archive.

September 15, 2026. This specifies the next focused study, using existing results as development evidence. **The claim, selection rules, budget grid and replication design are fixed here; the final source and selected learner recipes cannot be frozen until the gates below pass. No new development or confirmation training has started.** Machine-readable design: [connect4_claim_design.json](connect4_claim_design.json).

## Claim we will actually test

On the specified RTX 5090 host, original Connect4CNN representation 0, scripted opponent and native float32 implementation, does the frozen quality encoder with its development-selected learner deliver a higher mean held-out win rate by **60 seconds** than adapted Nature, IMPALA and Impoola at **each of 30, 45 and 60 seconds**, under the same declared learner-selection procedure?

This is one selected point versus nine reference points on a **predeclared discrete time grid**. It is not superiority over all times, architectures, optimized implementations, games or hyperparameters. The primary comparison concerns exported encoder/learner configurations, not a causal architecture-only effect with different learner recipes ignored. A state-input policy is outside this pixel-encoder panel; retain the historical state results separately and do not claim to beat vanilla state-based Connect4 training.

Passing the test would support a bounded native-training tradeoff claim. It would not establish SOTA, an optimal continuous frontier, or that no unsampled reference checkpoint/recipe can win. A current related-work/benchmark review and stronger baseline family coverage remain necessary for those claims.

## Why this target, and what the old data cannot prove

[Reproducible development analysis](results/connect4cnn/claim-design-20260915/REPORT.md) selects the latest checkpoint available at each budget without looking at its score. Input hashes, every selected point and missing records are retained. Four CPU safeguards cover declines, missing-versus-zero scores, invalid values and nonmonotonic times.

- Historical five-seed RTX 5060 results at 80s: ours 69.91%, Nature 56.32%; all five paired differences positive. At 100s: ours 77.36%, Nature 68.04%. The 5060 observations have different source/toolchain/hardware and are not pooled with the 5090.
- The single recorded 5090 seed at 60s: ours **71.58%**, Nature **59.78%**. The earlier 53.96s quality checkpoint scored 72.14%; we deliberately retain the later decline rather than cherry-pick it.
- The larger models have no checkpoint available at some early historical budgets. Those entries are missing, not zero. Coarse checkpoints and a full 13.312M learning-rate schedule mean these summaries are not new short-budget experiments.
- Five seeds give an unstable variance estimate; one 5090 seed cannot estimate training-seed variance. All existing observations used for this design become development data for the new claim.

## Frozen architecture/task panel

Use quality, Nature, IMPALA and Impoola as already defined in [the previous confirmation](CONFIRMATION_PROTOCOL.md). Quality remains one 16-channel SAME 7x7/stride-4 stage, no pooling/residual, flatten, projection 64. Every model retains H128/L1. Expected total parameters: quality 160736, Nature 138528, IMPALA 270496, Impoola 151712.

Original 7x6 game, representation 0, 1x36x44 pixels, seven actions, identical opponent/observation/reward semantics and float32 precision. Keep `compare.ini` rollout/core/optimizer settings except the declared LR/gamma challenge. No architecture, representation, initialization or kernel tuning using confirmation outcomes.

## Gate 1: native timing and exact evaluation

**Checkpoint availability:** establish one monotonic time origin immediately before native process launch. Include process startup, model construction, graph capture, rollouts, updates and checkpoint writing. Exclude prior compilation and later evaluation, reporting their costs separately. Resource-adjusted native uptime is not the primary timer.

Create an initial checkpoint and subsequent checkpoints at a proposed one-second wall cadence, checked at completed update boundaries. Record each checkpoint's step count, hash and ready time after writing finishes. Use a temporary file and atomic publication or an equivalent validated completion receipt; observing the beginning of a write is not availability. Select the last fully published checkpoint whose ready time is <= the requested deadline. Preserve missing availability and partial files as failures rather than reporting zero reward. Test equality/boundary cases, slow writes, startup delay and out-of-order filesystem notifications. The CPU historical selector does not implement this native clock/checkpoint machinery.

Each trajectory keeps the existing linear step-based annealing horizon of **13,312,000 decisions**; the measurement process stops at the earlier of completing that horizon or its **80-second process cap**. Evaluate checkpoint availability at 30/45/60/75s. A normal measurement cap is planned censoring of training, not an infrastructure crash, and only fully saved earlier checkpoints are eligible. A halted worker must not leave active descendants. If the trajectory ends early, its final eligible checkpoint remains the delivered model at later budgets. Do not reinterpret this as a learner independently annealed to each wall-time budget. Wall-based schedules are a different follow-up and must not be mixed into this study.

Validate new optional timing behavior on isolated canaries before enabling it; normal native training must preserve its old behavior when timing mode is off. Keep one GPU, synchronous rollouts and self-play disabled for this initial measurement mode. Benchmark overhead and effective checkpoint cadence for all four families before freezing the exact implementation.

**Evaluation allocation:** use exactly **1,024 preassigned evaluation episode IDs**, 16 per each of 64 slots, with recorded deterministic environment/policy RNG and recurrent-reset semantics. Fast slots cannot contribute extra sampled games. Log win/loss/draw/invalid action and decisions per assigned episode. Completed episodes receive their actual terminal outcomes. A missing record is an error, not a loss.

Connect4 differs from Pong: from the empty board, each nonterminal call places both players' pieces, so at most 21 agent decisions should finish a headless match. The native environment harness now checks per-episode decision counts, two-piece progress on nonterminal moves, reset boards/rewards and exact terminal log counts. These checks passed against the original game and all ten pixel representations, with 4,096 seeded parity transitions each, repeats and ASan/UBSan ([receipt](results/connect4cnn/claim-design-20260915/environment-tests.txt)). This supports the game-bound invariant; it does not implement or validate evaluator slot quotas or GPU recurrent resets. An over-bound match is a correctness failure, not an invented draw/truncation score. Fixed allocation still matters even though Connect4 does not have Pong's long-rally problem. Tests must additionally cover exact quotas, no duplicate IDs, action RNG/reset behavior, inactive-slot isolation, terminal boundaries and repeatable saved-model results. The existing pooled evaluator remains v1; v2 results cannot replace its records.

Use 512 assigned games for the development challenge (eight per slot), and 1,024 for confirmation. The matched evaluation seed is shared across models/recipes within each training-seed block. Training-seed replication remains the unit of primary inference, not the individual games.

## Gate 2: competitive and reviewable references

Audit exact architecture adaptations, image coverage, parameter counts, initialization, projection/core and kernel correctness. Ours and Nature share the same convolution primitive path, but layer shapes differ; inspect native timings and implementation choices rather than assume identical efficiency. For IMPALA/Impoola, use the received profiling evidence and resolve obvious implementation defects before freezing the measured backend. Do not attribute avoidable backend cost to architectural inferiority.

Document any retained performance limitations and restrict the claim to the native implementations actually measured. If backend improvements are made, validate independent float32 forward/all-gradient tests, pooling boundaries, deterministic eager/graph behavior and checkpoint/config correctness. Rebuild and apply the same source freeze to all final families; never optimize only our path after seeing final results.

## Gate 3: equal-opportunity learner challenge

Previous Nature discovery varied only training budget; it was not a comparable LR/gamma search. Before final confirmation, cross every fixed family with the same six recipes:

- Learning rate: .0003, .001, .003.
- Gamma: .8, .95.
- All other settings remain the captured `ocean/connect4cnn/compare.ini`, including replay ratio 1. The old common recipe is included.
- Training development seeds **62001–62003**, evaluation seeds **72001–72003**. Check local/remote receipts for prior use before freezing operational seed identity. Development outcomes never enter final inference.
- Four families x six recipes x three seeds = **72 attempts**. Each has the same 80-second training-process cap: at most **1,440s training allocation per family**, 5,760s total. Evaluation/builds are separate. Record actual consumed resources; equal ceilings do not imply identical realized cost.
- Select one learner recipe per family by the mean 60-second win rate across all three development seeds; break exact ties by LR ascending, then gamma ascending. All six recipes and failures remain visible. A recipe with missing/invalid planned results is ineligible; if a family has no eligible recipe, stop and repair the experiment rather than omit that family or score failures as zero.

This is a limited shared tuning allowance, not best-possible tuning or architecture-family optimization. Also report the matched-recipe development comparisons to expose recipe interaction. Do not expand the recipe grid after inspecting confirmation, and do not call per-family selected recipes a same-learner causal comparison.

## Final confirmation and analysis

After all gates pass, save exact source/config/binary/evaluation-protocol hashes and the four selected recipes. The design is **40 fresh paired training seeds, 63001–63040**, paired evaluation seeds **73001–73040**, four fixed configurations, **160 training jobs**. Collect the four declared checkpoint evaluations per job: **640 evaluations**, 1,024 assigned games each. If operational seed collisions are found, choose a new unused contiguous block and record it before any final output; never substitute a seed after seeing its result.

Primary outcome for each seed: quality's evaluated win fraction from its latest checkpoint available by 60s. Compare against each frozen reference's checkpoint at 30, 45 and 60s: **nine paired contrasts**. This prevents an earlier, better reference checkpoint from being ignored merely because that reference later declined. The 75s points and other quality checkpoints describe the broader observed curve; they are secondary, not additional success opportunities.

Use 100,000 whole-paired-training-seed bootstrap resamples, RNG seed **937163**, preserving all models and budget points within each sampled block. For each contrast, report the one-sided percentile lower bound at **alpha=.05/9**, with fixed linear quantile interpolation, plus effect size and all raw seed differences. Bonferroni controls multiplicity to the extent that the individual approximate bootstrap bounds achieve their coverage; these are not exact finite-sample guarantees. Report the limitations, instability/degeneracy and training failures explicitly. Do not use ordinary unadjusted pointwise bars as proof of frontier-wide dominance.

Predeclared success requires **all nine lower bounds >0**, with **all nine mean improvements >=5 percentage points**. The latter is a practical point-estimate threshold; it does not establish that each true improvement exceeds five points. A claim of at least five points would require lower bounds above .05, which this primary rule does not test. Missing planned outcomes prevent a complete-panel superiority conclusion; retain failures and any identification bounds rather than using only available seeds.

Fixed sample size: 40. No significance-driven additions, seed replacements, optional stopping or post-result changes to target time/model/recipe. If results are inconclusive or unfavorable, say so. New hypotheses require a new development/final split rather than recycling these final outcomes as untouched evidence.

[Power sensitivity](results/connect4cnn/claim-design-20260915/power_sensitivity.csv) uses a normal approximation to size the design, not a guarantee for the bootstrap analysis. Under a 10pp true difference and 15pp paired SD, 40 seeds gives roughly 95% per-contrast power at the corrected threshold; with a 5pp difference or higher variance, power is much lower. Joint success over nine contrasts can be lower than each contrast's power. Five historical seeds do not establish the assumed variance, so this is a transparent resource/effect-size compromise, not a promise of a positive result.

## Runtime, reproducibility and current status

Training-process ceilings are **96 minutes for the learner challenge** and **3h33m20s for confirmation**; these are maxima from the stated per-process cap, not measured ETAs. Builds, evaluation and audit add time. Measure canary throughput/evaluation duration and available disk before scheduling. Dense checkpoint files remain ignored and local; retain selected checkpoints and source/config/command/ready-time/hash receipts for independent review.

Use balanced predetermined family order within seed blocks on an otherwise idle GPU. Capture competing load without agent polling. No long job is launched by this specification. Current completed work is the historical selection/power analysis and its CPU safeguards. Native wall-budget checkpoint publication, exact-allocation evaluation, baseline review and the development challenge are **open execution gates**. Freeze an immutable launch manifest only after they pass; do not label this design as an already executed or fully validated protocol.

Reproduce development-only calculations:

```bash
.venv/bin/python -B research/tests/test_connect4_design.py
.venv/bin/python research/design_connect4_confirmation.py --output research/results/connect4cnn/claim-design-20260915
```

Keep Pong's findings and losses in the paper. Connect4 appearance tests, more games, expanded reference families and a search-method comparison remain necessary for broader robust-tool/generalization claims. The focused test above is the next bounded claim, not completion of the whole paper.
