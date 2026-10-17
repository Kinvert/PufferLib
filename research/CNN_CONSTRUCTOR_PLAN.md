# Potential deliverable: native PufferLib CNN constructor and search

October9 current direction: read [the realistic-image requirements](GENERALIZATION_PRIORITIES.md#current-operating-requirements-realistic-images-and-nature-range-curves).
The constructor must support varied practical game/camera image sizes and RGB,
not force every environment through36x44 grayscale. Human-readable640x480-class
native frames and explicitly declared policy preprocessing are separate controls;
build geometry from game state at the new resolution instead of enlarging the old
raster. Portable input metadata, storage/memory preflight and baseline adaptation
are near-term work. Target Nature-range training with roughly8 checkpoints,
first strides1/2/4/8, fixed unswept seeds and selective profiling. Existing native
paths remain specialized and no new GPU campaign is launched by this plan.

**Goal:** deliver a clean, usable PufferLib tool that finds a useful CNN performance-versus-training-time frontier for each pixel environment and search budget. A winning fixed CNN remains a possible deliverable; this is an additional route, not a replacement for the existing research or an accepted upstream feature.

September 15 sequencing: follow the [ordered milestones in the main paper plan](PAPER_PLAN.md#next-actions--ordered-implementation-milestones). The immediate [Connect4 protocol](CONNECT4_CLAIM_PROTOCOL.md) measures the full observed frontier with identical learner settings, decision budget and checkpoint cadence for all four fixed encoders. Validate timing/exact episode allocation, baseline fairness and simultaneous analysis, then confirm on fresh training seeds. Per-model hyperparameter selection and architecture search are separate later studies. Pong v2 remains necessary before further Pong quality claims; baseline attribution and portable image support remain delivery work. The [5090 audit](PONG_EVALUATION_AUDIT_RESULTS.md) found sparse match completion and substantial recipe dependence. The tool must expose reliable tradeoffs even when our current compact model loses; more architecture trials cannot repair a biased/incomplete evaluation protocol.

## User workflow

1. Select an environment, search budget, and allowed architecture building blocks through normal numeric INI settings.
2. Start with small, inexpensive architectures to establish an initial frontier.
3. Let native PROTEIN explore alternatives and more expensive features where they might improve that frontier.
4. Confirm selected configurations across fresh seeds and export ordinary training INIs, with performance/time estimates and uncertainty.

The output is a small menu of verified tradeoffs: low training cost, a chosen performance target, or higher performance at greater cost. Describe these as the **best configurations found within the search budget**, not guaranteed global optima.

## First implementation

- Build on the existing native configurable CNN and PROTEIN integration. Expose depth, channels, kernels, strides, pooling, residual connections and projection width; add dilation as a separately validated extension. Keep architecture construction, optimization and training in C/CUDA, with minimal changes to `src/`. External W&B/reporting can remain optional.
- Initially hold the recurrent core, environment representation and learner recipe fixed while searching architecture and training budget. Give finalists a separately declared hyperparameter-tuning allowance; joint architecture/learner search can follow once comparisons are dependable.
- Treat any single fixed learner recipe as one controlled slice, not evidence of a generally inferior architecture. Pong's recipe-dependent ranking makes a small shared recipe panel or declared comparable finalist tuning necessary for broader claims. Separate architecture-search and learner-tuning resources and account for native effective replay updates.
- Prefer small candidates initially, but retain broader exploration. Increasing complexity is optional, not a mandatory growth ladder. A larger model may learn enough faster to cost less overall.
- Construct each trial's network before training and start from fresh weights. Growing a live network or transferring weights between architectures is outside the first version.
- Validate legal shapes before launch, identify equivalent active architectures, and avoid repeating identical configuration/budget/seed trials. Record full configs, native timing, SPS, decisions, evaluation, failures and source/checkpoint hashes.

## Staged development

1. **Reliable constructor:** clean numeric configuration and exported INIs; use Connect4/Pong to validate the complete build/train/eval path and pixel contract.
2. **Budget-aware search:** establish a small-first sampling policy, then compare it with unrestricted search. Calibrate screening budgets so slow-learning candidates are not discarded simply for failing to learn in a tiny run. Predefine any promotion, stopping and learning-rate schedule rules; short-run checkpoints are not automatically equivalent to independently trained short-budget models.
3. **Verified frontiers:** freeze finalists and evaluate fresh seeds with matched budget schedules and appropriate uncertainty. Export useful configurations and a reproducible report.
4. **PufferLib-ready delivery:** extend beyond the current fixed grayscale input, demonstrate fresh-checkout use on an established pixel benchmark, replace temporary Python preparation where needed, and prepare clean upstream-reviewable code.

## Evidence for a paper

Primary question: **under a declared search resource budget, does the tool find better useful time/performance tradeoffs than competitive reference architectures and simpler search?** Compare comparably tuned Nature/IMPALA/Impoola families, random architecture search, and small-first versus unrestricted search. Use multiple tasks, full curves, failed runs and fresh-seed confirmation under the [paper plan's evidence standard](PAPER_PLAN.md#required-evidence-standard-defend-the-claim-like-a-thesis).

Report search cost separately from the cost of training an exported model, including whether repeated use amortizes the search. Per-environment architecture selection and transfer of one frozen architecture across environments are different claims. Useful engineering delivery does not by itself establish research novelty or SOTA; those require the corresponding comparisons and related-work review.

**Current status:** proposal only. Existing native construction/search and reporting provide a starting point; automatic small-first expansion, reliable candidate promotion, broad pixel portability and upstream acceptance are not yet established.
