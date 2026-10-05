# Potential priorities

The deliverable is a clean native PufferLib pixel-learning integration, backed by
reproducible evidence suitable for a paper. Beating Nature on a measured part of
the frontier is a hypothesis; it does not by itself establish SOTA.

Ordered priorities, recorded October 2, 2026:

1. **Automate the 5090 validation gate.** Build a bounded runner for short
   training, checkpoint reload, finite-weight checks and identical-seed
   repeatability. Preserve commands, configs, logs and source/build receipts.
   Passing a build or a configuration canary is not passing this gate.
2. **Strengthen CNN mathematical verification — current work.** Check encoder 5
   against an independent float64 reference for forward values and every
   parameter gradient: dilated convolutions, residuals, SAME pooling, adaptive
   readout and all five activation choices. Include boundaries/ties and unused
   coefficient slots, sampled finite differences and eager/graph/actor checks.
   Compile locally; execute the CUDA comparisons on the 5090. A CPU reference
   is an oracle for GPU results, not a substitute for GPU validation. Preserve
   failures and fixtures; do not tune tolerances to hide failures.
3. **Make failed sweeps and reporting reliable.** Audit interrupted trials,
   numerical failures, resource limits and sidecar/upload failures. Retain their
   configurations, status and consumed time; do not silently drop them or mark
   an incomplete campaign successful.
4. **Estimate candidate memory before allocation.** Add a native estimate of
   parameters, optimizer state, activations and workspaces for the actual batch
   configuration. Reject oversized candidates with an explicit reason before
   allocating; compare estimates with GPU measurements before relying on them.
5. **Finish paper-grade evaluation and timing.** Implement and validate exact
   episode allocation and checkpoint completion/timing receipts. Predeclare
   matched learner settings, seed allocation, full-frontier analysis and
   stopping rules. Account for uncertainty, failures, baseline efficiency and
   tuning/pretraining compute. See `research/CONNECT4_CLAIM_PROTOCOL.md` and
   `research/PAPER_PLAN.md`; neither canary success nor a best seed is a claim.
6. **Expand architecture options after current operators pass.** Investigate
   grouped/depthwise convolutions, bottlenecks and eventually SwiGLU. Add small
   native building blocks with explicit sweep bounds, correctness checks and
   measured overhead. Keep pretrained initialization a separate experimental
   axis with random-initialization controls.

## Constraints and next handoff

October 5 audit follow-ups: [the draw-rule correction](research/CONNECT4_DRAW_CORRECTION.md)
fixes the inherited environment defect and adds independent semantic fixtures.
Still open: retain completed checkpoint cells when training later fails (the
current parser rejects the whole failed process), and narrow the exact-evaluator
worker-count check's claim because its environment stepping is serial. Keep
these distinct from the unchanged statistical and GPU acceptance gates.

October 2 progress: item 2 is implemented/compiled and awaits 5090 execution.
Item 5 now has [native measurement and audit groundwork](research/CLAIM_PIPELINE.md),
including matched preparation and full curves. All four builds and local artifact
checks pass. GPU acceptance is pending; preliminary synthetic interval coverage
is insufficient, so the inference/confirmation gate remains open. No GPU runs
are currently requested. The other priorities remain outstanding.

- No encoder-5 GPU validation or sweeps on G240's 5060; use the separate 5090.
- No CPU neural-model testing/training in place of GPU checks, no system CUDA
  changes, and no dataset generation under the current design-only permission.
- Start the remote handoff at `START_HERE_5090.md` and
  `NEXT_5090_FLEX2_SWEEP.md`. Historical campaigns must not be relaunched.
- Keep production training/construction/search in C/CUDA. NumPy verification
  and external reporting are test/research tools, not the delivered trainer.
