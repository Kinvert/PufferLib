# RTX 5060 comparisons

October 5, 2026: Kinvert authorized a fresh local GPU comparison while the
separate 5090 is busy. This lifts the local hold for **four existing fixed
encoders only**: quality (encoder 4), Nature, IMPALA and Impoola. Encoder-5 tests
and sweeps remain reserved for the 5090; no 5090 job is requested.

## Current scheduled panel

- Corrected `connect4-full-board-draw-v2`, appearance 0, 1×36×44 grayscale.
- Same common learner, H128/L1, float32 and 13,312,000 decisions per model.
- One training seed, 173; evaluation seed 20173; 13 checkpoints per model,
  1,024 requested games each: 52 checkpoint observations total.
- Four short build/train/reload canaries must finish before full training.
- All training runs are serial on an idle 5060; keep other GPU jobs off it.
- Expected training from historical 5060 measurements: ours 2–3 minutes,
  Nature 2–3, IMPALA 20–22, Impoola 20–22. Budget approximately 50–65 minutes
  including canaries/builds/evaluations. These are estimates, not new timings.
- The existing runner caps each full training process at 2,400 seconds and each
  evaluation at 60 seconds. Failures are retained; a timeout is not a result.

Launch only when explicitly scheduled (this panel is scheduled):

```bash
bash ocean/connect4cnn/benchmark_local.sh 5060
```

The runner prints its unique `build/connect4cnn/hardware-5060.*` campaign and
`research/results/connect4cnn/rtx-5060/hardware-5060.*/` receipt directories.
It records status, logs and exact canary/full run paths. On complete success it
verifies all 52 checkpoint hashes/counts/finiteness, source and binary hashes,
the observation matrix and CSV/native evaluation agreement, then preserves the
small artifacts under `full/`. The full frontier is `full/analysis/curves.html`,
with chronological observations, means, missing cells and frontier flags in CSVs.
Full checkpoints/binaries stay in the original build directory. No automated
commit/push of new results occurs; review them first. Do not poll the long run.

## Interpretation and hardware separation

This is an **exploratory single-seed pilot**, using existing pooled-v1 evaluation
and approximate launch-to-checkpoint file times. Batched evaluation can exceed
its requested quota and favor faster-finishing episodes. Do not describe these
scores as the pending exact-episode protocol or as paper-grade dominance.
Whole-process training time/SPS and native adjusted uptime/SPS stay separate.

Keep these results separate from the [historical 5090 results](HARDWARE_5090_RESULTS.md)
and older [5060 confirmation](CONFIRMATION_RESULTS.md): those used the legacy
draw rule and earlier source. A later matched 5090 panel must run the same source,
recipe, rules, seeds and budgets; `benchmark_local.sh 5090` writes a separate
`rtx-5090/` tree when that machine is available and explicitly scheduled.
Whole-machine differences include CPU/toolchain effects, not just the GPU.

No new learning results were available when this scheduling note was written.
