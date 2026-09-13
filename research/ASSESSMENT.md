# Research assessment and starting direction

September 11, 2026. This synthesizes the companion reports and local source inspection. It proposes work; it does not report a new encoder, completed training sweep, or SOTA result.

## The project worth pursuing

Build a reproducible way to find small, fast visual policies for native PufferLib, then use it to select and optimize an encoder. The contribution can include both an efficient implementation and evidence about how to allocate compute between visual processing and recurrence. Keep those claims independently measurable.

Joseph's question is best expressed as: **given this task distribution, hardware, learner, and training budget, what combination of convolutional encoder, feature readout, and recurrent core reaches useful performance fastest?** FLOPs, elapsed time, and sample efficiency answer different parts of that question. A universal ratio is not a justified starting assumption.

The reviewed literature offers architectural baselines, component ablations, and compute-scaling methods. It does not give a directly transferable optimum for this native PufferLib/MinGRU setting. This bounded finding is a reason to experiment, not proof that nobody has investigated related allocation questions. [RL scaling](https://arxiv.org/abs/2301.13442), [Impoola](https://arxiv.org/abs/2503.05546), [2026 computation study](https://arxiv.org/abs/2602.05999).

## Corrections that change the plan

1. **BBF is evidence that encoder scaling can work.** Its successful method scales the residual encoder; the pasted response misinterpreted the smaller IMPALA-versus-Nature comparison as smaller-versus-wider IMPALA. Do not design a study that presupposes small encoders will win. [BBF](https://arxiv.org/abs/2305.19452).
2. **DreamerV3 scales more than its recurrent core.** Its model-size table also increases convolution channels. Its world-model training and imagination costs differ substantially from a policy encoder feeding MinGRU. [DreamerV3, model-size appendix](https://arxiv.org/pdf/2301.04104v2).
3. **Pooling primarily removes projection parameters in the small reference example.** The arithmetic in [CNN_ARCHITECTURES.md](CNN_ARCHITECTURES.md) shows an approximately 83% parameter reduction but only 1.7% reduction in conv/linear MACs for fixed-width 64-pixel IMPALA with a 256-unit readout. Those are derived counts, not measured speedups. Impoola already studies a 2×2 pooling alternative, so that experiment would be replication/extension. [Impoola](https://arxiv.org/abs/2503.05546).
4. **The local fast Breakout environment is state-based.** Its 118-value observation is not a pixel image or original ALE observation. The anecdotal 20M-step/s run has no verified configuration in the conversation and cannot supply a CNN/core ratio. [Local Breakout](../ocean/breakout/breakout.h), [configuration](../config/breakout.ini).
5. **The current core is native MinGRU.** The default encoder output is tied to `hidden_size`; independent channel width, intermediate feature dimension, and recurrent size need an explicit interface. [Architecture construction](../src/algo.cu).

“The CNN only encodes while the recurrent net reasons” is a useful motivation, not a strict separation of function. Spatial computation can happen in the encoder, and the core cannot recover distinguishing information that the encoder never passes along. Similarly, a fixed encoder architecture and frozen encoder weights are different proposals. Freezing and caching embeddings already has direct prior work. [SEER](https://arxiv.org/abs/2103.02886).

## First implementation milestone

Use the [CUDA/scaffolding design](CUDA_AND_SCAFFOLDING.md) to build one faithful Nature encoder, a numerical reference, and a replayed-observation timing harness. Add faithful IMPALA through the same configuration contract. Measure forward, gradients, one optimizer update, and realistic rollout/learner batch shapes before launching architecture search.

A lightweight reference in this directory's uv environment will allow quick temporary variants. A native implementation will establish integration and runtime behavior. Both should consume an immutable architecture specification, with separate identifiers for **architecture** and **backend**. Avoid repeatedly editing one shared encoder file and losing the identity of earlier experiments.

Expose convolution channels, readout style, feature dimension, recurrent width/depth, and an explicit feature-to-core adapter. Account for the adapter separately. Initially preserve the existing MinGRU core; introduce other cells only as separate controlled comparisons. A no-recurrence control is valuable, but it must also be implemented explicitly rather than assumed to be an existing supported setting.

The current NMMO3 encoder supplies an integration example using im2col and cuBLAS, not a ready-made RGB encoder. Its stale reference-test paths and the allocator/optimizer shape assumptions need attention when implementation begins. The CUDA report also identifies a stream/workspace binding question requiring verification before claiming deterministic library GEMMs. These are static inspection findings, not observed training failures.

## First learning experiments

Use original Procgen as the initial external pixel target: a small development panel first, then broader confirmation. Add paired fully/partially observable POPGym Arcade tasks to investigate memory demand directly. Preserve the environment's actual observation and timing semantics. Treat native pixelized games as useful integration tests with their own protocol. [Procgen](https://arxiv.org/abs/1912.01588), [POPGym Arcade](https://arxiv.org/abs/2503.01450).

Run a small deterministic grid before an adaptive architecture search. Fix preprocessing, learning settings, and seed lists; vary one small set of architectural factors. Then use the measured cost frontier to select experiments under a budget. The [sweep report](BENCHMARKS_AND_SWEEPS.md) provides staged budgets, promotion rules, confirmation, uncertainty estimates, and a result-record outline.

Two promising early axes are reducing expensive high-resolution convolution work and changing spatial readout. Their risks differ: early downsampling may erase small objects; pooling may erase useful position. Wider recurrence may help only when observations and backpropagation horizons support the required memory. These are hypotheses to test.

## What makes the result credible

Measure time to a predeclared return threshold, return at fixed interactions, return at fixed estimated compute, memory use, and repeated-run determinism. Preserve failed runs and missed thresholds. Charge the entire search budget, including discarded candidates and tuning. Use independent training seeds for confirmation; deterministic execution does not remove statistical uncertainty. [Statistical evaluation](https://arxiv.org/abs/2108.13264).

Compare with optimized systems that already use compiled components. Cleanba/EnvPool are relevant references. PlayTrain is a recent throughput comparator, but its Table 1 uses four H100s and 92 CPU cores and its generated game clones are not the original tasks. We must match hardware, task semantics, observation shape, and timing scope before calculating a speedup. [Cleanba](https://arxiv.org/abs/2310.00036), [EnvPool](https://arxiv.org/abs/2206.10558), [PlayTrain](https://arxiv.org/abs/2609.09059).

A useful first deliverable is a clean two-encoder comparison with reproducible measurements and a swappable implementation interface. That remains useful even if the best initial architecture is an existing one. It provides evidence for deciding where custom CUDA work or a larger allocation study is justified.
