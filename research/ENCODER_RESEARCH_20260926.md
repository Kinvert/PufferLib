# September 26 research refresh: useful next encoder experiments

Primary papers acquired with pinned versions, PDF hashes and Markdown conversions. See `research/papers.json`, `research/sources/<slug>/metadata.json` and `research/papers/<slug>.md`. Converted figures/equations remain leads; inspect the PDF and official code for reproduction. This note proposes experiments, not measured improvements in PufferLib.

## Shallow architectures beyond Nature

[Hadamax, 2025 v2](https://arxiv.org/abs/2505.15345v2) keeps a shallow encoder and combines parallel activated feature branches by elementwise multiplication, with max-pooling and GELU. Section 4 and Appendix B give the construction. Its PQN setup includes LayerNorm; its results do not directly predict behavior in our recurrent PPO-style learner. A native multiplicative block is a promising later experiment, but it needs two branches, extra weights and backward storage: shallow does not imply half the work. Encoder 5 now supports GELU/pooling exploration but does **not** implement Hadamax or its normalization.

[Aftab, 2026 v3 (September 24)](https://arxiv.org/abs/2608.07335v3) first compares eight visual topologies, then adds multiplicative/pooling processing, then compares complete value-estimation configurations. Architecture selection uses Atari-57, with later Procgen Hard evaluation. For us the useful comparison is its topology study and independent encoder ablations, rather than importing a full value head and attributing all improvement to the CNN. It is relevant newer related work when setting the eventual paper's baselines; the claimed final agent is not an encoder-only wall-time result under our protocol.

Read before the next structural extension: native operation/normalization code and width rules in the papers' linked official repositories; actual two-branch FLOPs, temporary memory and launch costs at our actor/training batches. A multiplication option should be a declared block type with numerical/reload tests, not arbitrary graph wiring in core PufferLib. Keep an ordinary single-branch control at comparable training/search budgets.

## Activations that learn their shape

[Adaptive rational activations, ICLR 2024 / arXiv v5](https://arxiv.org/abs/2102.09407v5) motivates learnable polynomial ratios and joint sharing in Atari RL. [Constrained rationals, 2026](https://proceedings.mlr.press/v330/surdej26a.html) studies robustness constraints in continuous control/continual learning. These support two separate investigations: activation flexibility, and whether sharing/constraints improve stability. Our encoder-5 implementation has independent operation coefficients with a positive denominator, not those complete methods. A rational coefficient curve can adapt beyond a fixed ReLU/SiLU/GELU shape, but expensive powers/reductions and instability may lose the wall-time frontier. First pass the GPU derivative/finite checks, then compare activations with identical architecture/learner/budget; do not confound a new activation with a new width.

## Future gated-block experiment: SwiGLU

Requested by Kinvert September 26: **add SwiGLU to things to try later. Not implemented.** The existing SiLU choice is an ordinary activation; SwiGLU forms two learned feature branches and combines them as `SiLU(a) * b`. See [Shazeer, GLU Variants Improve Transformer (2020), section 2](https://arxiv.org/html/2002.05202v1#S2). That paper tests Transformers; benefit for our pixel-RL CNN remains an experimental question.

Investigate a native sweepable gating option per convolution stage or projection, keeping the output width explicit. Two branches could use one packed convolution/projection producing twice the channels, followed by a CUDA gate kernel; verify this against an independent reference before optimizing. Account for both branches' parameters, FLOPs, saved activations and backward work.

Compare ordinary SiLU and ReLU blocks with SwiGLU under the same observations, learner/core, decision budgets and paired seeds. Include narrower gated variants to test whether the extra expressivity pays for its cost. Measure the whole score-versus-wall-time frontier, SPS and VRAM. Require GPU forward/gradient, repeatability and checkpoint/reload checks before a learning sweep. GEGLU can be a later related control once the shared gating path is validated.

## Game-image pretraining

[OCAtari v2](https://arxiv.org/abs/2306.08649v2) and its [official repository](https://github.com/k4ntz/OC_Atari) provide a relevant object-extraction starting point for Atari. We should audit pixel alignment and coverage game by game. [SGI v1](https://arxiv.org/abs/2106.04799v1) provides a distinct unlabeled temporal pretraining route. Neither proves that object supervision is best for our downstream learner. The native design can compare spatial object targets with temporal objectives later, while both feed images to the CNN. The exact proposed frame/label contract and leakage controls are in [PIXEL_PRETRAINING_DATASET_PLAN.md](PIXEL_PRETRAINING_DATASET_PLAN.md). No game-image dataset has been made.

## Practical next sequence

1. Validate the expanded encoder math on an available exclusive GPU, including regression of older shared kernels. Verify seeded train/reload and finite optimizer updates for every activation.
2. Start bounded discovery with only a few controls at fixed depths and matched decision budgets. Inspect coverage/duplicates and whole curves before adding controls. The existing GP does not become sample-efficient just because its search space is larger.
3. Profile actor and learner separately. Kernel fusion or an implicit convolution path should follow measured patch-buffer/launch/GEMM costs; smaller theoretical FLOPs are not a throughput guarantee.
4. Add one meaningful block family, such as controlled multiplicative branches or the SwiGLU experiment above, only after correctness and cost are understood. Treat normalization and backbone pretraining as independently logged axes.
5. Design the native pixel-label collector and fixtures; generate a bounded pilot only when authorized. Preserve scratch controls and pretraining compute in subsequent comparisons.

No timed GPU experiment, long search, dataset collection or SOTA claim is authorized by this research note. Existing evaluation/claim gates in [PAPER_PLAN.md](PAPER_PLAN.md) remain open.
