# Reading list

Read the first five entries first. The other papers address recurrence, learning stability, hardware efficiency, and credible evaluation. Inclusion is not endorsement of a paper's headline result as current state of the art.

## Core encoder question

| Paper and local conversion | Primary source | Why read it |
|---|---|---|
| [Nature DQN — Mnih et al., 2015](papers/nature-dqn.md) | [Nature](https://www.nature.com/articles/nature14236) | Establish the actual three-convolution baseline, preprocessing, and dense projection. Do not substitute the earlier 2013 DQN architecture. |
| [IMPALA — Espeholt et al., 2018](papers/impala.md) | [arXiv](https://arxiv.org/abs/1802.01561) | Read the deep residual encoder specification separately from the distributed IMPALA algorithm and V-trace. |
| [Impoola — Trumpp et al., 2025](papers/impoola.md) | [arXiv](https://arxiv.org/abs/2503.05546) | Direct encoder-design evidence: replacing flattening with global average pooling changes parameter use and Procgen generalization. Inspect per-game results and architecture appendix. |
| [BBF — Schwarzer et al., 2023](papers/bbf.md) | [arXiv](https://arxiv.org/abs/2305.19452) | Read §4 and the width ablations. This supports effective scaling with appropriate training changes; it does not establish that smaller encoders always win. |
| [RL scaling — Hilton, Tang, Schulman, 2023](papers/rl-scaling.md) | [arXiv](https://arxiv.org/abs/2301.13442) | Methodology for compute-optimal model size and intrinsic performance. Useful experimental framing, not a universal CNN/recurrent allocation rule. |

## Recurrence, optimization, and alternative ways to save compute

| Paper and local conversion | Primary source | Why read it |
|---|---|---|
| [DreamerV3 — Hafner et al.](papers/dreamerv3.md) | [arXiv](https://arxiv.org/abs/2301.04104) | Read the network and model-size appendices. Encoder, recurrent model, and other components scale together. Its learned world model has different costs/objectives from PufferLib's policy core. |
| [Were RNNs All We Needed? — Feng et al.](papers/mingru.md) | [arXiv](https://arxiv.org/abs/2410.01201) | Background for minimal recurrent architectures. Compare its equations with PufferLib's actual gated/highway implementation rather than assuming identity. |
| [IL scaling — Tuyls et al.](papers/il-scaling.md) | [arXiv](https://arxiv.org/abs/2307.09423) | Atari CNN/linear scaling and NetHack sequence modeling. These are imitation-learning results, with different data generation and optimization from online RL. |
| [Phasic Policy Gradient — Cobbe et al.](papers/ppg.md) | [arXiv](https://arxiv.org/abs/2009.04416) | Investigates representation learning and policy/value optimization phases; useful context for shared encoder training, not merely another CNN design. |
| [Primacy bias — Nikishin et al.](papers/primacy-bias.md) | [arXiv](https://arxiv.org/abs/2205.07802) | Helps explain how early data can constrain later learning and why reset interventions matter. Keep this distinct from every possible form of plasticity loss. |
| [DrQ-v2 — Yarats et al.](papers/drqv2.md) | [arXiv](https://arxiv.org/abs/2107.09645) | A useful visual continuous-control efficiency reference. Architecture results must be interpreted alongside augmentation and algorithm changes. |
| [SEER — Chen et al.](papers/seer.md) | [arXiv](https://arxiv.org/abs/2103.02886) | Freezes encoder layers and stores embeddings in replay. Direct prior work if “static encoder” means frozen weights rather than a fixed architecture. |
| [On the Role of Computation in RL — Ghugare et al., 2026](papers/computation-rl-2026.md) | [arXiv](https://arxiv.org/abs/2602.05999) | Separates parameter count from computation using repeated computation in a policy. Relevant to reasoning compute, but recurrence within a decision differs from memory across observations. |
| [Observational Overfitting — Song et al.](papers/observational-overfitting.md) | [arXiv](https://arxiv.org/abs/1912.02975) | Studies observation-space changes while keeping underlying dynamics fixed; useful for testing whether visual representation is the limiting factor. |

## Hardware and systems comparisons

| Paper and local conversion | Primary source | Why read it |
|---|---|---|
| [MobileNetV2 — Sandler et al.](papers/mobilenetv2.md) | [arXiv](https://arxiv.org/abs/1801.04381) | Candidate building blocks: depthwise convolutions, inverted residuals, and linear bottlenecks. Image classification results do not establish RL learning performance or GPU training speed. |
| [ShuffleNet V2 — Ma et al.](papers/shufflenetv2.md) | [arXiv](https://arxiv.org/abs/1807.11164) | Explains why FLOPs alone are an inadequate speed metric: memory access, operator fragmentation, and platform behavior matter. |
| [Cleanba — Huang et al.](papers/cleanba.md) | [arXiv](https://arxiv.org/abs/2310.00036) | Reproducible distributed actor/learner design and strong systems baselines. Especially relevant to PufferLib's asynchronous rollout/learner configuration. |
| [EnvPool — Weng et al.](papers/envpool.md) | [arXiv](https://arxiv.org/abs/2206.10558) | Compiled environment execution and throughput measurement. Distinguish simulator speed from complete learning throughput. |
| [PlayTrain — Truong et al., September 2026](papers/playtrain.md) | [arXiv](https://arxiv.org/abs/2609.09059) | Recent systems comparator that reports Nature/IMPALA encoder throughput. Its Table 1 uses four H100s and 92 CPU cores; generated game clones are not identical ALE/Procgen tasks. |

## Benchmarks and statistics

| Paper and local conversion | Primary source | Why read it |
|---|---|---|
| [Procgen — Cobbe et al.](papers/procgen.md) | [arXiv](https://arxiv.org/abs/1912.01588) | Procedurally generated pixel tasks and held-out levels for evaluating generalization. |
| [Statistical Precipice / rliable — Agarwal et al.](papers/rliable.md) | [arXiv](https://arxiv.org/abs/2108.13264) | Aggregate performance measures and uncertainty estimates. Use this before selecting a winner from noisy multi-task runs. |
| [POPGym Arcade](papers/popgym-arcade.md) | [arXiv](https://arxiv.org/abs/2503.01450) | Pixel-based recurrent-policy evaluation, including observability differences relevant to encoder/core allocation. |
| [Memory Gym](papers/memory-gym.md) | [arXiv](https://arxiv.org/abs/2309.17207) | Memory-focused visual tasks; inspect observation and memory protocols before choosing configurations. |
| [Revisiting the Arcade Learning Environment](papers/ale-protocols.md) | [arXiv](https://arxiv.org/abs/1709.06009) | Atari evaluation practices, stochasticity, and reproducibility. Essential context for comparable ALE claims. |
| [POPGym](papers/popgym.md) | [arXiv](https://arxiv.org/abs/2303.01859) | Broader memory benchmark context. Original vector-observation tasks alone cannot establish pixel encoder quality. |

## How to read efficiently

For an encoder paper, extract the layer table, preprocessing, policy/value sharing, core type, optimizer, and training budget before interpreting scores. For an efficiency paper, extract exact hardware, batch shapes, precision, rollout/update counts, and what the timer includes. For a benchmark paper, extract observation semantics, action repeat, train/test separation, and evaluation stochasticity. Record unresolved details rather than silently choosing favorable settings.

Every converted paper carries its source URL; exact local versions and byte hashes are in `sources/<slug>/metadata.json`. Conversions are search aids. Use PDFs when reading figures, tables, equations, and appendices.
