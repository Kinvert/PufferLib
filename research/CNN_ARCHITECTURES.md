# CNN encoders for PufferLib: architecture and research notes

Research date: 2026-09-11. This is an engineering research proposal, not a report of completed training experiments. Original conversation and pasted Claude response remain preserved in the repository root; corrections below should guide future work.

## 1. The question we can actually test

Find the encoder, projection, and recurrent-core combination that reaches a specified return with the least elapsed training time on specified hardware, while satisfying a declared reproducibility contract. Also measure return at fixed environment interactions and fixed estimated training FLOPs. These are separate objectives and can choose different winners.

Joseph's question has at least four independent variables:

| Component | Example choices | What changing it tests |
|---|---|---|
| Spatial encoder | Nature, IMPALA, fewer residual blocks, changed downsampling | Visual representation and spatial computation |
| Feature readout / projection | Flatten, global pooling, small spatial grid; output dimension `D` | Information retained for the core and projection expense |
| Temporal core | None, existing PufferLib minGRU, conventional GRU/LSTM; hidden size `H` | Memory capacity, temporal computation, implementation efficiency |
| Training system | Batch sizes, sequence length, update ratio, numerical mode | How architecture costs translate into learning time |

“Static encoder” needs a precise definition. A fixed architecture can still learn every update. Frozen weights are a different experiment. Caching outputs of an encoder whose weights keep changing also changes training semantics. Do not silently treat these as equivalent.

The strongest initial result would be a reproducible Pareto frontier and a small set of useful defaults. A universal encoder/core ratio or SOTA performance is an outcome to establish, not an assumption.

## 2. Nature DQN CNN, precisely

The 2015 Nature DQN uses stacked grayscale observations, three convolutions with ReLU, a 512-unit hidden fully connected layer, and action-value outputs. It is a feedforward agent with short observation history supplied through stacking. The original encoder input is `4 × 84 × 84`. See the Methods in [Nature DQN](https://www.nature.com/articles/nature14236), [local Markdown](papers/nature-dqn.md), and the independent implementation in [OpenAI Baselines](https://github.com/openai/baselines/blob/master/baselines/common/models.py#L13).

The following counts are our own arithmetic for valid convolutions, bias enabled, batch size one. They exclude the final action head. Shapes below are `channels × height × width`.

| Layer | Kernel / stride / padding | Output | Parameters incl. bias | Forward MACs, weights only |
|---|---|---|---:|---:|
| Input | — | `4 × 84 × 84` | 0 | 0 |
| Conv1 + ReLU | `8 / 4 / 0` | `32 × 20 × 20` | 8,224 | 3,276,800 |
| Conv2 + ReLU | `4 / 2 / 0` | `64 × 9 × 9` | 32,832 | 2,654,208 |
| Conv3 + ReLU | `3 / 1 / 0` | `64 × 7 × 7` | 36,928 | 1,806,336 |
| Flatten, linear + ReLU | `3136 → 512` | `512` | 1,606,144 | 1,605,632 |
| **Encoder total** | | | **1,684,128** | **9,342,976** |

An `A`-action value head adds `513A` parameters and `512A` MACs. A shared actor/critic head has a different count. Nature's final projection contains about 95% of encoder parameters but about 17% of its MACs. A parameter bottleneck and a convolution bottleneck are therefore different things.

For a Procgen-shaped `3 × 64 × 64` input, the convolution outputs are `32 × 15 × 15`, `64 × 6 × 6`, `64 × 4 × 4`. A `1024 → 256` projection yields **338,336 parameters and 3,414,016 MACs**. Call this `nature_64_d256`: it is an adapted Nature encoder, not the original complete DQN architecture.

The final Nature convolution has a theoretical receptive-field width of 36 pixels, derived recursively as `r_next = r + (kernel−1) × input_jump`, with output jump 8 pixels. Flattening then lets the projection combine locations across the complete map. Removing that projection or replacing it with pooling changes what spatial relationships can be represented cheaply.

## 3. IMPALA CNN, precisely

IMPALA names an actor/learner algorithm as well as a family of networks. Its deep encoder has three stages with channels `[16, 32, 32]`; each stage has one convolution, max pooling, and two residual blocks containing two convolutions each: `3 × (1 + 2 × 2) = 15` convolutions. The paper used DMLab and Atari; Procgen came later. Its Atari experiments used frame stacking without an LSTM. See [IMPALA, Figure 3 and §5.3](https://arxiv.org/pdf/1802.01561), [local Markdown](papers/impala.md).

The original [DeepMind implementation, `_torso`](https://github.com/google-deepmind/scalable_agent/blob/master/experiment.py#L129) specifies this stage:

```text
x = Conv3x3(x, output_channels=C, stride=1, SAME)
x = MaxPool3x3(x, stride=2, SAME)
repeat twice:
    skip = x
    x = Conv3x3(ReLU(x), C, stride=1, SAME)
    x = Conv3x3(ReLU(x), C, stride=1, SAME)
    x = x + skip
```

After all three stages: ReLU, flatten, linear to 256, ReLU. There is no BatchNorm in this torso. ReLU placement matters: adding a post-add ReLU to each residual block creates a variant.

For `3 × 64 × 64` and a 256-dimensional projection, our derived counts are:

| Stage | Output after pool and residual blocks | Parameters incl. bias | Forward MACs, weights only |
|---|---|---:|---:|
| Stage 1, C=16 | `16 × 32 × 32` | 9,728 | 11,206,656 |
| Stage 2, C=32 | `32 × 16 × 16` | 41,632 | 14,155,776 |
| Stage 3, C=32 | `32 × 8 × 8` | 46,240 | 4,718,592 |
| Flatten + projection | `2048 → 256` | 524,544 | 524,288 |
| **Encoder total** | | **622,144** | **30,605,312** |

For `4 × 84 × 84`, spatial sizes are `42 → 21 → 11`, and the same family with `D=256` has **1,089,232 parameters and 54,222,848 MACs**. The input contract must accompany every quoted cost.

### Padding is a porting trap

TensorFlow `SAME` pooling is not always identical to symmetric padding of one pixel. At even input sizes, a 3-wide, stride-2 `SAME` pool needs one total padding element and places it on the trailing edge. Symmetric `padding=1` puts one element on both edges and can produce the same output shape but a different alignment. On odd sizes they can agree. A CUDA port must reproduce the selected reference's windows, not merely its shapes. Max-pool padding also needs the correct sentinel value; zero is not equivalent to negative infinity for negative inputs.

Publish `original_same` versus `symmetric_pad1` as explicit variants if both are useful. Save tiny hand-checkable tensors around borders in correctness fixtures when implementation begins.

## 4. Impoola and the spatial readout question

Impoola replaces flattened spatial features with global average pooling before the projection. Its experiments report improved Procgen generalization; its interpretation emphasizes translation sensitivity. This makes it a required baseline, but does not establish a universal pixel-RL winner or a CUDA speedup. The paper's tables include policy/value parameters and use their own multi-add accounting, so their totals should not be mixed with our encoder-only, weight-MAC convention. See [Impoola v2, §4–5 and Appendix D](https://arxiv.org/html/2503.05546v2), [local Markdown](papers/impoola.md), [author code](https://github.com/raphajaner/impoola).

Keeping the preceding `64 × 64` IMPALA convolution stack fixed gives this derived comparison:

| Readout | Projection input | Projection parameters | Total encoder parameters | Total conv + linear MACs |
|---|---:|---:|---:|---:|
| Flatten | 2,048 | 524,544 | 622,144 | 30,605,312 |
| Global average pool | 32 | 8,448 | 106,048 | 30,089,216 |
| Adaptive average pool to `2 × 2` | 128 | 33,024 | 130,624 | 30,113,792 |
| Adaptive average pool to `4 × 4` | 512 | 131,328 | 228,928 | 30,212,096 |

Pooling arithmetic is excluded from the MAC column and must be timed. GAP saves roughly **83% of parameters but only 1.7% of conv + linear MACs** in this example. Memory, optimizer-state traffic, generalization, and core-input size can still make it useful. It is not evidence that GAP alone makes the whole encoder several times faster.

Impoola already evaluates `AvgPool(2,2)`, reporting a tradeoff between retained spatial information and generalization across agent-centered and non-agent-centered games. See [v2 §5.3 and Figure 7](https://arxiv.org/html/2503.05546v2#S5.F7). Our proposed flatten/GAP/`2 × 2` comparison is a controlled PufferLib replication and extension, with `D` held fixed; it is not a new pooling idea. The `4 × 4` row above is our cost calculation, not a reported Impoola result. Then vary `D` independently to distinguish readout choice from core-input capacity.

Reasoning from information flow: a core cannot reconstruct a cue from a history of embeddings if the encoder maps all relevant histories to indistinguishable embeddings. A larger recurrent state does not automatically compensate for spatial information discarded before it arrives. Conversely, an encoder need not retain every pixel to supply sufficient information for good control.

Test location-sensitive tasks and small-object motion explicitly. Global pooling, early downsampling, color removal, and frame-stack removal impose different information constraints. Preserve a full-input baseline for each.

## 5. Correcting the pasted research response

| Earlier claim or implication | What should guide this project |
|---|---|
| BBF shows that smaller IMPALA encoders beat wider IMPALA encoders and encoder scaling is harmful | BBF's performance improves with width, and it selects 4× width after observing comparable 4× and 8× performance. The comparison of a “smaller” ResNet concerns IMPALA versus the other CNN at corresponding widths. The unsuccessful scaling of SR-SPR does not establish that scaling BBF's encoder is harmful. [BBF §4/Figure 3](https://arxiv.org/html/2305.19452v3), [local](papers/bbf.md). |
| Dreamer demonstrates a fixed encoder while scaling only recurrence | Its model-size presets scale CNN channels and recurrent units together, along with other components. The RSSM also has stochastic representations and world-model training objectives; it is not simply a PPO LSTM. [DreamerV3](https://arxiv.org/abs/2301.04104), [local model-size Table 3](papers/dreamerv3.md). |
| Procgen was the original IMPALA benchmark | IMPALA predates Procgen; see §3 above. |
| A horizon-scaling result establishes an encoder-versus-memory allocation rule | Hilton et al. study model-size/compute scaling and a toy horizon manipulation. Reward horizon and perceptual memory demand are distinct variables. Their work provides useful methodology, not a measured encoder/core ratio. [Scaling laws for single-agent RL](https://arxiv.org/abs/2301.13442), [local](papers/rl-scaling.md). |
| A 20M-step/s Breakout sweep supplies a relevant CNN/LSTM ratio | The pasted conversation has no verified run/config source for that claim. This checkout's [CUDA Breakout](../ocean/breakout/breakout.cu) defines `OBS_SIZE=118` and emits state features; [its configuration](../config/breakout.ini) is not evidence for an Atari pixel CNN. |
| Nobody has studied this, so the project is necessarily novel | The reviewed papers do not settle our exact hardware/algorithm/task question. That is a bounded literature finding. It does not prove universal novelty, and it must be revisited before writing a paper. |

“Primacy bias” concerns excessive influence of early training experience. It is not a synonym for every scaling failure or all loss of plasticity. Resets, normalization, and optimizer settings are potential experimental factors, not universal fixes to turn on without controls. [Primacy bias paper](https://arxiv.org/abs/2205.07802), [local](papers/primacy-bias.md).

A useful newer connection is [On the Role of Computation in Reinforcement Learning](https://arxiv.org/abs/2602.05999), [local](papers/computation-rl-2026.md). It separates policy compute from parameter count and studies additional repeated computation. This broadens the question beyond width, but is not an empirical encoder/recurrent-core allocation law for our setting.

## 6. Accounting rules for every architecture

For dense convolution with groups `G`, dilation `d`, stride `s`, and symmetric padding `p`:

```text
Hout = floor((Hin + 2p − d(Kh−1) − 1) / s) + 1
parameters = Cout × (Cin/G × Kh × Kw + bias_enabled)
MACs = Hout × Wout × Cout × Cin/G × Kh × Kw
linear MACs = input_features × output_features
linear parameters = (input_features + bias_enabled) × output_features
```

Count one multiply-accumulate as one MAC and two FLOPs. State clearly whether bias additions, pooling, activation, residual addition, normalization, optimizer updates, and preprocessing are included. Our tables exclude them from MACs and include trainable biases in parameters. Width scaling is approximately quadratic in same-width interior convolutions, linear in the first convolution for fixed input channels, and linear in a projection whose output width stays fixed.

For a conventional single-layer LSTM with feature dimension `D` and hidden width `H`, matrix work per step is approximately `4H(D+H)` MACs; a conventional GRU uses approximately `3H(D+H)`. Bias conventions vary. These formulas **do not describe the repository's minGRU**, which must be counted from [its implementation](../src/algo.cu). Do not attach LSTM cost estimates to PufferLib's actual core.

For PPO-like repeated updates, an approximate run budget is:

```text
training FLOPs = rollout_forward_count × rollout_forward_FLOPs
              + learner_sample_count × learner_forward_backward_FLOPs
              + auxiliary / target / optimizer work
```

Count learner samples with repetition; use effective replay/update ratio, not just environment steps. A “backward is 2× forward” estimate is a rough accounting shortcut, not a measured law. Input gradients may be unnecessary at the first layer; frozen layers, recurrent scans, and fused operations also change the ratio.

Measure encoder, projection, recurrent core, heads, preprocessing, and environment work separately, then measure the complete system. Profile rollout batch sizes and learner batch/sequence shapes separately. Parameter count is not GPU time. [ShuffleNet V2](https://arxiv.org/abs/1807.11164), [local](papers/shufflenetv2.md), explicitly motivates measuring speed on the target platform beyond FLOPs.

## 7. Candidate experiments, in a useful order

These are proposed PufferLib experiments, including replications of existing literature findings. Preserve an immutable reference configuration for each published architecture.

1. **Establish exact references.** Nature at original shape and at the benchmark's native shape; IMPALA at base width; Impoola at the same width; existing PufferLib core and a no-recurrence control. Fix preprocessing and projection dimension for family comparisons. Also retain canonical dimensions for reproduction runs.
2. **Reduce high-resolution work.** Try one residual block per IMPALA stage; narrower first/second stages; and a stride-2 learned downsampling variant. In the derived baseline, stages 1–2 account for about 84% of convolution MACs. Moving work to lower resolution is a plausible source of substantial savings, with a risk of losing small objects and alignment detail.
3. **Compare spatial readouts.** Replicate the flatten, GAP, and `2 × 2` pooling comparison from Impoola in PufferLib, extending it to our core and timing measurements. Evaluate held-out levels and tasks needing position. This is a small, interpretable first implementation.
4. **Separate three capacities.** Sweep encoder stage widths, projection `D`, and core `H` independently. Begin with a small coarse grid and retain the frontier. Do not immediately spend a full budget on every Cartesian combination.
5. **Test economical convolutions.** Compare a small dense convolution against depthwise plus pointwise convolution, optionally a narrow bottleneck residual block. Fewer MACs may still lose on GPU because of memory movement and small kernels. Keep only variants that improve measured batch latency or memory before paying for long RL runs. [MobileNetV2](https://arxiv.org/abs/1801.04381), [local](papers/mobilenetv2.md), provides an architectural reference, not evidence of an RL win.
6. **Explore frozen lower layers separately.** [SEER](https://arxiv.org/abs/2103.02886), [local](papers/seer.md), freezes early encoder layers and stores embeddings for off-policy replay. A corresponding PufferLib experiment would need an explicit freeze schedule, cache validity rule, and a control with fresh encoder evaluation. This changes the learning procedure, so separate it from pure architecture comparisons.
7. **Only then choose CUDA specialization targets.** Rank kernels by contribution to complete training time and reuse across the surviving architectures. A fast Nature port and a fast IMPALA port are useful baselines even if the final architecture differs. Keep backend and architecture comparisons distinct.

For memory studies, keep frame stacking, hidden-state resets, and truncated backpropagation length explicit. Use a memory manipulation that changes access to information, not just a longer episode. Add a parameter-matched feedforward core if the question is whether recurrence helps beyond additional nonlinear capacity.

Do not call an entire Atari suite “reactive” or assume a state-based multiagent environment is a pixel benchmark. Document the policy's actual observation tensors.

## 8. What would constitute a useful contribution

A convincing artifact could be a small encoder family that reaches fixed return targets faster across several pixel tasks, together with reproducible settings and a clean PufferLib integration. A second contribution could be an automated search procedure showing how the best spatial/core allocation moves with budget and memory demand. Either needs actual learning curves and seed variability, not just a favorable microbenchmark.

Before choosing a default, require all of:

- Correct forward, gradients, state resets, and short training behavior against a reference implementation.
- Return versus interactions, return versus elapsed time, and return versus estimated total training FLOPs.
- Multiple seeds and confidence intervals; failed runs and missed return targets retained in the record.
- Explicit numerical and determinism settings. “Same seed” alone is not a determinism contract; bitwise repeatability on one pinned stack is different from numerical agreement across backends or machines.
- A full observation/preprocessing/config manifest, actual source revision, and enough timing metadata to reproduce the claim.

No architecture experiments or CUDA modifications were performed to produce this note. The arithmetic tables were independently evaluated from the displayed formulas; they are operation estimates, not hardware measurements.
