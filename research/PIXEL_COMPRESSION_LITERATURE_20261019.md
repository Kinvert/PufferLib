# Reading notes: fast visual representations for control

Research checked October19,2026. These are original summaries and project-specific
interpretations, not full-paper conversions or reproduced results. Primary sources
are linked per entry. Read alongside the [current-code audit](ARCHITECTURE_COMPRESSION_AUDIT_20261019.md)
and [full-frame backbone design](PIXEL_COMPRESSION_AND_BACKBONE_DESIGN.md).

The question is not whether a low-dimensional latent exists. It is whether we can
compute one cheaply from **real rendered game frames**, preserve small visual
cues and motion evidence, and train useful policies faster after counting every
stage. Hardware, algorithm and data differences limit transfer of published numbers.

## Most directly relevant papers

### SEER — stored embeddings,2021

[Chen et al., Improving Computational Efficiency in Visual Reinforcement Learning
via Stored Embeddings](https://arxiv.org/html/2103.02886v2), especially§4 and
the augmentation discussion. Freeze early convolutional layers and store their
outputs in replay, bypassing repeated convolution work during updates. The paper
studies off-policy control/Atari, not our native recurrent PPO-style learner.
It also stores multiple augmented embeddings when needed; a single cached vector
cannot reproduce arbitrary future pixel augmentations.

**Use here:** separate frozen-prefix inference from the trainable tail and record
latent provenance. Most useful precedent for the proposed runtime architecture.
Do not copy its speedup into our results: replay reuse, model shape, hardware and
freeze schedule determine the benefit. Read before implementing a latent buffer.

### ATC — decoupled temporal features,2021

[Stooke et al., Decoupling Representation Learning from Reinforcement Learning](https://arxiv.org/html/2009.08319v3),
§4.3–4.4. Augmented Temporal Contrast learns from nearby video observations and
tests frozen pretrained encoders, including multi-task settings. The important
negative result: shared/held-out-game Atari transfer is weaker than single-game
pretraining; increasing capacity partly helps shared training. Strong results on
some tasks do not establish a universal frozen visual backbone.

**Use here:** a temporal objective and a serious warning to reserve entire games,
not just new seeds, when testing transfer. Compare frozen and adaptable prefixes.
Our changing gaze also changes observation coordinates, so temporal pairs need
camera/action metadata in training targets, with no future information at inference.

### SGI — Atari pretraining,2021

[Schwarzer et al., Pretraining Representations for Data-Efficient Reinforcement
Learning](https://arxiv.org/html/2106.04799v1). Combines latent-dynamics and
goal-conditioned representation learning from unlabeled experience, followed by
task adaptation. It motivates game-aligned pretraining rather than assuming
ImageNet classification is the correct objective.

**Use here:** temporal/action-conditioned supervision as an alternative to box
labels. Sample-efficiency evidence does not establish native end-to-end speed,
nor that a tiny frozen encoder matches its larger fine-tuned configurations.
Keep pretraining data, downstream learner and model capacity explicit.

### SAC-AE — reconstruction can help,2019/2020

[Yarats et al., Improving Sample Efficiency in Model-Free Reinforcement Learning
from Images](https://arxiv.org/abs/1910.01741). Investigates instability when
combining visual autoencoding with off-policy RL and develops a stable
deterministic-autoencoder approach. This is not evidence that any VAE pretrained
on game screenshots will produce a sufficient frozen controller input.

**Use here:** reconstruction is a useful comparison objective. Evaluate whether
its latent preserves rare dangerous details instead of selecting by average
pixel loss. A decoder used only for pretraining need not burden deployment.

### DBC — control relevance without reconstructing pixels,2020/2021

[Zhang et al., Learning Invariant Representations for Reinforcement Learning
without Reconstruction](https://arxiv.org/abs/2006.10742). Uses behavioral
similarity through rewards and transitions to encourage task-relevant features,
including distractor robustness in visual control.

**Use here:** discard background variation while retaining differences that
change decisions. Caution: a representation sufficient for one reward function
may merge distinctions needed by another. Reward-conditioned compression is not
automatically a general game backbone, and full-state theoretical assumptions
do not directly settle our partially observed setting.

## Recent evidence worth incorporating, with limitations

### APE — adaptive visual pretraining,2025

[Zhang et al., Efficient Reinforcement Learning Through Adaptively Pretrained
Visual Encoder](https://arxiv.org/html/2502.05555v1), methodology and AppendixB.
Uses adaptive augmentation during ImageNet-100 pretraining and examines
downstream visual RL, principally with a ResNet18-based encoder. The appendix
explicitly discusses weaker behavior on multi-object Atari observations and
domain shift from natural-image classification. Some variants freeze early
layers and update a later layer; “pretrained” should not be paraphrased as
universally frozen.

**Use here:** an ImageNet route is worth comparing, but game-specific images and
multi-object spatial targets deserve priority. Augmentations that remove color
or fine markings are unsafe defaults for our discriminative sprites. This paper
does not certify our PPO learner or tiny-CNN wall-clock improvement.

### CoRP — compressed but detailed representations,October2026 preprint

[Chen et al., Compact Robot Policies Need Fine-Grained Visual Representations](https://arxiv.org/html/2610.08183v1),
§3–5. Uses DINOv2-initialized visual features, task adaptation and learned-query
compression for manipulation. Its ablations report that freezing and variational
compression can hurt, despite a useful hard token budget.

**Use here:** reducing token count and preserving fine-grained information are
compatible goals; freezing is an experiment, not a rule. This is a48.9M-parameter
demonstration-trained system with language/proprioception and closed-set evaluation,
not a small pure-pixel PufferLib CNN. Do not borrow its headline size or success
comparisons as our expected speed/generalization. Inspect its training seed and
evaluation scope before using it as a methodological template.

### DINOv3 — dense features and distillation,2025

[Siméoni et al., DINOv3](https://arxiv.org/html/2508.10104v1), especially distillation
and dense-feature evaluation. A large visual teacher can transfer representations
to smaller architectures, including ConvNeXt students. This is an available
teacher direction, not a reason to deploy a huge backbone inside every RL step.

**Use here:** compare spatial teacher targets against inexpensive game labels.
First establish that features resolve small action-relevant objects and that a
tiny student can retain them. Teacher inference/pretraining, model access terms
and export compatibility remain real costs. No weights downloaded in this audit.

### V-JEPA2 and2.1 — temporal latent prediction,2025/2026

[V-JEPA2](https://arxiv.org/abs/2506.09985) learns from video and adapts latent
prediction to action-conditioned planning. [V-JEPA2.1](https://arxiv.org/abs/2603.14482)
emphasizes spatially dense and temporally grounded representations through its
training objectives. These large-data/model systems motivate prediction in
feature space rather than reconstructing every visual detail.

**Use here:** borrow the question—what temporal and spatial information should a
compact feature preserve?—before borrowing the model. A reduced native student
needs its own collapse prevention, small-object probes and downstream tests.
Neither source establishes a fast CNN-only replacement in PufferLib.

### VC-1 / CortexBench — generalization audit precedent,2023

[Majumdar et al., Where are we in the search for an Artificial Visual Cortex for
Embodied Intelligence?](https://arxiv.org/abs/2303.18240). Broad evaluation of
pretrained visual representations finds no universally dominant frozen encoder;
domain/task adaptation matters. More diverse pretraining helps on average but
does not improve every task.

**Use here:** report per-game regressions and adaptation costs. An average gain
across related sprite games is weaker evidence than transfer to distinct unseen
visual domains. Do not equate a large natural-video pretraining budget with a
free initialization.

## Compression and small-network building blocks

| Source | Useful mechanism | What must still be demonstrated here |
|---|---|---|
|[MAE,2021/2022](https://arxiv.org/abs/2111.06377)|Learn features by reconstructing masked image content|Masking small critical objects can teach the wrong invariances; a lightweight decoder at pretraining says nothing about encoder runtime|
|[VQ-VAE,2017](https://arxiv.org/abs/1711.00937)|Discrete learned codebook|Quantization saves stored bits, not necessarily encoding time; rare-cue loss and code collapse need tests|
|[RoB,2023](https://arxiv.org/abs/2301.09451)|Distill a large self-supervised teacher into small students|Natural-image transfer results are not game-control results; spatial detail and tiny-student capacity matter|
|[Anti-aliased CNNs,2019](https://arxiv.org/abs/1904.11486)|Filter before subsampling to improve shift stability|Filtering can remove discriminative spots; verify control-information retention|
|[SPD-Conv,2022](https://arxiv.org/abs/2208.03641)|Space-to-depth before convolution to retain small-scale input information|Rearrangement alone is not compression; added channels can erase arithmetic savings|
|[MobileNetV2,2018](https://arxiv.org/abs/1801.04381)|Inverted residuals and narrow linear bottlenecks|Depthwise low-FLOP operators can have poor GPU utilization at our batch/width; full backward and optimizer cost count|
|[FasterNet,2023](https://arxiv.org/abs/2303.03667)|Partial convolution and hardware-aware throughput design|Measured classifier inference speed does not transfer to recurrent RL training; compare ordinary dense alternatives|
|[EfficientViT,2023](https://arxiv.org/abs/2305.07027)|Reduce memory-heavy attention/reshaping work|A token stage after a costly stem cannot recover stem time; tiny inputs may favor dense CNNs|
|[GLU variants,2020](https://arxiv.org/abs/2002.05202)|SwiGLU multiplicative branch, SiLU(a)×b|Not a scalar SiLU switch; extra projections/activations need narrower matched-cost controls|

The table is an annotated reading queue, not evidence that all these architectures
should enter one sweep. Introduce one qualified block family at a time.

## Labels, world models and misleading search matches

[OCAtari](https://arxiv.org/abs/2306.08649) and its
[official interface](https://oc-atari.readthedocs.io/en/latest/ocatari/core.html)
can help investigate pixel-aligned object annotations. The image and object
observation modes are distinct. Feeding RAM-derived objects at deployment would
change a pixel benchmark into state/object learning. Validate visible extent,
occlusion and game-specific completeness before treating extracted boxes as labels.

[DreamerV3's published world-model description](https://doi.org/10.1038/s41586-025-08744-2)
combines representation/dynamics/reward learning and imagined control. It is
evidence that compressed visual models can support learning; importing that whole
algorithm is a different project from replacing PufferLib's encoder.
[EfficientZeroV2](https://proceedings.mlr.press/v235/wang24at.html) similarly targets
sample-efficient control through model-based methods, not our encoder-only cost.

Searches also return video-language token compression such as
[MARC](https://arxiv.org/abs/2510.07915), and using RL to optimize a video codec,
such as [RL-RC-DoT](https://openaccess.thecvf.com/content/CVPR2025/html/Gadot_RL-RC-DoT_A_Block-level_RL_agent_for_Task-Aware_Video_Compression_CVPR_2025_paper.html).
These optimize different consumers and objectives. They do not establish that
skipping game frames or storing smaller files speeds native pixel-control training.

## Reading order and resulting decisions

1. SEER and ATC: architecture of frozen features/reuse, and negative transfer.
2. SGI, SAC-AE and DBC: what the representation should learn and what it can lose.
3. APE and CoRP: newer evidence for adaptation, data mismatch and retained detail.
4. Anti-aliasing, MobileNetV2/FasterNet, then RoB: practical small-stem candidates.
5. DINOv3/V-JEPA and VC-1: teacher opportunities and the limits of broad transfer.

Most defensible initial recommendation: train a small spatial encoder on diverse
game frames with visible-object/detail supervision or temporal features, retain
a reconstruction control, and compare scratch/frozen/fine-tuned versions. A
natural-image teacher is optional. Make the full-frame inference path fast and
measure total learning cost; neither a small latent nor a high probe score is
sufficient evidence on its own. This is our synthesis, not a theorem or an
established best recipe from any one paper.
