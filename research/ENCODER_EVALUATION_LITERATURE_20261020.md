# Does the encoder preserve what the game needs?

Research checked October 20, 2026. These are original reading notes, not full
paper transcriptions or reproduced experiments. The proposed PufferLib work is
specified in [ENCODER_INFORMATION_EVALUATION.md](ENCODER_INFORMATION_EVALUATION.md).
Published results below concern their own algorithms, inputs and hardware.
An initial [native Forage pilot](FORAGE_REPRESENTATION_PILOT.md) now implements
object/mark metrics, identity probes and a bounded 5090 runner; see its separate
receipts for actual build/host/GPU execution scope.

## Finding

Average image reconstruction error is not a measure of game understanding.
There are two distinct problems: large empty regions can hide the loss of a tiny
object, and complicated irrelevant scenery can consume scarce encoder capacity.
The papers below address these problems through representation probes,
task-relevant reconstruction, temporal objectives and distraction tests. No one
metric establishes that a compressed representation is sufficient for control.

## Read first: the closest matches to our problem

### 1. AtariARI / ST-DIM: measure whether the features retain game objects

Anand, Racah, Ozair, Bengio, Côté and Hjelm, *Unsupervised State Representation
Learning in Atari*, NeurIPS 2019.
[Paper, sections 3, 5.3 and 7](https://arxiv.org/html/1906.08226);
[authors' benchmark/code](https://github.com/mila-iqia/atari-representation-learning).

AtariARI evaluates frozen representations using linear classifiers for annotated
game variables. ST-DIM learns spatial and temporal features. The paper explicitly
discusses small objects receiving little reconstruction penalty and easy features
such as clocks satisfying an objective without preserving other objects. Its
controls include random encoders and a majority classifier.

**Our interpretation:** build pixel-coordinate/object-identity probes, including
the ball separately from the paddles and HUD. Freeze the actual exported latent;
train only the diagnostic head. Good probe results indicate accessible information,
not guaranteed successful control. RAM labels can include unobservable variables;
our single-frame targets must instead be visibly identifiable. This precedent is
much closer to our question than an image-quality leaderboard.

### 2. Segmentation Dreamer: supervise reconstruction where relevant objects are

Kim and collaborators, *Make the Pertinent Salient: Task-Relevant Reconstruction
for Visual Control with Distractions*, 2024 preprint / RLJ 2025.
[Preprint, sections 4 and 5.1.3](https://arxiv.org/html/2410.09972);
[authors' project](https://indylab.github.io/SD/).
The preprint and journal/project versions have different author lists; cite the
specific version used rather than merging their metadata.

The method uses task-relevance masks for auxiliary reconstruction targets while
retaining original image inputs. It studies simulator masks and estimated masks,
including incorrect masks, in distracting visual-control settings. Mask knowledge
is an assumption, not automatically discovered relevance.

**Our interpretation:** our native rasterizer could provide cheap evaluation masks
and, separately, privileged training targets. Keep full images at inference; do
not hand the policy object boxes or oracle crops. Masked-target reconstruction
and our proposed separately normalized per-object loss are different objectives.
The paper does not establish our objective, a tiny native CNN speedup, or general
Atari transfer. Incorrect masks can erase important evidence.

### 3. DBC: behavioral relevance can replace pixel fidelity

Zhang, McAllister, Calandra, Gal and Levine, *Learning Invariant Representations
for Reinforcement Learning without Reconstruction*, ICLR 2021.
[Paper, sections 4 and 6](https://arxiv.org/html/2006.10742);
[authors' implementation](https://github.com/facebookresearch/deep_bisim4control).

Deep Bisimulation for Control uses rewards and transition behavior to learn
representations that disregard irrelevant visual variation. Experiments include
background distractions and driving imagery.

**Our interpretation:** the right question is whether differences that change
future decisions survive compression. A representation optimized for one reward
can discard distinctions another game needs. Sparse rewards, partial observability
and approximate learned metrics prevent treating this as a ready-made universal
encoder objective. Include this as a later objective ablation, not an assumed fix.

## Alternatives and useful cautions

### 4. DreamerPro: reconstruction is not mandatory

Deng, Jang and Ahn, *DreamerPro: Reconstruction-Free Model-Based Reinforcement
Learning with Prototypical Representations*, ICML 2022.
[Proceedings and paper](https://proceedings.mlr.press/v162/deng22a.html).

Replaces image reconstruction with prototypical representation learning involving
recurrent world-model states, testing distracting and standard DeepMind Control
tasks. This is an alternative to spending capacity on all visible details.

**Our interpretation:** if reconstruction consistently loses important cues, test
a different objective rather than only increasing a pixel-loss coefficient.
DreamerPro changes a world model; its results do not transfer directly to an
encoder-only replacement in PufferLib.

### 5. ATC: test frozen features in actual downstream learning

Stooke, Lee, Abbeel and Laskin, *Decoupling Representation Learning from
Reinforcement Learning*, ICML 2021.
[Paper, sections 4.3–4.5](https://arxiv.org/html/2009.08319v3).

Augmented Temporal Contrast associates nearby observations and evaluates frozen
pretrained encoders through RL. Its multi-game Atari experiments show mixed
transfer; adding capacity partly helps, and held-out games remain challenging.

**Our interpretation:** reserve whole games, not just new seeds. A good latent
similarity score is not a substitute for training a new controller on the exported
encoder. Temporal similarity can also ignore small independent moving objects;
use object-specific diagnostics rather than assuming temporal training solves it.

### 6. SPR: preserve action-conditioned temporal information

Schwarzer and collaborators, *Data-Efficient Reinforcement Learning with
Self-Predictive Representations*, ICLR 2021.
[Paper](https://arxiv.org/abs/2007.05929);
[authors' code](https://github.com/mila-iqia/spr).

SPR predicts future latent representations using a learned transition model and
moving-average encoder targets, alongside RL and augmentation.

**Our interpretation:** a later pretraining objective can use actions and frame
sequences without reproducing every pixel. It needs additional machinery and
collapse controls. The current collector's isolated frame shards are not yet an
action-aligned sequence dataset; record timing, actions and terminal boundaries
before claiming support. Atari sample efficiency is not our measured wall time.

### 7. A 2025 large-scale study: inspect the mechanism, not just the headline

Luo, Ni, Bacon, Precup and Si, *Understanding Behavioral Metric Learning:
A Large-Scale Study on Distracting Reinforcement Learning Environments*, 2025.
[Paper, sections 4.3 and 5.2–5.4](https://arxiv.org/html/2506.00563).

The study evaluates behavioral-metric methods across noise settings and introduces
a denoising-factor diagnostic. Its isolated experiments find marginal standalone
benefit from explicit metric loss; other components such as self-prediction and
normalization can explain gains.

**Our interpretation:** separately test information preservation, nuisance rejection
and RL. Our proposed paired-render ranking test is a simpler project-specific
diagnostic, not a reproduction of their denoising factor. Never reward invariance
alone: a constant encoder is invariant to everything and useful for nothing.

### 8. Distracting Control Suite: vary appearance without changing the task

Stone, Ramirez, Konolige and Jonschkowski, *The Distracting Control Suite—A
Challenging Benchmark for Reinforcement Learning from Pixels*, 2021.
[Paper](https://arxiv.org/abs/2101.02722).

The benchmark varies backgrounds, colors and camera pose, including combinations,
to study visual-control robustness.

**Our interpretation:** our paired appearance worlds need a similar separation
between nuisance changes and task changes. A horizontal flip changes image
coordinates; it is not an appropriate unqualified latent-invariance target.
Changing an apple's dangerous marking changes meaning, not nuisance.

### 9. LPIPS: helpful image diagnostic, insufficient control test

Zhang, Isola, Efros, Shechtman and Wang, *The Unreasonable Effectiveness of Deep
Features as a Perceptual Metric*, CVPR 2018.
[Paper and proceedings](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html).

The work evaluates deep-feature distances against human image-similarity judgments,
motivating LPIPS instead of relying exclusively on pixel error or PSNR/SSIM.

**Our interpretation:** LPIPS is optional supplementary evidence. Human-perceived
image similarity is not sensitivity to a lethal two-pixel marking or precise ball
location. Its own network and spatial averaging can have blind spots. Do not make
LPIPS the sole selection score or introduce a heavy runtime perceptual network.

### 10. OCAtari: an annotation route for later real Atari evaluation

Delfosse and collaborators, *OCAtari: Object-Centric Atari 2600 Reinforcement
Learning Environments*, 2023/2024.
[Paper](https://arxiv.org/abs/2306.08649);
[authors' source](https://github.com/k4ntz/OC_Atari).

Provides Atari object extraction and interfaces useful for object-centric research.

**Our interpretation:** extracted object state can help build evaluation labels,
but must not become the pixel policy's observation. Check visibility, coordinate
alignment, occlusion and supported-object coverage per game. This is an annotation
candidate, not proof that our present native environment buffers have these labels.

## Decision for this project

First implement diagnostics: per-object reconstruction, frozen-latent location and
identity probes, paired appearance tests, then downstream score/time curves.
Only then choose whether object-balanced reconstruction, supervised pixel-space
targets or temporal training deserves the next pretraining allocation. An MSE
decrease currently proves neither useful perception nor an advantage over Nature.
