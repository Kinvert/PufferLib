# Compact visual inputs and a reusable PufferLib backbone

October 20 implementation update: full-frame compression kernels and an optional
native reconstruction pretrainer now exist. Read
[current compression status](FULLFRAME_COMPRESSION_PROGRESS.md),
[autoencoder implementation](AUTOENCODER_PRETRAINING.md) and
[existing weights](PRETRAINED_BACKBONES_20261020.md). Qualification limits apply;
there is no downstream transfer/Pareto result yet. This remains the design record.
Start with the
[source audit](ARCHITECTURE_COMPRESSION_AUDIT_20261019.md) and
[paper notes](PIXEL_COMPRESSION_LITERATURE_20261019.md).

## What we want the representation to retain

**Kinvert's clarification during the audit:** the main deliverable must work on
real human-visible games. Start with full rendered frames at sensible resolutions
(roughly640x480, preserving native aspect ratio); do not make8x8 sprite sources
or known tile boundaries the premise. The packed/tile arithmetic below audits
our current implementation and supplies diagnostics, not the main deployment
contract. Full-frame architecture is specified next.

The target is enough information to choose good actions from image history,
at minimum total training cost. Reconstructing every background pixel is not
the objective. Nor is the smallest possible vector regardless of task success.

Keep object identity, small dangerous markings, position, obstacles, and the
evidence needed to infer motion. A recurrent core can retain past evidence;
it cannot recover a distinction that every observation has erased. A single
frame does not expose hidden velocity or intentions. Pixel-space labels may
supervise what is visibly identifiable, not hidden state masquerading as vision.

The human display can remain640x480 while the policy consumes a compact sensor
representation derived from that image. This follows Kinvert's October19 request
to investigate aggressive compression. It does not shrink the game grid, change
rules, remove difficult objects, or authorize an unreported observation advantage.
Keep full-resolution reference images and make reduced policy views inspectable.

## Quantitative reduction opportunities

For18 RGB tiles plus4 geometry values, current image dimensions are192H×96W.
The following is scalar/byte arithmetic, **not throughput prediction**. FP32
sizes include geometry; ratios use image scalars only.

| Per-tile resolution / representation | Packed H×W | Image scalars | FP32 observation bytes | Image-scalar reduction |
|---|---|---:|---:|---:|
|32x32 RGB current|192×96|55,296|221,200|1×|
|16x16 RGB|96×48|13,824|55,312|4×|
|8x8 RGB|48×24|3,456|13,840|16×|
|4x4 RGB diagnostic|24×12|864|3,472|64×|
|18 tiles ×32 learned features|ordered features|576|2,320|96×|
|18 tiles ×16 learned features|ordered features|288|1,168|192×|

At32agents×32steps, current FP32 observations alone occupy226,508,800bytes
(216.016MiB) per such buffer. Sixteen features/tile would occupy1,196,032bytes
(1.141MiB), about189.4× smaller including geometry. These exclude network
activations, gradients, optimizer state, buffer copies and allocation rounding.

Do not replace image area with feature area without counting channels:
`96×192×3 -> 12×24×64` is only3× fewer scalar values, despite64× fewer spatial
locations. Float32→uint8 is4× fewer bytes, not4× less CNN arithmetic. Space-to-depth
reduces spatial size but preserves scalar count; a subsequent learned projection
is what makes it lossy and smaller.

The existing tile network already produces16features/tile. That proves the
interface scale is plausible, not that a frozen16-feature encoder generalizes.

### A baseline shape trap

Original valid Nature kernels8/s4,4/s2,3/s1 require each input dimension≥36.
The16px-tile packed image has width48 and works;8px tiles have width24 and do
not. Do not silently label a changed SAME-padded Nature as the original baseline.
Options are a declared common zero-padding adapter for every family, an explicitly
named Nature-small adaptation, or keeping the direct Nature comparison at16px
while comparing more aggressive learned codecs as complete systems against its
best measured pipeline. Upscaling a tiny image back to the old tensor loses much
of the compute benefit. Each option needs a separate observation/model manifest.

## Route A: fixed pixel reduction before neural work

For sprites, generate smaller tile atlases once at initialization from the final
32x32 RGB tile bytes, including skin, texture and animation frame. Runtime writes
the reduced atlas into the policy buffer. The current renderer already caches
full tiles; use that same simple C pattern. Keep the display atlas separate.

Compare these filters, with fixed output alignment and edge rules:

| Filter | Why try it | Important failure mode |
|---|---|---|
|Area/box average|Cheap, deterministic; meanpool and avgpool are the same operation here|Dilutes a tiny spot/ball; may make lookalikes equal|
|Nearest|Very cheap; tests duplicated-pixel redundancy|Sampling phase can miss a whole marking|
|Low-pass then decimate|Reduces aliasing and shift sensitivity|Blur can remove the very detail the task needs|
|Max/min or mean+range channels|Retains extrema lost by averaging|Per-channel extrema can fabricate colors; extra channels cost bandwidth|
|Small multiscale representation|Cheap context plus detailed local cues|More inputs and operators; crop policy changes must be declared|

Use32→16 first, then8, and4 only as a stress diagnostic. Do not assume smaller
is better. [Anti-aliased CNNs](https://arxiv.org/abs/1904.11486) motivates controlled
filtering, not guaranteed preservation of reward-relevant high frequencies.

**Exactness scope:** precomputing each tile's box reduction equals reducing an
assembled tile-aligned image only when the filter bins and boundaries match.
Filters spanning adjacent tiles need halos or runtime handling. Independent ROI
views should not blur across their artificial packed seam. Overlap, transparency,
camera motion and animated sprites must be reduced after the correct composition,
or tested against that reference. Do not conflate replacing artwork with resizing
the original observation.

For arbitrary frames, reduction still runs each step. A fused native kernel can
resize and normalize, but GPU-side resize after uploading FP32 full frames does
not save H2D traffic. CPU-side reduction saves upload bytes at CPU cost. Measure
both critical paths. Environment-specific atlas precomputation is not available
for camera images and must be labeled as such.

### Test loss of meaning before full RL

Enumerate the **small existing** sprite/skin/texture catalog, not a new large
dataset. For each transform, compare:

- Exact output collisions between task-distinct sprites; nearest-pair distances.
- Safe/deadly lookalikes and color-swapped skins, not just average image error.
- Every sampling phase, boundary location, clipping and current texture.
- Small moving-object visibility across consecutive frames and terminal views.
- Native full-image reduction versus cached tile reduction, where equivalence is claimed.

No exact collision is only a necessary check: two barely different outputs can
still be difficult to learn. Conversely a collision between behaviorally equal
decorations can be harmless. Use task-distinct labels only to audit the codec,
never feed them into the policy. Later use held-out visual probes and actual RL
to assess learnability; do not choose a representation solely by reconstruction.

## Route B: a tiny pretrained visual encoder

Start with a small **full-frame** RGB encoder, not a large ImageNet model in every
environment step. It must process arbitrary pixel locations without object IDs,
tile-kind lookup, oracle crops or semantic grid boundaries. One concrete design
family to test is:

```text
actual rendered frame, e.g.480H x640W RGB
   -> cheap learned stem that sees fine details BEFORE discarding resolution
   -> staged spatial reduction, small channel counts
   -> spatial feature map or small set of position-bearing visual tokens
   -> small trainable adapter -> MinGRU -> policy/value heads
```

Illustrative output budgets from921,600 input scalars:

| Representation | Scalars | Scalar reduction |
|---|---:|---:|
|120H x160W x16 features|307,200|3x|
|60H x80W x16 features|76,800|12x|
|30H x40W x16 features|19,200|48x|
|15H x20W x32 features|9,600|96x|
|128 position-bearing tokens x32 features|4,096|225x|

These are candidate **internal** feature budgets, not render sizes or promised
lossless encodings. Count any positional/auxiliary channels separately. Learn
local distinctions before aggressive aggregation. Compare a dense strided stem,
a narrow bottleneck stem and a coarse-context/detail branch; the latter must
derive its detail selection from images or learned gaze, not simulator objects.
Prefer fixed spatial grids before adding a complex token router.

The first implementation's4x4average readout is provisional. Pooling can make
two small objects at different positions inside one bin look identical to the
controller, particularly when their local feature maps are translated copies.
A recurrent core may track position from history after a known start, but that
does not establish good single-frame localization or arbitrary-spawn transfer.
Before promoting this graph on quality, test within-bin translations and compare
a larger spatial grid or position-preserving feature moments. Keep any position
features derived from image coordinates rather than simulator object locations.
Fast compression that destroys required position is a negative result to retain.

Keep an efficient conventional-resize pipeline as a baseline. Feeding Nature
unmodified640x480 and its enormous flattened projection would make a weak speed
baseline, not a fair claim against practical Nature usage. All systems start from
the same full image; their declared internal preprocessing is part of system cost.
Also run matched-preprocessing encoder comparisons to isolate architecture effects.

The existing shared tile model remains a smaller diagnostic and a reusable
unit-test substrate. A controlled scratch/pretrained pair for it is:

```text
32x32 RGB tile (or declared reduced version)
   -> inexpensive learned spatial stem
   -> one small low-resolution mixing block
   -> 16 or32 features, kept in spatial order
18 visual feature vectors + existing4 geometry values
   -> trainable adapter -> existing MinGRU -> learned movement/gaze/value heads
```

Reuse the existing tile topology as the first scratch/pretrained pair; only then
change it. A second graph should accept arbitrary image patches without semantic
tile boundaries. A spatial feature grid or small2x2 readout is safer than assuming
global average pooling retains position. A multiscale stem can mix coarse context
with detail; keep its new spatial coverage matched across compared policies.

An encoder trained once is not automatically frozen. Test three distinct modes:

1. Scratch, end-to-end RL, existing control.
2. Pretrained initialization, end-to-end RL: possible sample-efficiency gain,
   but similar per-update visual cost.
3. Pretrained frozen prefix, small trainable adapter/core: potential compute gain,
   but possible negative transfer. A later partially fine-tuned tail is a fourth mode.

Also test a frozen-random-prefix diagnostic. It distinguishes the value of
learned pretraining from an architectural change or random-feature projection.
Do not market that diagnostic as a competitive tuned baseline.

### What to pretrain on

Use the [pixel-space annotation contract](PIXEL_PRETRAINING_DATASET_PLAN.md).
Collection remains future work, not part of this research audit.

- **Visual supervision:** identities and visible masks/centers in the actual
  frame, with hard pairs of similar sprites. For current aligned cells, a small
  classifier or mask head is simpler than a full detector. Boxes alone cannot
  distinguish identically shaped objects. Discard auxiliary heads for RL.
- **Temporal prediction/contrast:** useful for motion and reusable scene features.
  Record actions, timestamps, reset boundaries and gaze movements; otherwise
  camera/gaze displacement can be mistaken for object dynamics. Avoid positive
  pairs that force two meaningfully different scenes to be identical.
- **Teacher distillation:** a larger game-trained or pretrained visual teacher
  supplies spatial targets; the tiny student alone runs downstream. Verify the
  teacher distinguishes lookalikes first. Global classification embeddings can
  omit tiny objects. Large natural-image teachers are optional, not prerequisites.
- **Reconstruction baseline:** small deterministic autoencoder, decoder used only
  during pretraining. Compare plain versus foreground/detail-weighted errors;
  any label-derived weighting is supervised, not wholly self-supervised.

First compare objectives separately on one encoder graph. Only combine those
that add measurable downstream value. A useful proposed combined objective is
`L_visual + lambda_temporal*L_temporal + lambda_detail*L_detail`, with an explicit
dimension budget and measured runtime. Its weights must be selected on development
data, not confirmation tasks. Prevent trivial constant features in latent
prediction using a validated contrastive/teacher/variance mechanism; stop-gradient
alone is not a proof against collapse.

Do not begin with strong VAE KL pressure: it trades away information, including
rare cues. A discrete VQ code is an option if storage is genuinely limiting;
codebook lookup/quantization and training complexity are additional costs. A
256-dimensional feature is already small; quantizing it may save less than
removing repeated full encoder passes.

## Route C: actually reuse the representation in the learner

Current `src/pufferl.cu` stores observations in rollout buffers; `src/algo.cu`
re-executes the encoder during training. Freezing weights without changing those
paths can skip backward/update but still repeats forward. A distinct frozen
prefix path should produce features during collection and store them once:

```text
rollout: pixels -> frozen prefix -> stored z -> trainable tail/core -> actions
update:                    stored z -> trainable tail/core -> loss/backward
```

This is related to [SEER](https://arxiv.org/html/2103.02886v2), whose replay setting
differs from our learner. Our proposed integration must preserve sequence order,
terminal resets, behavior logprobs/values and recurrent starting states. Keep
the frozen prefix separate from parameter registration for the trainable optimizer;
zero gradients alone can still allow optimizer momentum/decay to mutate weights.
Bind every feature buffer to an encoder hash/version and preprocessing manifest.

If any upstream weights change, stored features become stale. Initially support
a fixed prefix for the entire run, with no hidden online fine-tuning. Later a
versioned freeze schedule must flush/re-encode affected data and account for it.
Pixel augmentations need recomputation or predeclared encoded variants; feature
augmentation is not automatically equivalent. Do not cache recurrent outputs:
they depend on trajectory history and resets, unlike independent image features.

Simplified cost model per collected observation, assuming R learner presentations:

```text
end-to-end visual work ~ F_actor + R*(F_learner + B_learner + U_visual)
frozen + stored z     ~ F_actor + feature-write/read
```

Here F/B/U are measured forward/backward/allocated optimizer costs, not a claimed
fixed ratio. With illustrative equal F and B=2F, R=1 gives4F→F in the visual
portion; atR=4 it gives13F→F. Environment, action transfer, recurrence, policy
learning and pretraining remain. These are conditional arithmetic, not4×/13×SPS.

Pretraining break-even: if pretraining costs P and downstream runs save
`T_scratch-T_pretrained>0`, amortization requires approximately
`N >= P/(T_scratch-T_pretrained)`. Example only: one hour of pretraining and
30seconds saved/run requires120runs before additional data/search costs. Publish
cold-start total cost and warm-start downstream cost separately.

## Two reuse opportunities that need honest claim boundaries

**Frozen atlas features:** for a finite sprite atlas, compute embeddings from
its rendered RGB bytes once, then gather the same vectors. This can approach
state-sized transport, but it exploits known grid boundaries and finite repeated
images. Never replace embeddings with object IDs or rewards. Demonstrate equality
to the frozen pixel encoder for every atlas entry and show uncached pixel inference
on held-out images. Report this specialized mode separately from general vision.

**Repeated patches with a trainable encoder:** within one weight version, identical
independent tiles can share a forward computation. Sum downstream gradients in
fixed order before a shared backward call. This requires exact bytes, no stochastic
per-instance transforms or batch-dependent normalization, and invalidation after
each update. It changes accumulation order unless carefully reproduced; qualify
numerics and repeatability. For whole-image CNNs, neighboring receptive fields
make a changed-cell cache much harder: invalidate halos and dependent features.
This is a later systems experiment, not the first backbone implementation.

## Deployment contract

Export encoder-only tensors with ordered names/shapes, dtype/layout, operations,
activation parameters, normalization, preprocessing, data-split identity, training
revision and hashes. Load before initializing new task-specific core/heads, with
explicit ownership of random streams. Full-policy checkpoints must embed or bind
the exact frozen backbone and preprocessing; a raw same-sized weight blob is not
sufficient compatibility evidence. Restore and evaluate the complete pipeline.

Inference uses visible pixels plus the already-declared geometry. No annotations,
hidden object states, future frames or teacher outputs may bypass the learned
visual path. Temporal history at inference must match training. Include a normal
uncached pixel path so the artifact remains usable outside the sprite renderer.

The eventual upstream change should be a small encoder and explicit load/feature
interface after evidence selects it. Keep pretraining/search tooling external to
the clean environment code, with native C/CUDA deployment and the working CUDA
stack preserved.
