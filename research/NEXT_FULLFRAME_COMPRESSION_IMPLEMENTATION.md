# Implementation handoff: full-frame visual compression, first increment

October 19, 2026. Inspected HEAD:
`9481b9f972d454bf396b2f91b5f1f5b20216b130`, `/home/claude/cnn`.
**Acceptance specification.** The required tools below are now implemented;
runtime qualification is still pending. Read `FULLFRAME_COMPRESSION_PROGRESS.md`
for the live packet before launching anything. Optional standalone `timing`
is implemented with separate GPU-free `prepare-timing`; its GPU execution still
requires accepted math/smoke. The required smoke also retains native timing and processSPS.
This document does not itself launch a campaign.

## 1. Assignment and decisions already made

Implement three native FP32 image-to-feature pipelines, connected to PufferLib
training and deterministic evaluation, with captured builds, numerical tests,
encoder-only export/import, bounded GPU smoke comparisons and a compact report.
The outcome is a usable, pretrainable full-frame encoder interface, not a trained
general backbone or a claimed Pareto improvement.

Inputs must be actual full rendered RGB frames. No object IDs, sprite-atlas
lookup, tile boundaries, oracle crops or packed player/gaze ROIs. The encoder
can reduce resolution internally. Preserve game rules and human display. Test
arbitrary RGB buffers too: the encoder must not depend on renderer internals.

Fixed first-increment choices:

- Games: existing `flappycnn`, then `pongcnn`, using their `CNN_IMAGE_V1` paths.
- Raw HWC: Flappy **640×420×3**, Pong **640×620×3**. Preserve aspect ratios.
- FP32 NHWC RGB in [0,1]. No new grayscale/crop/frame-stack/action-repeat or
  geometry inputs. All policies retain their game's existing learned actions.
- Existing MinGRU H128/L1 and policy/value heads; unchanged learner/optimizer.
- Three fixed pipelines, no architecture/seed sweep. Pipeline0 is the initial
  pretraining candidate after this increment passes.
- No large dataset, pretraining trainer, frozen rollout cache, cuDNN rewrite,
  optimizer change or overnight campaign in this increment. See §9 for what follows.

Read root AGENTS.md; work directly in `~/cnn`. Preserve unrelated work and frozen
evidence. Use direct edits and simple C/CUDA. Reuse the existing uv Python3.12
venv for configuration/process/report glue. No CUDA/driver/system/Torch installs.
Ordinary PufferLib behavior must stay unchanged when the research flag is absent.
No commit or push is requested merely by this document.

## 2. Existing interfaces to reuse

Read these actual files before implementation:

| Existing path | Responsibility |
|---|---|
|`ocean/connect4cnn/image.h`|Raw image shape/layout and raster helpers|
|`ocean/flappycnn/image_v1.h`, `ocean/pongcnn/image_v1.h`|Full-frame observations; leave unchanged|
|`research/build_image_policy.sh`|Source capture, native build and host metadata pattern|
|`research/policy_metadata.cu`|Full-policy registration/parameter layout|
|`src/ocean.cu`|Encoder dispatch and research-gated include|
|`src/algo.cu`|Encoder callbacks, allocator, initialization and learner contract|
|`ocean/connect4cnn/nature.cu`, `flex.cu`, `flex2.cu`|Primitive kernels and existing constructors|
|`research/native_encoder_reference.cu`, `test_sprite_packed.cu`|Independent CUDA reference/harness patterns|
|`research/image_training_smoke.py`, `run_image_evaluator_smoke.py`|Native training/checkpoint and exact-evaluator acceptance patterns|
|`research/image_curve_feedback.py`, `image_curve_bindings.py`|Existing full-frame curve feedback for later reuse|
|`research/image_campaign_budget.py`, `image_reporting_reserve.py`|Complete-panel budget/reporting patterns|

The endpoint-only feedback finding applies to `sprite_overnight.py`. The separate
image workflow already implements curve feedback; do not rebuild it unnecessarily.
Its legacy schemas do not automatically admit a new encoder or preprocessing.

Do not route this through packed sprites or change `SPRITE_CELL`. `src/nature.cu`
and `ocean/connect4cnn/nature.cu` are different integration contexts. Represent
internal reduced dimensions in layer descriptors, not global `OBS_*` macro changes.

## 3. Exact pipelines

Use research-only `[policy] encoder=8`, gated by new compile definition
`CNN_COMPRESSION_RESEARCH`, and numeric `[policy] compression_model=0/1/2`.
ID8 was not found in inspected dispatch/recipes. Recheck before editing if the
repository advanced; if occupied, allocate the next unused ID consistently.
Reject unsupported values; no silent fallback.

All trainable convolutions/projections have bias and ReLU. No normalization,
dropout, residual, dilation or gating in v1. Use current Kaiming/zero-bias
initialization and record tensor registration order. Different graphs need not
have identical random tensor bytes; conventions and seeds must be matched.

### Pipeline0: learned compression

```text
RGB H×W×3
 -> Conv4×4 / stride4 / SAME / C8 + ReLU
 -> Conv3×3 / stride2 / SAME / C16 + ReLU
 -> Conv3×3 / stride2 / SAME / C16 + ReLU
 -> adaptive average pool4×4×16
 -> flatten256 -> Linear128 + ReLU
 -> existing MinGRU H128/L1 -> policy/value heads
```

SAME: output=`ceil(input/stride)`; total padding=
`max((out-1)*stride+kernel-input,0)`; leading padding=floor(total/2), remainder
trailing. Dilation1/groups1. ReLU derivative at zero is0.
Adaptive bins: start=`floor(i*input/output)`, end=`ceil((i+1)*input/output)`;
average all entries in the bin. Overlapping bins on odd dimensions are intentional;
backward gathers all their contributions in fixed order.

Flappy stages:160×105×8 →80×53×16 →40×27×16 →4×4×16 →128.
Pong stages:160×155×8 →80×78×16 →40×39×16 →4×4×16 →128.
Encoder parameters including biases: **36,776**. Spatial bottleneck256scalars.
This is a test hypothesis, not proven small-detail preservation or speed.

### Pipeline1: area reduction control

```text
RGB H×W×3
 -> nonoverlapping4×4 RGB means, partial edges divided by actual count
 -> Conv3×3 / stride1 / SAME / C8 + ReLU
 -> Conv3×3 / stride2 / SAME / C16 + ReLU
 -> Conv3×3 / stride2 / SAME / C16 + ReLU
 -> adaptive4×4×16 -> flatten256 -> Linear128 + ReLU -> same core/heads
```

Area output=`ceil(H/4)×ceil(W/4)×3`, fixed row-major sums per channel.
Specify this operation directly, not an undocumented image-library resize.
Post-first-conv shapes match pipeline0; parameters **36,608**. The first filters
differ, so this is a system ablation, not a parameter-identical comparison.

### Pipeline2: practical resized Nature

Same area4 operation, followed by valid Nature Conv8/s4 C32 → Conv4/s2 C64 →
Conv3/s1 C64 → flatten → Linear128; bias/ReLU after each trainable layer.
Flappy reduced input160×105 gives39×25×32 →18×11×64 →16×9×64.
Pong reduced160×155 gives39×37×32 →18×17×64 →16×15×64.
Label it **resized RGB Nature, H128 projection**. It is neither original DQN nor
the retained full-input Nature. Keep old builds/results untouched.

All three start from identical raw frames. Include internal resize/pooling in
timing. In metadata, keep raw environment shape and internal encoder shapes
distinct; do not describe the environment as emitting reduced images.

## 4. Files to create and allowed integration changes

| Proposed new file | Required implementation |
|---|---|
|`research/fullframe_compression.cu`|Private encoder callbacks/descriptors and three pipelines|
|`research/fullframe_compression_reference.cu`|Independent float64 CUDA reference, no production convolution/pool/GEMM helpers|
|`research/test_fullframe_compression.cu`|Outputs/all gradients, eager/graph/process repeat, layouts, export/import|
|`research/fullframe_compression.sh`|Entry point: prepare, inspect, host-check, math, smoke, timing, report|
|`research/fullframe_compression.py`|Preparation, bounded process supervision, receipts and reporting only|
|`research/recipes/fullframe_compression.ini`|Fixed models/shapes, learner overrides, seed roles and caps|
|`research/tests/test_fullframe_compression.py`|Meaningful manifest, allocation, budget and receipt-rejection tests|
|`NEXT_5090_FULLFRAME_COMPRESSION.md`|Tested remote commands/scope after implementation, no overnight launch|

Modify `src/ocean.cu` only as necessary to include/dispatch the new encoder under
both research and image flags. Existing0/2/4 dispatch is unchanged without the
new selection. Extend native metadata consistently. Do not enlarge shared layer
capacities or replace baseline kernels solely to accommodate the private graph.

Reuse compatible existing primitives in production, with a simple private loop
over explicit dimensions. Do not pass a resized tensor into a whole Nature
forward that infers its batch from raw `OBS_HEIGHT*OBS_WIDTH`. Allocate scratch
once through the normal allocator; no allocation/I/O in forward/backward.

Build the exact captured source bytes, including modified dispatch and metadata.
Record all hashes, recipes, compiler/runtime identities and source dirtiness.
HEAD alone is insufficient. Reject source mutation during preparation. Preserve
the unrelated weight-mirror edit; unset `PUFFER_WEIGHTS_MIRROR` for comparisons.

## 5. Encoder-only export/import contract

Export `encoder.bin` plus `encoder.json`, schema
`fullframe-compression-encoder-v1`. Record pipeline/operations, raw HWC, internal
shapes, dtype/layout/range, padding/pooling, ordered tensor names/offsets/shapes,
counts, source identities and binary SHA256. Include encoder projection weights,
exclude scratch/padding/core/heads. Bind raw HWC strictly in v1 even if some
graphs could later support cross-size import.

Expose string `[policy] compression_init = None` (or an export-directory path).
During the encoder initialization callback, perform ordinary initialization
and consume the same random stream, then overwrite **only encoder** weights
before returning. The remaining policy initializes normally. Import must consume
no extra random draws. Same-seed scratch/import tests must retain byte-identical initial
core/head tensors. Validate shape/operation/hash/length/finiteness before device
mutation; reject incompatible equal-sized blobs too. Atomically publish fresh
exports/receipts. Scratch-weight roundtrips are integrity tests, not pretraining.

Full-policy experiment checkpoints get immutable sidecar manifests binding
preprocessing, encoder graph, initial backbone identity and learner. Evaluation
must validate that binding before loading. Do not silently change global legacy
checkpoint format or bless old files as compatible.

Deliver `fullframe_compression.sh export-encoder --packet PACKET --game GAME
--model N --checkpoint FILE --out FRESH_DIRECTORY`. It must validate the full
checkpoint layout/manifest before extracting the encoder's registered tensors;
it must not guess that the first N floats are compatible. Add export/import
roundtrips and wrong-model/same-sized-wrong-manifest rejection tests. Full-policy
reload clears `compression_init` so the saved policy is authoritative; retain
the original initialization hash in provenance without requiring its old path.

## 6. Tests and bounded GPU packet

### Preparation and numerical acceptance

Host preparation compiles/configures/registers shapes and runs raster/manifest
tests; it is not CPU neural training or GPU validation. Existing game raster/RNG
parity must remain unchanged. Add deterministic RGB fixtures with no game IDs:
translated dots/edges, RGB ramps and clipping. Tiny fixtures are not a dataset.

CUDA reference cases: all3pipelines,160×192 and161×193 RGB, batches1/3,
positive/zero/signed inputs. Both source shapes keep resized Nature valid.
Also use actual Flappy/Pong shapes at B1/B16, positive and signed cases. Check
native allocation estimates before execution; no silent size/batch reduction.

Compare every output and trainable gradient. Test input derivatives in the
harness even if the production callback returns none. Cover area edge divisors,
adaptive overlap, padding, channel order and raw/internal stride addressing.
Finite differences use smooth coordinates and explicitly report nonsmooth skips.

For new encoder fixtures, predeclare
`abs(error) <= 6e-5 + 8e-4*abs(reference)`; area-filter checks use
`2e-6 + 2e-6*abs(reference)`. Preserve stricter existing Nature regression
contracts; these are not replacement tolerances. For signed cases, compare
backprop on actual native activation branches independently, alongside smooth
float64 checks. Record branch disagreements/failures; no post-failure waiver.
Eager/graph and two fresh processes must match bytes on the same pinned stack.

October19 implementation clarification, numerical protocol `ffc-native-branches-v2`:
the initial packet failed at161×193/B1/model0/signed. A separate diagnostic found
one native activation1.4901161193847656e-8 where the double ReLU was0; the
independent gradient on actual native branches agreed to1.05e-8. Preserve that
failed packet as failed. In a fresh packet, compare every intermediate forward
and require every output/parameter/input gradient against the independent
actual-branch oracle at the original tolerances. Also require full-double
gradients when branches agree. Signed branch differences must lie within an
explicit2e-6 post-ReLU neighborhood and remain recorded with the differing
full-double gradients; positive/zero fixtures must have no branch differences.
This qualifies the derivative of the executed native network, not equality to
a double network that takes another branch. No tolerance was widened.

### Training and exact evaluation

After math passes:3pipelines ×2games ×2same-seed repeats =12training jobs,
seed7,8192decisions/job. Apply identically:

```ini
[base]
seed = 7
async = 0
cudagraphs = 1
checkpoint_interval = 16
[vec]
total_agents = 16
num_buffers = 1
num_threads = 2
num_policies = 1
hist_policy_percent = 0
[policy]
encoder = 8
hidden_size = 128
num_layers = 1
[train]
gpus = 1
horizon = 16
minibatch_size = 16
total_timesteps = 8192
[selfplay]
enabled = 0
```

Freeze other learner/reward/physics settings from `config/default.ini` plus
each game's INI, shared across models. Strip inherited sweep sections and
resume/load settings. Set `compression_model` per candidate. The learner encoder
batch is16observations, not16×16; confirm actual registered buffers. The inspected
trainer's checkpoint interval counts epochs:256decisions/epoch ×16epochs gives
4096, with another at8192. Assert receipts match these two checkpoints.

Repeat checkpoints must match hashes at both steps. From the first repeat's
final weights, run graph, repeated graph and eager exact evaluation for each
of6policies:18processes,2assignedepisodes each,1slot,max256steps, development
suite420. Add2slot evaluation for every policy with the identical episode IDs
and seeds:6processes. Reuse existing evaluator seed semantics. Keep1337 for
later learning evaluation; never sweep seeds. Preserve capped/missing outcomes.
Zero score does not fail a plumbing smoke; omitted episodes do.

One GPU process at a time; use shared reservation and idle checks. New packet
limits: math≤900s; training+eval≤900s; training process≤45s; eval process≤30s;
artifacts≤10GiB with20GiB free reserve. Initial5060 estimated allocation target
≤4GiB. If it does not fit, report it and prepare5090 execution with the same
contract, rather than secretly changing inputs. Failures stop with receipts;
no automatic retries. No TUI watching. Existing short5060 authorization is not
authorization for an overnight search.

## 7. Command interface to implement

**Implemented required commands; use a fresh packet and respect the live queue:**

```bash
cd /home/claude/cnn
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash research/fullframe_compression.sh prepare --out build/fullframe-compression-v1
bash research/fullframe_compression.sh inspect --packet build/fullframe-compression-v1
bash research/fullframe_compression.sh host-check --packet build/fullframe-compression-v1
bash research/fullframe_compression.sh math --packet build/fullframe-compression-v1 --hardware 5060
bash research/fullframe_compression.sh smoke --packet build/fullframe-compression-v1 --hardware 5060
bash research/fullframe_compression.sh report --packet build/fullframe-compression-v1
```

Prepare is GPU-free, creates a fresh directory and refuses overwrites. Inspect
and report are offline, execute no policy and validate receipts/hashes. Separate
stage receipts allow moving from successful math to smoke without redoing math;
failed/completed stages cannot rerun under the same packet. Validate hardware
identity. A5090run uses its own fresh packet. Never restart a historical allocation.
Remote delivery must need only pull/read/launch, not coding or package installs.

Optional `timing` stage after acceptance: CUDA-event forward and forward+backward
at B1/16,20warmup and100timed iterations, five fresh process replicas per
pipeline/game shape, total cap300s. Include resize/pool/copies; record encoder
timing separately from full-trainingSPS. Smoke processSPS is startup-dominated.
For profiling, follow shared GPU-profiling documents and require actual GPU
activity, not merely NVTX. Keep all timed workloads exclusive.

## 8. Definition of done

- Three native pipelines; independently verified shapes/parameter counts.
- Legacy builds/metadata unchanged when the flag is absent.
- Raw image hashes/game/RNG parity preserved; no semantic state enters the CNN.
- Independent numerical, graph/process, memory and export/import gates pass.
- Twelve trainings match paired checkpoint hashes; all24exact evals preserve
  assigned episodes and graph/eager/slot-count equivalence.
- Report every checkpoint, failure, processSPS/native timing, allocation,
  source/learner/preprocessing identity and scope limitation.
- Compact report at `research/results/fullframe-compression-v1-<hardware>-<date>/README.md`;
  raw weights/arrays stay ignored under build.
- Tested5090handoff distinguishes smoke/timing from future learning; no speed,
  quality or generalization claim beyond the actual measurements.
- Selective future commits preserve unrelated edits and use Kinvert's fork only
  if publication is requested. No system/dependency changes.

If a gate fails, retain/diagnose it and report incomplete delivery. Compilation
alone or pictures do not satisfy this assignment. Do not expand the search until
this increment works.

## 9. Next increment chosen, not included in this smoke

First pretraining objective: pixel-aligned visual identity/foreground supervision
on **full scenes**, using a small training-only spatial head before4×4readout.
Classes describe visibly identifiable objects, not reward tiers or hidden roles.
Labels follow final rendered visibility; occlusion/unknowns cannot become oracle
inputs. No full bounding-box framework is necessary for the first trial.

Follow `PIXEL_PRETRAINING_DATASET_PLAN.md`. Before collection, freeze scene/game/
asset/trajectory splits, foreground-balanced loss and sampling on development
data; retain unweighted loss and rare-detail/per-class diagnostics. Exact loss
weights and collection size require a separate bounded pretraining specification;
do not invent a large collection while implementing this encoder interface.

Compare pipeline0 scratch, visually pretrained, and deterministic-autoencoder
reconstruction controls with matched data/compute. Compare fine-tuning against
frozen conv prefix plus trainable projection/core. Temporal prediction is the
next objective ablation; large natural-image teachers are optional. Count all
data generation/pretraining/teacher cost.

Implement stored frozen-prefix rollouts only after the representation supports
useful learning, with cached/recomputed equivalence and encoder-version binding.
Then integrate qualified graph choices into versioned native PROTEIN feedback.
Keep the fixed training seed set and separate held-out worlds/games; never claim
that seen development games became independent tests just by changing seeds.
