# Measure preserved game information before scaling pretraining

October 20, 2026. Proposed evaluation/implementation contract, **not implemented
or executed** by this research update. No dataset, GPU run, dependency change or
training-code change is part of this note. Literature and primary sources are in
[the companion reading notes](ENCODER_EVALUATION_LITERATURE_20261020.md).

## What our current measurements actually mean

At research delivery `50a29b65`, the autoencoder logs RGB MSE and an optional
edge-weighted objective. `research/autoencoder_pretrain.py::qualify` requires
validation MSE to decrease during an eight-update canary. That is a training-loop
check, not encoder accuracy. It could improve by learning the typical background.
The optional qualification never executed locally because the shared GPU stayed
occupied. No trained autoencoder or useful-perception result currently exists.

`research/autoencoder.cuh::ae_loss` weights visible contrast when `edge_weight>0`,
then divides by the total number of active RGB values. This is not a semantic
mask, per-object normalization, or background-balanced metric. Checkerboards,
HUD edges and irrelevant texture can receive weight too. Objective values at
different edge weights are not directly comparable; retain ordinary MSE.

An illustrative calculation, not an experiment: a 640x480 black RGB image with
one white 16x16 ball has MSE `256 / 307200 = 0.0008333` against an all-black
prediction. Its PSNR is about 30.79 dB, despite deleting the entire ball. Matching
99.917% of pixel channels would be a similarly misleading description. Actual
game frames contain other objects; the example isolates area imbalance.

Also separate encoder and decoder quality. A weak decoder can blur an object
whose location is still available in the latent. Conversely, a decoder can draw
a plausible average ball without encoding its actual location. Evaluate both
reconstruction and the exact exported latent, then actual control.

## Priority order

|Order|Deliverable|Why it comes first|
|---|---|---|
|1|A deterministic annotated evaluation slice plus deliberately bad baselines|Check that the evaluation notices a deleted ball before trusting any encoder score|
|2|Per-object reconstruction and fixed-capacity frozen-latent probes|Distinguish background fidelity, object loss, position loss and decoder weakness|
|3|Matched appearance and critical-cue pairs|Check nuisance robustness without accepting a constant or cue-blind representation|
|4|Small matched downstream RL confirmation|Determine whether the diagnostic gain helps decisions at measured training cost|
|5|Only then expand objectives/data/pretraining time|Avoid spending an overnight run optimizing an uninformative surrogate|

## 1. Labels describe the final image; inputs stay full pixels

For native Pong and Flappy, derive annotations from the **actual full-frame RGB
rasterization**, not a separate miniature renderer or raw world-coordinate box.
The current paths are `ocean/pongcnn/image_v1.h` and
`ocean/flappycnn/image_v1.h`; shared primitive rounding/clipping lives in
`ocean/connect4cnn/image.h`. The annotation hook is future work.

Record frame/episode ID, decision index, raw HWC, skin, renderer/source hash,
visible object role/class, half-open image box `[x0,y0,x1,y1)`, and preferably a
visible-pixel mask. An optional compact span list avoids storing full ID images.
Mirror clipping, ellipse/outline shapes, flips, draw order and occlusion exactly.
Apply Pong's upward-world to downward-image conversion and HUD exclusion in
the same place as the image. Color inversion does not invert object identity.
No comparison such as `RGB != black` defines foreground across all skins.

For outline pipes, distinguish visible outline pixels from physical collision
geometry. Label the visible gap/context separately; the empty space between two
pipes can matter more than the pipe interiors. Never label an occluded object
as visible because its simulator state exists. If labels cannot be inferred from
the declared image/history, mark them unavailable for that probe.

Keep three uses separate:

|Use|Are simulator-derived labels allowed?|Do they change the experiment?|
|---|---|---|
|Evaluation masks/probe targets only|Yes, in separate files|No extra policy input; reveal the annotation dependence|
|Auxiliary pretraining targets|Yes, explicitly declared|This is supervised/privileged pretraining, not unlabeled reconstruction|
|Policy inputs or oracle-selected crops|Only in a separately named state/oracle control|Changes the observation advantage and cannot support a pure-pixel comparison|

Here ROI means an **evaluation/loss region**. It does not mean our earlier
learned-gaze action interface. Discovering or choosing attention regions at
deployment is a different architecture experiment. Whole-frame encoders should
continue seeing identical full images in this comparison.

## 2. Evaluation data and checks for leakage

Prepare a small declared pilot before collecting a large corpus: for example,
32 whole trajectories per game, at most 32 retained frames each, streamed for
measurement. This is a proposed budget, not permission or an executed collection.
Retain separate natural-trajectory and curated stress-case reports; neither may
silently replace the other. Record all sampling/coverage shortfalls.

Use disjoint pretraining, probe-training, development and final-test trajectories.
Choose all seed lists before training; never sweep seeds. Development worlds
may use fixed bases 420/1337; final worlds need a new frozen list. These bases
alone do not prove separation: bind actual generated IDs and compare image and
sequence hashes. Do not randomly split adjacent frames into train and test.

Random actions often produce short dying/reset trajectories. Report coverage of
object locations, sizes, edge positions, near misses and longer successful play.
If using a trained collector, freeze its checkpoint and action protocol. Keep
all failed/short episodes. Curated cases may supplement missing coverage but
must be declared, physically/raster-valid and separately tabulated.

All candidates get the same frame IDs, targets and preprocessing. Keep a locked
final test out of objective tuning. Seen-game/new-world, unseen appearance, and
whole held-out game are separate transfer claims. Pretraining on both Pong and
Flappy does not make either game unseen.

## 3. Metrics that cannot hide the ball behind the background

For RGB reconstruction error `e(p) = mean_c((prediction-target)^2)`, define for
each visible object instance with mask `M_j`:

```text
object_MSE(j) = sum_p M_j(p) * e(p) / sum_p M_j(p)
```

Report the ball, each paddle, bird and each pipe role separately, then macro
averages across instances and roles. A large pipe must not outweigh a tiny bird
simply by covering more pixels. Missing/clipped objects need explicit visibility
and sample counts; an absent class produces N/A, not a perfect score. Also retain
natural frequency-weighted results so equal-instance scoring is transparent.

Add fixed context regions: a declared ring around each object, pipe-gap corridors,
and the remaining background/HUD separately. Exclude other objects from background
and disambiguate overlapping regions. These catch ghost objects, enlarged blobs,
and moving an object outside its ground-truth mask. Report counts of false object
detections on genuinely empty regions; foreground-only MSE would miss them.

For Pong/Flappy, an independently specified color/shape extractor can evaluate
reconstructed visible-object centers and presence against annotations. Set its
thresholds using original images and development corruptions, freeze them across
candidates, and report failures. Never give it simulator positions as predicted
coordinates. This decoder diagnostic does not replace latent probes.

Mandatory summary columns:

|Diagnostic|Report|
|---|---|
|Global fidelity|MSE; optional PSNR/SSIM; no invented percent encoder accuracy|
|Critical objects|Per-role instance-normalized MSE, presence precision/recall, center error|
|Position|Pixel MAE and recall within predeclared 1/2/4-pixel tolerances, plus size-normalized error|
|Local context|Ring/gap reconstruction error and false detections|
|Fine identity|Per-class recall, macro F1 and confusion matrix for lookalikes|
|Coverage|Counts by object area, contrast, image edge and important event|
|Latent health|Per-dimension variance/dead dimensions; descriptive, not a control score|

PSNR, SSIM or LPIPS alone cannot establish critical-object preservation. Scores
from a recognition network can inherit its blind spots; none is a replacement
for measured location/identity and downstream RL.

## 4. Freeze the encoder and test its exported features

Probe the same post-activation latent that PufferLib receives, including its
projection. A probe on a richer intermediate map is a separately named result.
No decoder, ground-truth box, hidden state, image crop or future frame enters a
single-frame probe. Freeze encoder weights and fit only a diagnostic head.

Use a fixed linear probe first, and one small fixed MLP only as a diagnostic for
nonlinear accessibility. Keep capacity, training data, update budget and tuning
procedure matched. Learned probes and all neural evaluation should run on GPU
when implemented; this document starts no CPU or GPU training.

Proposed targets:

|Game|Single-frame targets|History-only targets, separately evaluated|
|---|---|---|
|Pong|Ball presence/center, paddle extents/centers, ball-to-player offsets|Ball direction and short-horizon intercept using consecutive latents|
|Flappy|Bird center/visible size, visible pipe edges and gap clearance|Vertical motion and short-horizon clearance after a declared action|
|Later sprite games|Visible object identity, dangerous marking, obstacle/exit location|Animation-consistent identity and enemy motion|

Velocity is not generally identifiable from one image. Use a declared history
window and elapsed decision indices. The current collector does not store an
action-aligned temporal label stream; extend its manifest before temporal probes
or SPR-style training. Do not infer sequence timing from filename adjacency.

Include constant/majority and train-mean-position predictors, the same encoder
with random frozen weights, and a simple pixel/downsample reference at its stated
cost. A supervised raw-image probe can check target learnability, but is not a
guaranteed ceiling. A constant encoder should perform like the bias-only probe;
random features may already be useful and cannot be omitted.

Poor linear results do not prove that no nonlinear decoder could recover the
information. Strong probe results establish access to these particular labels,
not Markov sufficiency or a ready-to-play policy. Report this limit explicitly.

## 5. Paired render tests: reject distraction without erasing important changes

Use deterministic, independent rendering RNG. Re-render one saved game state
without advancing physics/world RNG. Keep latent size and preprocessing fixed.
Construct an anchor, a nuisance-only image, and a relevant-change image.

Nuisance examples: background texture changes with identical object pixels and
geometry. Relevant changes: move the ball, change gap clearance, or swap a safe
and dangerous visible marking. Include static objects; frame differences alone
cannot define relevance. Recoloring may be semantic in some games and must not
automatically be labeled nuisance. Flips require transformed position targets,
not unconditional embedding equality.

Report paired latent distances and critical-change ranking accuracy with ties
counted as half, alongside probe predictions. Use a declared distance and any
feature normalization fitted only on probe-training data. An all-constant encoder
gets half-tie rankings and must also fail object-preservation controls. This is
our proposed sensitivity diagnostic, not the published denoising-factor formula.
Distance rankings alone are not a necessary or sufficient control criterion.

Test important details at stride phases and image boundaries: translating a
small object by one pixel can expose loss from strided convolution or pooling.
The current narrow encoder uses a 4x4 adaptive spatial readout before 128 latent
values; that may dilute precise location, but this remains a hypothesis to test.

## 6. Verify the evaluator against intentionally wrong outputs

Before evaluating a learned model, require the following expected behavior:

|Control|What the report must reveal|
|---|---|
|Identity reconstruction|Zero reconstruction error, correct visible-object extraction|
|All black or training-set mean image|Potentially low global MSE but missing-object/position failures|
|Ball/bird deleted, rest unchanged|The named critical object's reconstruction and presence fail|
|Ball shifted 1/2/4/8 pixels|Localization reports the displacement; context catches ghosts|
|Background changed, objects unchanged|Object targets unchanged; report nuisance sensitivity separately|
|Wrong dangerous marking, same silhouette/color totals|Identity confusion despite near-matching global image statistics|
|Constant or shuffled latents|Probes cannot legitimately beat leakage-aware bias controls; paired ranking cannot claim success|

These are planned acceptance fixtures. They were not run in this update. If the
metric calls a deleted ball good, fix the metric before optimizing the encoder.
Freeze expected outcomes and extractor thresholds before evaluating candidates.

## 7. Training changes, only after measuring the failure

First compare plain MSE, the existing contrast-edge weighting, and a new proposed
object/context-balanced loss under matched architecture/data/steps. For the latter:

```text
L = lambda_object * macro_object_MSE
  + lambda_context * macro_context_MSE
  + lambda_background * background_MSE
```

Each region average has its own declared pixel-count normalization. Use fixed
nonnegative coefficients and handle absent/overlapping regions explicitly.
Do not silently renormalize differently between candidates. Keep some global/
context supervision in the initial experiment; exclusively reconstructing small
foreground masks could discard walls, gaps, static targets or future-relevant cues.
This formula is our proposal, not an implemented loss or a Segmentation Dreamer
reproduction. Training masks add supervision and their collection cost must count.

If pixels look poor but latent probes are good, investigate decoder capacity
before enlarging the online encoder. If both lose position, inspect stride,
pooling/readout and bottleneck capacity. If appearance tests fail, test safe
augmentations and spatial/temporal objectives. If probes pass but RL fails,
investigate history, coverage and learner adaptation instead of declaring victory.

## 8. Downstream evidence and the actual Pareto claim

Compare the same architecture from scratch versus pretrained/fine-tuned first;
frozen pretrained inference is a separate control. Later compare qualified Nature
and IMPALA under the same pixel/action/task contracts. Hold per-game learners
matched, independently train each policy's actions, save the planned checkpoint
schedule, and compare full score/time curves without choosing checkpoints by
final-test score. Use the seven fixed replicas 7/42/69/123/173/2027/9001 and
independent evaluation worlds. Never sweep or replace weak seeds.

Use paired seed differences and uncertainty across training replicas; bootstrap
episodes/trajectories for probe uncertainty rather than pretending correlated
frames are independent. Distinguish one shared pretrained checkpoint from seven
independently pretrained backbones: downstream replicas do not measure pretraining
seed variability. Retain missing/failed cases, per-game regressions and storage.

Track pretraining/data/annotation/teacher cost, online encoder time, learner time,
process SPS and actual RL quality. Shared GPUs invalidate clean speed comparisons;
schedule timed comparisons with exclusive access. Report pretraining as a separate
cost and in total/amortized costs with a declared reuse count. Initialization
alone does not remove online convolution work. A frozen cache remains separate
implementation work.

There is no universal justified probe-accuracy cutoff today. Set pilot thresholds
from declared pixel tolerances and the evaluator controls before examining final
candidate results. Passing diagnostics earns a small RL experiment, not an SOTA
claim. Better measured downstream score/time behavior is the evidence required
for expanding the actual Pareto front.
