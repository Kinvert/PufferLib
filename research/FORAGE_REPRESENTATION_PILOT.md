# Native Forage representation pilot

October 21 clarification: this pilot trains reconstruction autoencoders; its
classifier is a separate diagnostic, not their training objective or the primary
reconstruction verdict. Kinvert's intended deliverable is a smaller **spatial
feature grid** consumed by another CNN, not the current 128-value export. See
[the spatial compressor contract](SPATIAL_AUTOENCODER.md) for the next design.
Keep object/marking reconstruction results distinct from classifier results.
The new spatial implementation runs through
[NEXT_5090_SPATIAL_AUTOENCODER.md](../NEXT_5090_SPATIAL_AUTOENCODER.md); commands
on this page remain the retained vector-model diagnostic.

Implementation entry points:

|File|Role|
|---|---|
|`forage_representation.py` / `.sh`|Bounded preparation, collection, qualification, comparisons and offline receipt audit|
|`forage_encoder_env.h`|Build-only adapter; keeps the actual Forage environment unchanged|
|`forage_annotations.h`|Visible final-image cell identity and per-skin foreground masks|
|`forage_region_weights.cuh`|Class/instance-balanced foreground plus separate background loss weights|
|`forage_encoder_probe.cuh`|Full-frame features/reconstruction, native CUDA linear probe, per-object/mark metrics|
|`forage_probe_checks.cuh`|Independent CUDA loss/head references, finite differences and bad-reconstruction fixtures|
|`test_forage_annotations.c`|ASan/UBSan native raster alignment and appearance-independent world/RNG checks|
|`recipes/forage_representation.ini`|Explicit data, time, pretraining and probe budgets|

Run instructions are in [NEXT_5090_FORAGE_REPRESENTATION.md](../NEXT_5090_FORAGE_REPRESENTATION.md).
This implements an initial slice of [the information evaluation plan](ENCODER_INFORMATION_EVALUATION.md).
It does not implement every planned localization/history/paired-distance test.

## Pixel/target contract

Original Forage is 20x15 cells with full 32x32 RGB sprites: 640x480 pixels.
Every sample is the unchanged native observation buffer, captured before stepping.
Collection stops at a terminal rather than mislabeling an auto-reset frame.
The target is a separate row-major 300-byte identity grid, with the visible player
overlay replacing any obscured underlying label. It is not a scalar reward map.
The 13 classes are the native `FORAGE_NAMES`/enum order.

For original/palette/mirrored/textured skins, masks follow the source glyphs and
their exact 2x enlargement/mirroring. Imported sprites use their source alpha.
Transparent pixels match the actual cached ground tile. No `RGB != black` rule
or extra bounding-box rendering is used. Each cell is a known evaluation unit;
the encoder is never given a crop or the true class at inference.

## Balanced reconstruction objective

For each frame, normalize foreground error separately within each visible object
instance, average instances of the same class, then average present non-ground
classes. Border trees therefore cannot outweigh apples by their instance count.
The remaining visible ground/background has a separate normalized term:

```text
objective = object_weight * mean_present_class(mean_instance(foreground_RGB_MSE))
          + background_weight * background_RGB_MSE
```

Default coefficients are 1 and 0.1. Ordinary raw-image MSE remains separately
logged. `loss_mode=0` uses the existing reconstruction objective; `loss_mode=1`
uses these annotation-derived weights and requires `edge_weight=0`.
No masks/labels enter the encoder. This is privileged auxiliary training, not
pure unsupervised reconstruction. It may still underweight the tiny distinguishing
markings within a large apple; the diagnostic is designed to expose that failure.
No improvement is assumed from implementing the loss.

Validation padding has zero derivative; each active frame retains its own
object/background normalization. The new CUDA check verifies each class's weight
mass, the weighted objective and all input derivatives at tolerances small enough
that an all-zero gradient cannot pass. Generic network/Adam checks are retained.

## What the recognition probe measures

The exact exported 128-dimensional post-ReLU latent is frozen. A linear head with
300x13 outputs predicts the complete visible grid. It has 503,100 parameters,
including biases, intentionally held identical across all candidates. It is a
diagnostic head and does not become part of the deployment encoder.

Only training frames determine feature mean/scale, inverse-frequency class loss
weights, normalization and deterministic minibatch order. Adam/softmax and all
neural operations run native CUDA. CPU work is collection, metadata, scalar
metrics and reporting. The encoder/decoder parameter bytes are checked unchanged.

Report collectible macro F1 separately from ground, trees and player; keep every
class's confusion, recall, support and false positives. Apple/spotted-apple and
mushroom/toadstool balanced accuracies restrict the choice to the two class logits
at true object cells. That conditional score diagnoses identity; it does not give
credit for finding the object. Pair-only success with poor full-grid recall must
not be called successful perception.

Constant features train the same head and reveal position/class priors. Random
features start from exactly the pretraining model's step-zero parameter bytes.
Both learned cases share the same initial full state and differ in the declared
loss only. A per-cell training-majority baseline is also tabulated. The final
checkpoint is fixed by schedule/cap, not selected by recognition performance.

Reconstruction metrics include global/background MSE, separately averaged visible
object-instance errors for each class, and the exact RGB-difference pixels between
each lookalike pair's native templates. The last metric makes errors in the
distinguishing markings visible even when silhouettes and color counts match.

## Implemented checks and limits

The host sanitizer compares 60 full native observations with their labeled cached
templates across reset/step and five skins. Matching labels/player/world RNG across
skins checks that appearance does not alter the task. Native CUDA calibration
constructs all classes and tests identity, all-background, erased apples and
swapped apple types for all skins. Region sums/counts are checked against an
independent scalar calculation; the classifier's loss/gradients use a direct
float64 CUDA reference and eight selected finite differences. Qualification also
checks actual AE resume and repeated probes reloading the same encoder checkpoint,
and requires memory instrumentation. Saved probe-head reload is not a separate
qualified deployment interface.

These are implemented checks, not assertions that GPU checks have passed locally.
Consult [the build/host result receipt](results/forage-representation-20261020/README.md)
for actual execution status. No prior failed packet or unsupported WDDM tool is
upgraded by a successful host test.

The collected random-play pilot covers few independent worlds and short episodes.
The same world across skins is a paired appearance comparison, not another world.
Seen/unseen skins are separate report rows. No held-out game, temporal sufficiency,
generalization guarantee or downstream return/time advantage is established.
Next experiment depends on what fails: decoder, location capacity, fine identity,
appearance robustness or actual learner adaptation. Do not automatically expand
pretraining time because global MSE decreases.
