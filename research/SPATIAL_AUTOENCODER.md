# Intended deliverable: a learned spatial image compressor

Kinvert's clarification, October 21, 2026. Research code and instructions are in
[the spatial 5090 handoff](../NEXT_5090_SPATIAL_AUTOENCODER.md). Consult execution
receipts for validation scope; implementation alone does not establish useful
compression or improved downstream learning.

Train an autoencoder on full game frames at roughly human-visible resolution,
preserving each game's aspect ratio. Reconstruction must preserve important
objects and their fine visual differences: for example, a normal apple versus
a speckled apple in Sprite Forage. Correctly reconstructing a large background
must not hide failure to reconstruct those objects.

```text
PRETRAINING
full RGB image -> encoder -> smaller spatial feature grid -> decoder -> image
                                                                    |
                            compare with original; emphasize objects/markings

GAME LEARNING / DEPLOYMENT
full RGB image -> saved encoder -> smaller spatial feature grid -> CNN -> policy
```

The intermediate grid is a learned, compressed image representation. It retains
height and width and may have multiple feature channels rather than three RGB
channels. Its dimensions are a design choice to test, not a fixed target taken
from an illustrative number. Do not replace this grid with a single pooled
vector before passing it to the downstream CNN.

## Training and evaluation

Train with reconstruction loss weighted heavily toward visible object regions.
Normalize object contributions so background area, large sprites or numerous
trees cannot dominate smaller important items. Keep a separately weighted
background term: it still matters for context and for detecting invented objects.
The existing class/instance-balanced Forage objective is a starting point.

Measure reconstruction on held-out images, reporting per-object errors and
errors on the pixels that distinguish lookalikes. Show original/reconstructed
object crops alongside the numbers. Report background error separately. The
current objective weights object foreground; its marking metric measures the
distinguishing pixels separately. Those are not the same as an implemented
extra marking-weighted training term.

Object labels and masks define loss/evaluation targets in pixel space. They
are not inputs or oracle crops for the encoder or deployed policy. Classification
is optional supporting evidence about the representation; classifier F1 is not
the autoencoder objective or the primary verdict on reconstruction quality.

After pretraining, discard the decoder and train the downstream game policy on
the compressed spatial grid. Establish a frozen-encoder baseline first so this
is actually a reusable image compressor; any fine-tuning is a separately named
comparison. Neither path has been validated downstream by the existing pilot.

## Difference from the retained vector implementation

`fullframe_compression.cu` model 0 currently runs three convolutions, pools their
spatial map to 4x4x16, then projects it to 128 numbers. `autoencoder.cuh` reconstructs
from that vector. The existing export supplies the vector to the policy core.
That is a reconstruction autoencoder, but it is not the spatial compressor above.

The new opt-in implementation lets the decoder reconstruct directly from the
spatial encoder output, trains that path, exports an encoder-only spatial artifact,
and gives that artifact a downstream CNN consumer. Merely exporting an intermediate
map from the old vector-trained model is a useful possible control, not an
equivalent implementation of this training objective. Preserve old checkpoints,
schemas and results as vector-model evidence.

The spatial artifact must declare raw image shape/normalization, latent height,
width, channels, layout and dtype, encoder architecture/weight hashes, and source
checkpoint. Keep the deployment path free of decoder and diagnostic-classifier
work. Verify exact export/reload features and native forward/backward behavior,
repeatability, resume and memory checks before longer training.

## What would demonstrate success

1. The reconstruction preserves objects, positions and distinguishing markings
   on held-out frames, with background and corruption controls reported.
2. The exported encoder produces a smaller spatial representation with the same
   features after reload; downstream inputs contain no simulator labels.
3. Matched downstream training learns effectively from those features. Record
   total time and SPS including full-frame encoder inference and downstream CNN
   work; fewer feature values alone do not establish a speedup.
4. Compare with simple resizing and training directly on pixels under matched
   tasks/evaluation. Report pretraining cost separately as well as any amortized
   deployment benefit. Preserve all failures and weak seeds.

This documentation update launches no corpus collection, GPU job or training
and restarts no completed packet. User-reported remote overnight reconstruction
improvements remain attributed remote findings until their receipts are imported
and audited here; old local memory-tool failures remain unchanged.
