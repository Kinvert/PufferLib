# Shared packed sprite CNN preparation

October16, G240. **No GPU query, neural execution, training or learned-policy
evaluation occurred in this allocation.** This is benchmark preparation.
Completed cleanup/pilot sources and their prior evidence are committed at
`837ba6f6`; new opt-in branches leave those captured packets untouched.

## What works now

- All six shared-helper sprite environments can produce the same96-wide by
 192-high RGB observation: player3x3 above chosen gaze3x3, then four geometry
  values. Every policy still learns movement and gaze independently.
- A captured-core builder compiles all six native float32 game targets with
  selectable tile6/Nature2/IMPALA0/Impoola7/Flex4 encoders. Baseline image
  encoders have a common geometry projection before the same recurrent core.
- Native host descriptors register actual weights and activation shapes;
  preparation records their parameter counts, checkpoint lengths and partial
  encoder memory estimates without allocating a CUDA policy.
- A fresh packet allocates30 trainings,8192 decisions/eight checkpoints each,
  seed7.180 initial-raster manifests cover all five families and all36 unique
  game/skin/world combinations. All candidate starting worlds/pixel hashes
  match. These are initial manifests, not540 completed evaluation episodes.
- A CNN-only native PROTEIN recipe exposes18 existing architecture coordinates;
  inherited learner/seed/etc. sweep sections are removed. A separate fixed-
  seven-replica confirmation preparation mode exists, but was not allocated
  or executed here. Sprite all-game aggregate feedback is still future work.

## Verification executed

Twelve actual ASan/UBSan raster/game suites pass: six games times old tile
and new packed layout, each3072 paired steps, three skins, independently
cropped full-frame bytes, one-hot/state parity, resets/gaze timing, sentinel
bounds and hidden-cell mutation. No CPU CNN computation.

Six native binaries and a new packed-input independent float64 CUDA tile
oracle **compile**. The oracle has not run; its future GPU pass would still
not qualify baseline/geometry-adapter math or the full learner.

Seven actual artifact/scalar tests pass: complete packet inspection,
independent all-family parameter counts, missing-job/seed-search/metadata/
checkpoint-schedule rejection, and architecture-only sweep axes. Seven
existing legacy metadata tests also pass175 actual native host registrations/
rejections, preserving old grayscale parameter layouts. Descriptor regressions
are not numerical GPU certification.

Example Key counts with H128/L1, identical movement/gaze heads:

| Encoder | Full policy parameters | Encoder training activation MiB/item |
|---|---:|---:|
|Tile|106,024|1.906|
|Nature|1,458,208|3.096|
|IMPALA|1,348,800|12.624|
|Impoola|173,248|12.589|
|Flex initial shape|372,136|1.199|

Activation estimates exclude complete learner/core/optimizer and concurrency
costs. No batch-fit, SPS, training-speed or quality result follows from them.
Different readout parameter counts are expected on this larger RGB input.

## Preserved attempts and limitations

`build/sprite-packed-20261016` failed compilation: the initial adapter used
the small ordinary Nature activation descriptor with Flex pooling and lacked
its pooling namespace. It did not execute. The fix reuses the existing larger
research Nature descriptors and IMPALA pooling functions, followed by fresh
successful `-v2` and all-game builds. No failed packet was overwritten.

Initial preparation packets and their tool versions are historical; the
current inspected packet is `build/sprite-packed-preparation-20261016-v3`.
The earlier versions are superseded preparation only, not failed training.
`build/sprite-packed-all-20261016` contains actual binaries/source receipts;
`build/sprite-packed-raster-20261016` contains scalar sanitizer runs;
`build/sprite-packed-legacy-metadata-20261016` and its descriptor-checks
directory contain fresh legacy registration receipts. No prior successful
GPU allocation was restarted.

Forage and Lane still have distinct earlier contracts and are not included
in this six-game packet. Existing old tile-format checkpoints are not
silently accepted as new packed-format results. Tile boundaries remain an
explicit architectural prior; camera-image generalization is unproven.

Next: GPU baseline/adapter math checks, short common-batch memory/timing
calibration, repeat/reload/exact-eval canaries, budget calibration and then
the sprite-suite aggregate-feedback/execution supervisor. No long campaign
launcher, push,5090 execution or Nature/Pareto/paper claim is delivered here.
See [the protocol](../../SPRITE_PIXEL_BENCHMARKS.md).
