# Spatial autoencoder implementation and RTX 5060 checks

Status: implemented and short local CUDA checks **PASS**. Full target/memory
qualification and the larger 5090 reconstruction comparison are **NOT RUN here**.
Instructions: [NEXT_5090_SPATIAL_AUTOENCODER.md](../../../NEXT_5090_SPATIAL_AUTOENCODER.md).

The encoder takes unchanged 640x480 RGB and emits a smaller **spatial feature
grid**. The decoder trains directly from that grid. Object-balanced reconstruction
and distinguishing-mark errors are central; this runner fits no classifier.
The saved encoder feeds another CNN and the normal recurrent policy. Frozen
prefixes are excluded from the optimizer; fine-tuning is a separate mode.

Four initial choices: stride/channels 8/8, 16/8, 16/16 and 32/16. Three encoder
convolutions (4/4, 3/2, 4/variable stride) avoid an FC bottleneck and blind gaps
at the strongest stride. Native manifests independently match parameter counts
and latent shapes. Version-2 checkpoints and `spatial-compressor-v1` artifacts
remain distinct from the retained vector model/schema. A numeric bank index
exposes complete pretrained sizes to native PROTEIN, with learners/seeds fixed.

## Completed local evidence

Primary immutable packet: `build/spatial-ae-5060-20261021-v10`.
Hardware: NVIDIA GeForce RTX 5060, 8,151 MiB reported; exact UUID/driver in receipts.
This is correctness/integration work, not a clean throughput benchmark.

- 53 host/scalar/artifact tests pass: 12 retained AE, nine spatial, 15 Forage,
  17 compression contracts. These do not compute a CPU neural network.
- Native CUDA build, five-skin annotation host checks and ASan/UBSan pass.
  Retained Flappy/Pong vector-AE native targets also compile in
  `build/spatial-ae-legacy-20261021`.
- All four spatial sizes pass independent float64 CUDA output/all-parameter/
  image-gradient checks at 160x192/B1, including 48 selected finite differences
  across the four cases and three exact eager/graph repetitions per case.
- A separate fresh bounded allocation passes all four at 161x193/B2 and
  480x640/B1. The retained 160x192/B1 vector model also passes its independent
  oracle and 16 selected finite differences. These are smooth-positive fixtures,
  not arbitrary-ReLU-branch/full-learner mathematical proof.
- Both frozen and fine-tuned consumers pass independent CUDA forward/all consumer
  parameter-gradient checks on their actual imported features; fine-tuned mode
  also checks all consumer input derivatives. Eight consumer cases pass selected
  finite differences and exact eager/graph repeats. Incorrect spatial-shape
  exports fail before imported weight mutation.
- Four-update object-loss AE trainings match every scheduled checkpoint across
  fresh repeat and optimizer/data-order resume for each size. Plain controls
  have identical initial full-state bytes. Twenty tiny AE training processes
  retain 52 finite states; these are canaries, not quality experiments.
- Eight reconstructions (four sizes x two objectives) use the same six-frame
  fixture. Native image/annotation alignment and independently audited exact
  frame/cell quotas pass. Every reloaded exported feature grid is byte-identical
  to its training model's feature grid. Metrics preserve class/split/skin and
  distinguish object/background/marking errors.
- Four real frozen-policy canaries and one fine-tuned canary train 8,192
  decisions each; each reloads its saved policy and performs one additional
  update. Native PROTEIN adds three actual 8,192-decision trainings; bank IDs
  were `[1,1,0]`. Duplicates remain. This is random warmup, not GP-active evidence.
  All fixed training seeds remain 7. No meaningful task-quality result follows.
- The primary packet records 60 supervised process receipts, including three
  host-only collectors and a native sweep parent with three training children.
  The separate additional-math packet records nine CUDA workers. The successful
  stage checksums bind all retained inputs/results, including raw artifacts.
- The actual HTML report was generated: full frames plus fixed first-instance
  object crops, 240 browser-readable BMPs. No arrays/rasters/weights are committed.

Open the local canary viewer:
`build/spatial-ae-5060-20261021-v10/spatial-smoke/reconstructions.html`.
Its four-update pictures are workflow checks, not a pretrained-backbone result.

## Preserved failures and earlier scopes

Earlier directories remain unchanged. v1 failed its recipe time-cap check before
building. v2/v3 exposed parameter re-registration erasing live device pointers
in the test/inspection setup. v4/v5 exposed a shadowed decoder-layer count in the
independent reference, copying only 128 latent derivatives for a spatial model.
The fix handles the actual spatial extent; tolerances were not relaxed. v6 passed
the first-size bounded smoke. v7 completed all size/consumer checks but native
PROTEIN refused a missing `prune_pareto` config field. Required fields and a host
regression were added. v8 is preparation only. v9 passed four-size/native-sweep
smoke. v10 adds paired plain reconstruction and actual report/viewer generation.
No failed/completed allocation was restarted in place.

Compact checksums, native results and representative failures are under
`receipts/`; full frozen sources/logs/arrays remain in their ignored build packets.
The captured local source contains the pre-existing uncommitted
`PUFFER_WEIGHTS_MIRROR` hook in `src/pufferl.cu`; these workers explicitly unset
that environment variable. That unrelated hook is excluded from this delivery.
Target preparation/qualification captures the 5090's own actual sources.

## Remaining gates and claims

No new local memory-instrumentation acceptance, full 5090 pretraining, robust
multi-seed downstream evaluation, encoder transfer, Pareto or SOTA claim. Old
WDDM/compression failures remain failures. The 5090 must pass its new full
qualification before the larger pilot. Keep all old vector results separate.
No task process/reservation remains after these completed local allocations;
do not automatically restart or extend them.
