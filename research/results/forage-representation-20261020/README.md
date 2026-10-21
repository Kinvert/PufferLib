# Forage perception pilot: implementation and scoped verification

October 20, 2026, G240. [Implementation](../../FORAGE_REPRESENTATION_PILOT.md),
[5090 commands](../../../NEXT_5090_FORAGE_REPRESENTATION.md).

Implemented native CUDA full-frame plain/object-balanced autoencoder training,
object and distinguishing-mark reconstruction metrics, fixed-capacity frozen
grid-identity probes, constant/random controls, exact frame/target/artifact audits
and a bounded comparison runner. This is not an RL/Pareto result.

## Executed build/host checks

- Latest Forage packet: `build/forage-representation-20261020-v5`.
  Native CUDA compilation, scalar encoder/decoder registration and template checks
  pass. The ordinary Flappy/Pong autoencoder targets also compile in
  `build/autoencoder-legacy-recognition-20261020`.
- ASan/UBSan passes 60 complete native Forage observations: five skins, twelve
  reset/step observations per skin, exact visible label/template/pixel alignment
  and appearance-independent player/world RNG traces. Lookalike templates differ
  in every skin. No neural computation occurs in this host test.
- Fifteen new host data/metric/receipt checks, twelve existing autoencoder checks
  and seventeen compression regressions pass: **44 checks**. These cover bad
  background-dominated summaries, absent classes, target/frame quotas, changed
  commands/features/labels and test-split exclusion from pretraining.
- Two fresh tiny collector runs each write twelve actual full RGB frames from
  three fixed worlds (train/development/test), skins 0 and 4. Their entire dataset
  manifests have the same SHA256:
  `40a7a00765e5ec04a001291163011f92f36d9ae7a41155772cad8e43e17b09ed`.
  This binds every image/label hash, action seed, world seed and episode result.
  Skin-paired labels/terminal traces match. Each fixture is about 43 MiB; this is
  collection QA, not the default roughly 1.5 GB pilot corpus.
- The initial v1 build failure (duplicate dynamic observation-size definition)
  remains in its ignored packet. A research-only adapter fixes it; the Forage
  environment and normal core sources are unchanged by this task. v2–v4 are
  preserved intermediate GPU-free preparations, not old scientific passes.

## Completed bounded RTX 5060 checks

The primitive, integration, three-case network and export/import smokes are COMPLETE.
The first waited for 120 seconds of stable idle; no foreign campaign supervisor
was running. Thirteen serial native GPU processes completed, with no task GPU
reservation remaining. Do not restart these allocations or turn them into a
local full pilot. Hardware: RTX 5060, driver 610.88; raw stage/process receipts
remain in v5, with compact copies here.

- New probe-head gradients match the independent float64 CUDA reference:
  maximum error `1.3893e-8`; all eight selected finite differences pass.
  Object/background weight normalization, weighted loss derivatives, inactive
  validation tail and train-only feature normalization pass. All twenty
  identity/background/deletion/swapped-apple metric fixtures pass across five skins.
- Three independent encoder/decoder math cases pass at batch 1:
  160x192, 161x193, and actual 480x640. They compare all 71,763 parameter
  gradients, outputs and input gradients, plus unchanged weights and exact
  eager/graph repeat bytes. Sixteen selected network finite differences pass;
  combined with the head tests there are **24 selected probes**. This is the
  declared smooth-positive fixture scope and fixed mixed tolerance, not arbitrary
  branches, batch 2, every graph or full-learner acceptance.
- Actual native plain and object-balanced training each complete **two updates**
  on the tiny QA frames. Both objectives decrease; initial full state bytes match.
  Four distinct **four-update** head fits run for constant, random, plain and
  object-balanced features. Checkpoint parameters loaded exactly, every frame/
  target/prediction row passes the audit, and encoder/decoder bytes remain frozen.
  A fifth fit reloads the same balanced checkpoint and matches raw features,
  normalized statistics, targets, head weights and predictions byte-for-byte.
  These tiny canaries are integration evidence, not item-recognition results.
- Encoder-only exports from both actual two-update states load through the native
  Forage research policy path. Encoder bytes match exactly, core/head bytes and
  initialization RNG remain unchanged. Fresh export fixtures preserve the
  completed native training receipts; completed canary folders were not edited.

Full new GPU qualification and the default pretraining/probe comparison have **not**
run. The prepared 5090 allocation still requires independent math/selected finite
differences, exact short training/reload/resume repeats and four memory suites.
No unsupported WDDM tool, prior failure or completed allocation is waived/retried.
Neither successful collection nor declining reconstruction MSE proves useful
perception. No claimed item-recognition score, RL improvement or SOTA result is
established by these build/host receipts.

Compact receipts are included here. Raw fixtures, source snapshots and binaries
stay under ignored build. Receipt driver copies preserve the completed commands;
they are not new launch instructions. No Torch/CUDA/system/dependency changes
were made.
