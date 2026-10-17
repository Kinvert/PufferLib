# Realistic-image foundation — G240 RTX 5060

October14 retention update: with explicit owner approval, the historical raw
`build/image-portability-20261012` tree was deleted after verified compact receipt
preservation. Sources, logs, configurations, hashes and summaries remain; local
arrays/weights/binaries no longer support raw replay. See AGENTS.md for the
archive path/hash. The historical outcomes below remain unchanged.

This is implementation and bounded validation evidence. The new overnight
5090 campaign is **not ready**, and no speed, learning or Pareto advantage is
established by these smokes. The full remaining work is in
[`REALISTIC_IMAGE_SWEEP_PROGRESS.md`](../../REALISTIC_IMAGE_SWEEP_PROGRESS.md).

## Native input contract

`CNN_IMAGE_V1` generates float32 NHWC frames directly from game geometry, with
one grayscale or three RGB channels, and no resizing/preprocessing. Inputs vary
by game: Connect4 H576/W672, Pong H640/W620, Flappy H640/W420, Breakout H330/W576,
Snake/Maze H616/W616. Snake/Maze retain the existing 11x11 policy visibility;
world physics, actions and RNG are unchanged. The version is `human-geometry-v1`.

Nature uses valid convolutions 8/s4, 4/s2, 3/s1. Flex supports first strides
1/2/4/8 and derives its tensors from actual input dimensions. IMPALA/Impoola
derive stage/flatten/GAP/scratch dimensions likewise. Encoder5 is excluded.
Legacy builds retain their original 36x44/channel1 input and contracts.

## Actual checks

- Twelve native host raster probes cover six games and grayscale/RGB, all 41
  drawing IDs, exact repeat/overwrite/range/guards, ASan/UBSan, and 64-step
  reward/terminal/RNG parity against legacy raster builds. Hidden velocity and
  off-crop tile checks pass; all ten occupied-board Connect4 drawings differ.
- Six RGB native trainers compile and expose explicit image metadata.
- Fresh per-game metadata builds register all 60 declared validation layouts
  and storage estimates. Parameter counts match independent scalar arithmetic.
  Twelve deliberate unsafe-index/encoder5 cases reject before GPU allocation.
  Flappy RGB IMPALA/Impoola also construct; this is not baseline CUDA math.
- Independent CUDA checks pass on Flappy H640/W420/RGB, H128, one-stage
  C8/K8/GAP/projection64: stride1 at B1/4; strides2/4/8 at B1/4/16. Total:
  66 process/case instances and 264 native calls. Forward/parameter-gradient
  comparisons, selected finite differences, parameter preservation
  and exact eager/graph/repeat/process bytes use unchanged tolerances. The
  reference executes on CUDA; there is no CPU CNN reference. Other shapes,
  graph families/batches, full heads/core/optimizer and reload remain separate.
- One real Flappy training fixture completes 256 decisions and two finite
  checkpoints at128/256. Four actors, horizon16, minibatch16, seed7, mixed
  appearance seed17. The 59,976-parameter policy is a plumbing candidate.
  Process1.467584817s /174.436 SPS includes startup; it is not sustained SPS.
- The final checkpoint evaluates through the new per-game image layout gate:
  five exactly assigned episodes in four slots, seed7007/drawing0. Graph,
  repeated graph and eager rows match exactly; an independent tail episode
  reproduces the second wave. Four GPU processes plus supervision take
  6.480840181s. Earlier seed420 preparation is preserved and unexecuted.
- Fresh legacy native registration retains v1 and historical H128/L1 Connect4
  counts: quality160,736; Nature138,528; IMPALA270,496; Impoola151,712.
- 52 tests pass, including actual retained-registry relocation/rehashed-tamper
  checks and four actual legacy registrations. Host descriptor/scalar/artifact
  tests are distinguished from CUDA tensor checks.

Hardware: G240 RTX5060, UUID GPU-56b0df25-b4c9-46eb-e498-bc5c71904097,
driver591.86, CUDA compiler12.8.93. Existing process-local runtime/toolkit was
reused. No system/CUDA installation, dataset or global dependency changes.
GPU inference/training uses the shared reservation and competing-job checks.

## Evidence and limits

Compact reports, compiler/source/binary identities, raster traces and descriptor
receipts are under `receipts/`, with `receipts.sha256`. Raw CUDA arrays, native
builds, weights and full execution packets remain in the ignored local
`build/image-portability-20261012/` allocation. Committed receipts alone cannot
rerun the numerical audit without those arrays; do not call them complete
transported evidence or complete vendor/toolchain provenance.

Early missing include/feature-macro, wrong async-section and trace-name errors
remain in that allocation. An intermediate IMPALA build's source check failed
when an unrelated Python test changed during compilation; fresh v3 builds
capture compiled native sources and explicit build/runtime tools separately.
Earlier successful CUDA builds have earlier captured source versions; later
metadata/identity edits do not retroactively qualify a fresh build.

Storage floors are already 1.93–2.89GiB for Nature at B16/four actors across
most games. Flappy GAP is0.656GiB; IMPALA/Impoola3.932/3.598GiB. These exclude
some extra runtime allocations and are **not measured peak usage**. The old
64-actor/M2048 recipes cannot be carried forward without new calibration.

Portable all-game adaptive panels, baseline CUDA math, eight-checkpoint
campaign contracts, fresh confirmation seed roles, cost/coverage objective,
learning/time calibration and clean approximately12h stopping remain unfinished.
No long campaign launched on G240 or the 5090.
