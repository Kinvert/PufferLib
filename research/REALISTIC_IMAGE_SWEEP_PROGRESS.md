# Realistic-image sweep implementation

October 12, 2026. Active implementation in `/home/claude/cnn-audit-20261008`;
the retained `/home/claude/cnn` checkout and prior campaign are untouched.
This is temporary Kinvert research, not the eventual upstream PR.

The requested end state is a calibrated overnight 5090 architecture search,
matched per-game learners and fixed seeds, about eight checkpoint observations,
targeted controls, native PROTEIN feedback across all six games, complete
reporting and a simple committed handoff. **That end state is not ready yet.**

## Implemented foundation

- `CNN_IMAGE_V1` selects direct geometry rasterization at explicit compile-time
  height/width and either one grayscale or three RGB channels. Default builds
  retain the historical float32 36x44 input. Layout is NHWC; there is no
  preprocessing/downsampling. Grayscale uses an explicit semantic gray palette,
  not a luminance conversion of the RGB palette.
- Practical input sizes (height x width): Connect4 576x672, Pong 640x620,
  Flappy 640x420, Breakout 330x576, Snake/Maze 616x616. Snake/Maze retain their
  11x11 local policy field of view. World geometry, RNG and actions are unchanged.
- Nature retains its valid 8/s4, 4/s2, 3/s1 convolutions. Flex derives all shapes
  from the input and supports first strides 1/2/4/8. IMPALA/Impoola derive stage,
  flatten/GAP and scratch sizes from the input. Encoder 0/tiny, 1 and 5 remain
  excluded from this image contract; IMPALA/Impoola use their explicit build
  families and selector 0.
- Native `eval_exact_info` exposes the image contract. Connect4 image suites
  have a new version binding the actual nonzero empty-grid hash; legacy empty
  boards/suites keep their original contract.
- `policy_metadata --storage FULL.ini` registers the actual policy tensors
  without CUDA allocation. It reports a storage rejection floor, not measured
  peak VRAM. Fresh native host checks now accept the declared layouts and
  reject batches exceeding 32-bit indexing before activation registration.
- `native-image-metadata-registry-v1` freezes a separate actual native metadata
  target for each game's input and optional baseline family. New descriptors
  use v2 and bind environment, shape, layout, dtype and preprocessing; old
  registries/descriptors retain v1 and strict 36x44 validation. The exact
  evaluator acceptance path can freeze and replay these per-game image tools.
- `prepare_image_inputs.py` freezes 60 short validation configs, including
  quality/Nature, C8/C16 x residual off/on, and stride probes. Learners and seed7
  are paired within each game. These adapted B16/H16/four-actor configs are
  **uncalibrated plumbing fixtures**, not the overnight learners or budgets.

## Executed checks and receipts

All development artifacts are under ignored
`build/image-portability-20261012/`. Failures and earlier source versions stay
there. The first raster compile lacked `-I.`; Breakout needed feature-test
macros; one early metadata binary still read async from `[train]`; one trace
comparison used the wrong Flappy output filename. These were fixed with fresh
outputs, without changing the CUDA stack or historical evidence.

The final twelve host raster probes (six games x grayscale/RGB) pass
ASan/UBSan, complete overwrite, guard values, range/finiteness, exact repeats,
no environment-struct/RNG mutation and 64-step reward/terminal/RNG trace parity
with legacy raster builds. New probes separately check hidden velocity and
off-crop tile invariance, and ten distinct occupied-board Connect4 drawings.
These are game/raster checks; no CPU CNN executes.

Six normal RGB trainers compile; `*-image-info.json` records the actual compiled
contracts. The fresh `*-metadata-v3` builds include the index guards and v2
descriptors. `metadata-verification-v3/REPORT.json` records 60 actual native
registration/storage cases, each matching independent scalar parameter counts,
and 12 deliberate rejections (oversized batches and encoder5 on every game).
Two additional Flappy RGB host builds construct IMPALA and Impoola. No policy
tensor math/GPU allocation runs in these registration checks.

Actual RTX5060 independent-CUDA checks on 640x420 RGB, H128 and the one-stage
C8/K8/GAP/projection64 graphs:

| First stride | Batches | Process/case instances | Native calls |
|---|---|---:|---:|
| 1 | 1,4 | 12 | 48 |
| 2 | 1,4,16 | 18 | 72 |
| 4 | 1,4,16 | 18 | 72 |
| 8 | 1,4,16 | 18 | 72 |

All 66 instances/264 calls pass exact eager/graph/repeat/process bytes,
parameter preservation, layer-forward and backward comparisons against the
direct float64 CUDA oracle on actual float32 branches, plus smooth-graph and
selected GPU finite-difference checks. Tolerances are unchanged. These use
`native-cnn-image-graph-check-v2`; legacy v1 batches remain 1/64/2048. Stride1
B16, other shapes/channels/graphs, Nature/IMPALA math and full-policy numerical
qualification are not established by these results. Captured source versions
precede later raster/identity edits; bind future runs to fresh build receipts.

`flappy-train-smoke/REPORT.json` records one real native run after matching
stride4/B4/B16 math: 256 decisions, training seed7, two finite checkpoints
at128/256, 59,976 full-policy parameters, 1.467584817s process time. Process SPS
174.436 is a startup-dominated plumbing measurement, not sustained throughput
or a Pareto/learning result.

`flappy-evaluator-execution-v2/REPORT.json` now records actual native inference
of that final checkpoint through the new per-game host layout gate, using the
newer source-matched full-RGB trainer. Graph/repeat/eager each evaluate five
assigned episodes with four slots; a separately initialized one-episode tail
matches the original second wave. All exact per-episode rows agree. The four
native processes plus supervision take 6.480840181s. Training seed7 and eval
seed7007 remain development-only. Earlier seed420 preparation is preserved
and unexecuted. This is one Flappy drawing/checkpoint/batch, not learning,
full-policy numerical or all-game acceptance.

52 current host/configuration/artifact tests pass, including scalar shape,
paired-learner/stride, versioned image/checkpoint and visible-empty-board checks,
plus historical continuation/evaluator regressions. They are separate from
actual CUDA and native host construction evidence. The explicit retained
registry tests also reject rehashed image/compile and binary-provenance edits
and inspect relocation without native execution. Fresh legacy host tools
retain the original v1 descriptors and quality/Nature/IMPALA/Impoola counts
160736/138528/270496/151712; they perform no neural math.

Compact evidence is committed alongside
`research/results/realistic-images-5060-20261012/README.md` and its checksummed
receipts. Full raw numerical arrays/checkpoints/builds remain local; the
compact receipts are not a complete transported numerical evidence archive.

Storage floors in GiB for the four-actor/H16/B16 RGB fixtures:

| Game | Nature | Locked quality | C8/K8/s4/GAP |
|---|---:|---:|---:|
| Connect4 | 2.805 | 1.403 | 0.944 |
| Pong | 2.890 | 1.438 | 0.968 |
| Flappy | 1.934 | 0.974 | 0.656 |
| Breakout | 1.331 | 0.692 | 0.465 |
| Snake | 2.748 | 1.375 | 0.926 |
| Maze | 2.748 | 1.375 | 0.926 |

These are rejection floors, not measured peak usage or speed evidence. Flappy
IMPALA/Impoola floors are 3.932/3.598GiB at the same geometry. Large-input
activation and rollout storage remain substantial even when GAP makes the
readout small. Do not reuse the old M2048/64-actor recipe without calibration.

## Remaining work before the 5090 handoff

1. Complete portable per-game build/metadata/checkpoint/suite bindings and
   memory/index rejection tests. Confirm native baseline construction with
   fresh builds. Avoid sending new images through old 36x44 metadata gates.
2. Extend the versioned main campaign to roughly eight scheduled checkpoints;
   retain old bounded smoke contracts and historical receipts.
3. Freeze disjoint development/confirmation training and episode seed lists;
   never sweep seeds. Choose realistic-image per-game learner/vector geometry
   with memory and learning calibration, identically across architectures.
4. Qualify needed new-image graphs/batches; run short reserved 5060 all-game
   train/checkpoint/exact-eval/repeat-eager/aggregate-to-native-PROTEIN smokes.
   No competing GPU processes, CPU CNN reference, long G240 training or retries
   of completed allocations.
5. Implement the predeclared curve/coverage objective and targeted controls
   through the same all-game runner. Keep every decline, duplicate and failure.
   Use a fresh campaign; don't relabel/replay the 45 tiny-image observations.
6. Calibrate the total approximately12-hour 5090 allocation, including
   validation/evaluation/reporting and a between-complete-panels stopping reserve.
   Selectively profile real bottlenecks; no claimed speed from scalar MAC counts.
7. Write the new simple 5090 setup/build/canary/calibration/run/report handoff,
   commit/push only to Kinvert/PufferLib `cnn-research`, and give Kinvert a short
   message to relay. Don't launch the overnight campaign here.

The first independent CUDA bundles have large raw arrays; preserve them locally
and commit only compact audited receipts/source identities with their checksums.
Do not mark the active goal complete from this foundation alone.
