# Realistic-image sweep implementation

October 12, 2026. Active implementation in `/home/claude/cnn-audit-20261008`;
the retained `/home/claude/cnn` checkout and prior campaign are untouched.
This is temporary Kinvert research, not the eventual upstream PR.

## October12 curve/seed/stopping implementation

October 13 follow-up: controller tests now cover clean stopping both before the
first trial and after two complete panels. They verify the final native replay,
absence of a partially launched next trial, and rejection of altered admission
receipts. These are mocked orchestration checks, not GPU runtime evidence.

The separate Pong loss diagnostic identified cancellation in the test oracle's
large double-precision dense reduction. Compensated summation is now confined
to the independent oracle's loss-forward path; production encoder kernels,
probe perturbations and acceptance tolerances are unchanged. At the original
perturbation, the diagnostic error falls from `1.56889e-5` to `2.85371e-8`.
The historical failed allocation remains failed. A fresh full Pong allocation
against the new, explicitly versioned loss contract now **passes** RGB640x620,
B1/4/16: 18 cases, 72 calls and six independent worker processes. This is encoder
numerical evidence; it does not qualify all other shapes or establish learning.

`image_validation_collection.py` can audit and combine separately completed
shape allocations from one frozen catalog. It refuses missing shapes, failed
checks, changed configurations/bundles and mixed hardware. It executes no CUDA
or CPU neural reference and does not restart any allocation.

The Flappy RGB640x420 initial-save path now executes successfully. Two fresh
native runs each save finite full-policy weights at 0/64/128/192/256/320/384/
448/512 decisions, with byte-identical SHA256 values at every matching stage.
The policy has 59,976 parameters; process times are 2.019283315s and 2.019242834s.
These startup-dominated 512-decision checks do not establish sustained speed or
learning. Matching B1/4/16 CUDA checks pass 18 cases/72 calls beforehand.

`build_image_delivery.sh` compiles and registers all six default RGB native
trainers and their host metadata using the existing uv Python 3.12 venv. Actual
six-game compilation and host registration pass. The first launcher attempt
incorrectly resolved the venv interpreter symlink, losing its packages; a second
metadata import lacked the shared `claim` module path. Both failures are retained.
The launcher now preserves the venv path and imports metadata through the normal
panel module. Completed native builds were reused for fresh host registrations;
no failed or completed GPU allocation was restarted.

Image panel-v5 now declares the same policy's initial checkpoint plus exactly
eight trained checkpoints. The research-only native save option is disabled
for ordinary/legacy training. Its full shared-panel execution subsequently passes
as recorded below. Explicit episode-seed
lists and a frozen development/confirmation
role document are checked during preparation and inspection; confirmation
cannot enter curve search.

Feedback-v4 now calls the balanced eight-Nature-time-anchor objective and
reconstructs it during its offline native receipt audit. It requires a measured,
audited Nature panel, fixed normalization/role inputs and identical per-game
pixels, learners, budgets, worlds, core, episode assignments and native sources.
The new runtime-pilot tool measures complete trials and reporting; the overnight
branch requires that proof, reserves a whole next panel and separates
`budget_stop` from failure. These integrations have scalar and mocked controller
tests, not a completed real new-version learning campaign.

Nature's original full-image harness initialization failed its selector guard;
that test-only wiring is fixed. Subsequent fixed-tolerance numerical comparisons
reached all cases but hit export/hashing deadlines. All failed allocations are
retained. Optional exact-byte hardlinks reduce duplicate export storage, and a
declared per-batch partition retains two fresh processes per case and all four
native modes. The resulting Connect4 RGB576x672/B1/4/16 Nature report passes18
instances/72 calls. Its following Pong shape fails a strict float64-oracle
finite-difference probe (relative error3.09765632e-5 versus the unchanged2e-5
tolerance). Thus **that original multishape allocation remains failed**. A separate
bounded CUDA loss-sensitivity diagnostic is prepared/executed under its own
receipts; it cannot qualify the failed allocation or change tolerances.

Next: establish useful matched image learners, qualify the actual chosen control
graphs and prepare the calibrated5090 launcher. No long local
campaign or final push/handoff has been made.

The zero-step Flappy checkpoint also passes exact assigned-episode graph,
repeat, eager and independent-tail comparisons: five episodes/four slots,
5.73002936s supervision window. This verifies loading/evaluating the measured
initial policy; it is not full-policy numerical or learning evidence.

The new all-game image panel-v5 now **passes** its bounded 5060 execution and
offline receipt audit: six jobs, 54 finite checkpoints, 369/369 drawing/checkpoint
evaluations (including 41 initial-policy evaluations), and 12 exact repeat/eager
checks. Each job trains 512 decisions with one assigned episode per drawing.
The campaign takes 599.80070706s; training processes total 13.530211734s.
This is checkpoint/evaluator plumbing, not learning or frontier evidence.

The curve-feedback learning gate is now implemented and scalar-tested. It
recomputes fixed final-versus-initial improvements for every development seed
and refuses an unlearned game, including a rehashed counterfeit gate receipt.
Actual eligible full-image learning controls remain outstanding. Reporting
reserves now scale with discovery history and slower observed optimizer calls,
with matching admission/audit tests. The final overnight handoff remains unfinished.

`report_image_feedback.py` re-audits image panels/curve-feedback before exporting
process SPS, all monotonic checkpoint intervals, raw native SPS/uptime bins,
selected affordable anchors and per-panel interactive curves. It retains
zero-step startup cost without inventing initial training SPS and flags rather
than alters native uptime anomalies. Its actual GPU-free report of the completed
5060 panel passes (six jobs/54 checkpoints/110 native bins). Three export tests
and nine related budget/reserve checks pass. Runtime-pilot calibration now
measures this full report, including exports/HTML, rather than only the audit.
The campaign report route still requires a real eligible feedback-v4 pilot;
no learning or frontier qualification follows from the successful smoke report.

`prepare_image_calibration.py` adds GPU-free paired-control preparation and
offline qualification. It derives budgets/learners from explicit curve-v4
inputs, requires all fixed development seeds and both all-shape numerical
manifests, then freezes quality/Nature initial+eight-checkpoint panels. Its
qualification path preserves a complete negative report and refuses to create
Nature anchors for an unlearned game. Four scalar/mocked assignment, manifest
and negative-qualification checks pass; actual native preparation and full
control qualification remain pending. The final focused host/scalar/mock suite
passes 71 tests; its initial unit fixture used inconsistent checkpoint units,
and both that failed test log and the corrected-fixture pass are retained.

## October12 shared pipeline follow-up

Additional October13 Nature numerical allocations pass RGB576x672,640x420 and
330x576/B1/4/16 (54 cases/216 calls). With retained Pong, four distinct shapes
pass from the same catalog. The900s follow-up stops before616x616 starts and
remains incomplete. A separate fresh600s allocation checks only that previously
unstarted square shape; it does not repeat a completed case or restart the
failed batch. That square allocation now passes18 cases/72 calls. Thus all five
Nature shapes individually pass90 cases/360 calls under the same catalog/config.
The separate combined index now passes its offline raw-array audit,90 cases/
360 calls, with matching catalog/configuration/device receipts. Nature and the
C8/GAP control both have this five-shape encoder acceptance atH128/B1/4/16;
other control graphs, full-policy math and useful learning remain separate gates.

The full-RGB all-six-game/all41-drawing aggregate-to-native-PROTEIN loop now
actually passes three proposals:18 short trainings/123 exact evaluations/
36 repeat-eager checks, with270 independent CUDA process/case instances/
1080 calls on all five image shapes. Final native replay observes all three,
including a preserved rounded duplicate. The complete offline raw receipt audit
passes. The loop takes600.494s; this is64-decision plumbing, not learning.

The earlier eight-checkpoint packet is **failed**, not superseded as a pass:
six trainings/48 finite checkpoints/280 of328 evaluations/ten repeat checks,
then Maze's30s evaluation timeout. A separate five-episode final-Maze
graph/repeat/eager probe passes exact rows at36.6–36.8s/process. No horizons or
assigned episodes were shortened in that diagnostic. Numerical workers now
share the campaign deadline and native layout/storage admission precedes CUDA.

Read[the retained report](results/realistic-image-panel-5060-20261012/README.md),
[pipeline contracts](REALISTIC_IMAGE_PIPELINE.md) and[overnight specification](IMAGE_OVERNIGHT_PROTOCOL.md).
At that earlier checkpoint the curve/seed/admission work was still unimplemented.
Its later code/testing status is recorded above. Meaningful new-image
Nature/learner calibration, real new-version acceptance and a verified5090
handoff remain outstanding. No long campaign has launched here.

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

1. Shared portable build/metadata/checkpoint/suite bindings now execute, including
   pre-CUDA native memory/index rejection and complete all-game feedback. Qualify
   needed baseline image math and calibrated batches; don't route new images
   through old36x44 metadata tools.
2. Extend the versioned main campaign to roughly eight scheduled checkpoints;
   retain old bounded smoke contracts and historical receipts.
3. Freeze disjoint development/confirmation training and episode seed lists;
   never sweep seeds. Choose realistic-image per-game learner/vector geometry
   with memory and learning calibration, identically across architectures.
4. The bounded full-RGB5060 all-game train/checkpoint/exact-eval/repeat-eager/
   aggregate-to-PROTEIN smoke now passes. Qualify additional needed new-image
   graphs/batches, targeted spatial controls and Nature comparisons.
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
