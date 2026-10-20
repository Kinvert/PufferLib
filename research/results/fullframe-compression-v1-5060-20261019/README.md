# Full-frame compression v1: implementation and local qualification

October 20 terminal update: v6 completed all 120 numerical workers and 144 GPU
finite-difference probes, then stopped at the first Compute Sanitizer suite.
The tool returned 99: WDDM debugger initialization failed/device unsupported,
with two tool errors. This is not memory acceptance. No planned training,
import or evaluation smoke ran. Preserve the failed allocation; do not restart
it or change the system/CUDA stack. Read
[current status](../../FULLFRAME_COMPRESSION_PROGRESS.md). Historical progress
notes below describe the preceding implementation and diagnostic stages.

The native pipelines take full RGB Flappy640H×420W and Pong640H×620W. The
source rasterizers/game rules are unchanged. Pipeline0 learns a narrow strided
stem and adaptive4x4readout; pipeline1 uses area4 plus a small CNN; pipeline2
uses area4 plus resized RGB Nature/H128. All share MinGRU H128/L1 and matched
per-game learners. No dataset/pretraining, sprite/atlas shortcut or system/CUDA
change is included.

## Completed host evidence

- All three pipelines build for both games, plus the independent CUDA harness.
- Independent encoder counts match native registration:36,776 /36,608 /
  1,255,712 on Flappy and36,776 /36,608 /2,042,144 on Pong.
- Registered storage floors:Flappy1.806/1.770/1.681GiB and
  Pong2.663/2.610/2.490GiB for models0/1/2. These are rejection floors, not
  measured peak memory guarantees.
- Seventeen new host artifact/config/process tests pass; eleven existing image
  input/memory tests pass. No CPU neural model was executed.
- A legacy full-frame trainer builds without the new research flag. Native
  Nature metadata is identical with and without that flag.
- Two ASan/UBSan full-resolution raster suites pass64world steps across all
  seven drawings each. Final frame FNV64:Flappy6f0e01dc273e0d85,
  Pongf1822d7ce4429c05. Raster probes do not qualify neural math.

## Preserved numerical failure and its diagnosis

`build/fullframe-compression-v1-5060-20261019` is preparation-only. Before any
GPU launch, stronger gradient fixtures/finite differences were introduced in
a fresh packet; the old preparation was not relabeled as executed.

`build/fullframe-compression-v1-5060-20261019-v2` **failed** after40passing
cases/144GPU finite-difference probes. Failure:161×193/B1/model0/signed,
comparing parameter gradients through the full float64 network. The original
failure and source snapshot remain unchanged.

A separate bounded CUDA diagnostic reproduced exactly one first-layer ReLU
branch difference at output(y5,x25,channel1):native1.4901161193847656e-8,
float64ReLU0. Layer forward maximum errors0–3 were
2.83e-7/8.63e-8/5.73e-8/3.60e-8. Independent backward on actual native
branches agrees to1.0464933225540207e-8. This is diagnostic evidence, not a
retroactive pass for the failed packet.

Fresh `v3` introduced declared numerical protocol`ffc-native-branches-v2`:every
layer forward is checked; all outputs/parameter/input gradients must pass the
independent actual-branch reference at the original tolerances. Full-double
gradients also must pass when branches agree. Signed branch differences must
stay in a2e-6 post-ReLU neighborhood; positive/zero cases must have none.
The differing full-double gradients and branch counts remain recorded. This
tests the derivative of the executed native function, rather than attributing
a derivative from the other side of zero to it. No tolerance was widened.

Shared GPU occupancy was respected:the first diagnostic launch found another
task training soccer and refused execution. A bounded queue later ran the
single diagnostic when idle. No foreign process was interrupted.

`v3` was then interrupted after its first worker because a foreign campaign
started its next training job. No case was qualified into its progress table;
its failed stage and successful-but-unqualified worker remain. The fresh `v4`
packet also checks actor/train forward equality and runs three device-memory
sanitizer cases. Its bounded queue waits for the specific foreign supervisor
process to exit, followed by stable GPU idle, before starting qualification.

That v4 queue was canceled before any CUDA launch to align the harness with
the trainer's actual `-O2` and default FMA compiler settings. `v5` is the current
prepared packet with matched compiler settings. A new external Admiral campaign
was already active at its first GPU check, so no v5 math stage started there.
That unstarted v5 queue was canceled:the previous busy refusal had created an
empty math directory before admission. Directory creation now follows admission;
a host regression test checks the busy path. No v5 CUDA stage ran.

Fresh v6 adds RGB ramps, translated edges/dots, clipping and zero-input fixtures.
Its live queue waits for the observed foreign campaign owner to exit, then
120seconds continuously idle, with a1hour waiting cap. It runs the declared
math→smoke→offline-report chain once. A later foreign process fails the active
stage. Check actual process/queue state before any new launch. Offline reporting
now rechecks exact assignment coverage, process receipts, repeat checkpoint hashes
and graph/eager/slot comparisons. Native binned timing series are retained
separately from monotonic processSPS; bin means cannot establish native-averageSPS.

The executed smoke supervisor is captured per stage. Fixed `[sweep]` logging
keys remain because native `train` reads metric/downsample there; all inherited
`[sweep.*]` search axes are stripped. No optimizer sweep is launched.

## Remaining first-increment gates

The fresh complete math panel,12paired trainings,24exact evaluations,
18valid/rejected import checks, offline evidence audit and final5090handoff
status must be verified from terminal receipts before this report is complete.
The smoke is8Kdecisions/policy, not a quality/generalization/frontier result.

## Local artifacts

- `build/fullframe-compression-v1-5060-20261019-v6`:current qualification packet.
- `build/fullframe-compression-v1-5060-20261019-v5`:prepared; canceled before CUDA.
- `build/fullframe-compression-v1-5060-20261019-v4`:prepared; canceled before CUDA.
- `build/fullframe-compression-v1-5060-20261019-v3`:foreign-job interruption.
- `build/fullframe-compression-v1-5060-20261019-v2`:failed original numerical packet.
- `build/fullframe-compression-development/branch-diagnostic*`:retained diagnostic
  arrays, frozen test inputs, source hashes, process receipt and layer findings.
- `build/fullframe-compression-raster-20261019`:two sanitizer rasters/physics receipts.
- `build/fullframe-compression-legacy-trainer-20261019`:normal baseline build.
- `build/fullframe-compression-development/legacy-nature-comparison.json`:
  with/without-flag native metadata equality.

Raw arrays/weights remain local under ignored build. Final compact evidence
will be copied alongside this report after acceptance. No commit/push/5090
execution is claimed here.

Compact **host-only** receipts are now in `receipts/host-v6/INDEX.json`:
34files/116,284bytes, including source/build manifests, six registered layouts,
six fixed recipes, host tests/readers, raster receipts and the legacy metadata
comparison. Their presence does not qualify any still-pending GPU gate.
The later `receipts/host-limits-v6/` retains the14-test deadline/allocation
follow-up, its reader and the explicitly incomplete preacceptance report.
`receipts/host-import-v6/` retains the final15-test reader/export audit follow-up.
Final reporting rechecks current export hashes and every imported encoder versus
core/head tensor at its registered offset, as well as the native process receipt.
Optional encoder timing now has two successful GPU-free native builds against
the same frozen sources; GPU timing remains unexecuted. Its17-test host follow-up
and captured timing/build inputs are in `receipts/host-timing-v6/`.
