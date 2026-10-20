# Full-frame compression continuation notes

October20 update:the v6 queue is TERMINAL, unified-exec session89498 exited1.
All120numerical workers passed, including144GPU finite-difference probes and
paired fresh-process checks. The first Compute Sanitizer run then exited99:
WDDM debugger interface initialization failed and the tool reported unsupported
device. Its native numerical worker passed, but ERROR SUMMARY reports2errors;
this is NOT a memory-check pass. No training/evaluation/import smoke started.
Read v6 stages/math.failure.json and logs/math/memcheck-m0.log/process.json.
The stale queue.json says executing; actual terminal receipts take precedence.
Do not restart this failed allocation or change Windows/CUDA/system settings.
The implementation/5090handoff remains uncommitted and unpublished. Full GPU
acceptance is incomplete; remaining gates need a working memory-check environment
and the declared training/evaluation/import packet. Prior waiting notes follow
as history, not an instruction to launch another queue.

October19,2026. Required runtime gates were incomplete.

Latest goal audit:blocked on exclusive GPU availability after the same external
occupancy persisted across at least three goal turns. All implemented host/native
build work is retained,17host tests pass, and required GPU gates remain pending.
The already-live bounded queue is preserved, not duplicated or canceled. It may
finish or fail while the goal is blocked; inspect session89498/PID882705 and
terminal stage receipts before any action when resuming. Snapshot of verified
foreign GPU ownership and queue liveness:the v6 packet's external-gpu-wait.json.
After the existing queue succeeds, run the current offline report before claiming
acceptance; its loaded reader predates the last audit additions. Optional GPU
timing is prepared but unexecuted and not part of that live queue.

## Current execution

Current packet:`build/fullframe-compression-v1-5060-20261019-v6`.
Native builds, registration and host-check completed. The bounded queue is
waiting for the observed foreign campaign owner877672 to exit and120seconds
stable GPU idle before math, then smoke, then report.
Waiting cap1hour; actual math/smoke caps900seconds each. Never restart it just
because a status file is quiet. Verify the actual process/session first.

In this Codex thread the unified-exec queue session is **89498**. Its console is
redirected to `build/fullframe-compression-development/validation-v6-queued.log`.
Check `queue.json`, `stages/*.started.json`, terminal results/failures and actual
process state. A prior v5 busy refusal occurred before stage admission and is
preserved in `math-v5.log`; it did not start CUDA or write a math-start marker.

Shared GPU is in use by other research work, including selfplay campaigns.
Do not kill or pause those processes or use their between-job gaps. This task
does not require new permissions. All CUDA/system/dependency settings are intact.

## Implemented

- Native encoder8, macro-gated in src/ocean.cu. Three pipelines per the handoff.
- Private runtime layer descriptors keep raw source pixels full resolution.
- Encoder JSON/binary import validates graph/shapes/hash/finiteness before import;
  consumes ordinary initialization RNG before replacing only encoder tensors.
- Native host metadata, memory registration, captured-source builds.
- Independent direct-sum CUDA reference, actual native-branch backward, all
  parameter/input gradients, layer forwards, actor/train equality, graph/process
  repeats, GPU finite differences and memcheck packet.
- Supervisor for12paired8192decision trainings,24exact evals and18import checks.
- Export slices actual registered encoder tensors from bound full checkpoints.
- Offline report and 5090 instructions. They remain qualification-pending.

Native source/model files in this checkout match the compiled v6 snapshot.
The runtime supervisor records its actual loaded source per stage. Preserve
the unrelated11-line checkpoint mirror in src/pufferl.cu; comparisons unset it.
Seventeen new host tests and eleven existing image/memory tests pass. Full-size
ASan/UBSan raster suites and a normal legacy build/metadata comparison pass.

## Preserved failures and preparations

- v1:preparation only; strengthened fixtures before any GPU execution.
- v2:40passing math workers, then a signed odd-shape full-double gradient failure.
- Diagnostic:one ReLU branch difference at1.49e-8; actual-branch gradients agree
  to1.05e-8. Raw arrays/logs/source snapshots retained in development directory.
- v3:first worker succeeded but foreign GPU work appeared; no accepted progress.
- v4:prepared with extra actor/memory checks, queue canceled before CUDA to fix
  compiler congruence. Source/data preserved; no completed allocation restarted.
- v5:uses trainer's-O2/default FMA and explicit `ffc-native-branches-v2` protocol.
  Its unstarted queue was canceled after discovering the earlier busy refusal
  created an empty math directory; no CUDA stage ran. Admission ordering is fixed
  and regression-tested. Evidence:queue-admission-fix.json in its packet.
- v6:prepared and host-checked. Adds RGB ramps, translated edges/dots, clipping
  and true zero-input fixtures to the CUDA checks. Captures actual stage
  supervisor bytes, including stronger offline coverage/repeat auditing and
  native timing retention. Native metrics are bin means; no invented native
  averageSPS is derived from them. Queue session89498, no duplicate launch.

The old gradient failure remains failed. New protocol requires independent
layer forwards and all native-branch gradients at unchanged tolerances. Signed
branch differences are recorded/bounded at2e-6; full-double gradients are also
checked when branches agree. Positive/zero fixtures require identical branches.

## Next actions

1. Poll the existing queue handle briefly, do not stream native TUI logs.
2. On failure, read the exact failing process and preserve it. Diagnose before
   any new bounded allocation; no tolerance relaxation or repeated blind runs.
3. On success, independently audit all120math workers/3memory suites,
   12training jobs/24checkpoints/24evals/48episodes/18import checks and their
   source/config/process bindings. Check all model/replica/skin assignments.
4. Finish compact evidence under research/results, update its README, AGENTS.md
   and NEXT_5090_FULLFRAME_COMPRESSION.md to actual qualified scope.
5. Optional timing is now implemented and GPU-free preparation passes for both
   games. Native sources come from the same frozen packet; the timing-only
   source/compiler/binaries have a separate hashed supplement. Run only after
   accepted math/smoke:60workers,B1/B16,five replicas,20warmup/100iterations,
   actual rendered reset frames,pinned H2D included,300s cap. GPU timing is still
   UNEXECUTED and is not learning/frontier evidence or part of the live queue.
6. No commit or push has been made for this goal. Never push official upstream.

Compact host-only evidence has been retained under
research/results/fullframe-compression-v1-5060-20261019/receipts/host-v6,
with34files/116284bytes and an explicit INDEX.json. Current reader additionally
binds layout/storage JSON back to the native metadata log/process receipts.
The already-loaded queue supervisor predates only that added offline check;
rerun the current offline report after completion for the strongest audit.
The current reader also summarizes registered counts/storage floors and actual
numerical errors/branch disagreements/finite-difference coverage. The offline
preacceptance report correctly returns incomplete while GPU stages are absent.
Two additional host tests verify deadline/allocation rejection before policy
execution; retained in receipts/host-limits-v6, alongside that report/reader.
The final host follow-up adds encoder-export mutation rejection and an offline
recheck of actual imported encoder/core/head bytes using registered offsets.
Its15-test evidence and reader are in receipts/host-import-v6. This audit is
offline; it does not replace the pending real native import checks. Run the
current report after the loaded queue finishes, since that older reader lacks
these last offline additions.
Seventeen-test timing-guard/source-mutation evidence, the new native timing
source and GPU-free build receipts are in receipts/host-timing-v6. Owner877672
has exited, but other GPU work resumed before the stable-idle window finished.
Queue882705/session89498 remains the only existing validation queue; verify it
live before considering any launch. Do not restart completed/failed packets.

The4x4readout may discard fine within-bin position; that limitation is now
documented in the backbone design. The first smoke tests correctness/plumbing,
not sufficient learning or a universal useful representation. Do not broaden it
into a dataset/pretraining/overnight sweep to obtain a favorable score.
