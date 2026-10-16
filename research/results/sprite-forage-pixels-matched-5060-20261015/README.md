# Partial Forage: five-minute state versus our tile CNN

Complete and audited on RTX5060/G240, October15. Same visible cells, gaze
actions, relative geometry, game/rewards and learner recipe; fixed training
seed7. Pixels learn substantially better than in the earlier prototype,
but this is still a single-seed development comparison, not a robust win.

| Observation / policy | Train seconds | Decisions | Process SPS | Collection perf | Grape exits | Parameters |
|---|---:|---:|---:|---:|---:|---:|
|Partial one-hot state / MLP + MinGRU|308.687|64,094,208|207,635|55.08%|73.83%|83,200|
|Partial RGB / our shared-tile CNN + MinGRU|295.053|4,587,520|15,548|53.71%|55.08%|103,976|

Both final checkpoints are also the checkpoint nearest300 monotonic seconds,
using the rule frozen before training. Their post-rename elapsed times are
308.597s and294.900s; the primary comparison is approximate equal time,
not identical time or decisions. Process times include startup and eight
checkpoint writes, excluding evaluation. The two training processes total
603.740s; the full training/evaluation campaign takes612.286s. Calibration
and its acceptance evaluations are separate additional work.

The CNN's collection score is1.37 percentage points lower in this pair,
and its grape-exit rate18.75 points lower. Pixel training process SPS is
13.35 times lower. These are observed ratios for these exact recipes and
hardware, not intrinsic architecture-speed or statistical superiority claims.
Only23/256 state episodes and21/256 pixel episodes collect a perfect score.
State exits189/256 times; pixels141/256. Both die twice. Mean evaluation
length is257.51 state /330.34 pixel decisions. Collection perf is not win
rate and similar collection means do not imply equivalent behavior.

## What changed, and what did not

Both policies receive the same seventeen visible cells: eight player
neighbors plus a full3x3 ROI selected by the policy. Actions are movement5,
absolute gaze X20 and gaze Y15. Both receive player world X/Y and signed
gaze-minus-player X/Y. State has thirteen one-hot categories per cell,225
values total. Pixels have seventeen full32x32 RGB tiles plus the same four
geometry values,52,228 values total. No hidden targets, nearest-object
features, full-board CNN, downsampling or pretrained backbone are used.

The CNN shares a4x4/stride2/C8 then3x3/stride2/C16 encoder across all tiles,
projects each tile to sixteen features, concatenates tile features and
geometry, then projects to128. Both policies use H128/L1 MinGRU and the
same three action heads. This is our custom tile CNN, not Nature or IMPALA.

The common recipe is64 agents, horizon64, minibatch256, four threads,
one buffer, synchronous float32 native CUDA training. Other learner/reward
settings are inherited from the selected state policy. LR0.0141914776
anneals over each run's own fixed decision budget. Full configuration and
timing rules are in [the protocol](../../FORAGE_PIXELS_MATCHED_TIME.md).

**The smaller common batching changes the previous state recipe** from
256 agents/minibatch2048. The earlier state result remains93.55% collection
and90.23% exits at83,886,080 decisions/97.307s, with all eight checkpoints
reproduced byte-for-byte. This new55.08% state result does not replace it.
Batching, decision budget and annealing schedule differ; this experiment
does not isolate which difference explains the lower state score.
See [the retained state report](../sprite-forage-state-controls-5060-20261015/README.md).

The old pixel prototype's nearest-five-minute score was5.47% at303.072s;
its final score declined to3.52% at377.147s. The current53.71% is an
encouraging engineering result, but actions, geometry, batching and decision
budget changed together, so do not attribute that improvement to one change.
The full-resolution pixel path is still slow. Precomputed smaller sprite
tiles remain the next proposed speed experiment, not an implemented result.

## Exact evaluation and acceptance

Every final policy receives the same256 assigned worlds:128 episode IDs
at each fixed development suite seed420/1337,16 slots,512-step cap.
Native manifests bind world/policy RNG and initial world hashes; observations
match each mode's own manifest. Recurrent state resets per assigned episode.
Episode completion order never selects the evaluated sample. Training seeds
are not swept. These reused development worlds are not final publication tests.

| Suite seed | State collection | Pixel collection | State grape exits | Pixel grape exits |
|---|---:|---:|---:|---:|
|420|52.34%|52.73%|77.34%|60.16%|
|1337|57.81%|54.69%|70.31%|50.00%|

Separate from-scratch throughput calibrations selected the full decision
budgets using speed alone. Both calibration policies pass exact graph,
repeat, eager and independent final-wave replay comparisons, including
all three action heads. The allocation has four trainings, twelve evaluation
processes,616 actual episode executions, and eighteen finite/hash/size-checked
checkpoints. The offline audit passes requested/resolved recipe matching,
same initial worlds, exact evaluation assignment, native parameter counts,
timestamp-only checkpoint selection, and nonoverlapping native processes.

Earlier independent CUDA smooth-positive/zero-weight checks cover the
unchanged encoder at B1/3/8/64; this is not independent mathematical
qualification of B256 gradients, arbitrary ReLU branches or the full learner.
The new trainings have not been independently repeated; one seed is not
cross-seed confirmation. No Pareto/generalization/SOTA claim is established.

Both jobs use the shared reservation and idle checks before each native
process. This does not certify absence of all external CPU/GPU contention
throughout execution. Native adjusted uptime/SPS remain in the receipts;
the table uses monotonic process measurements. Hardware: RTX5060,8,151MiB,
driver610.88. No system/CUDA/dependency changes, main-core overlay, pretrained
weights, CPU neural computation, commit or push were needed.

## Evidence and reproduction

Raw packet: `build/sprite-forage-pixels-matched-5060-20261015/`.
Frozen binary/source: `build/sprite-forage-onehot-relative-build-20261015/`.
The source/build was reused unchanged for both modes. Compact receipts,
configs, metrics, episode CSVs, source snapshots and tools are preserved
here with `SHA256SUMS`. Full training stdout and weights remain in the raw
packet; the compact copy does not certify complete vendor/system/toolchain
closure or substitute for those weights.

Reaudit without GPU execution:

```bash
.venv/bin/python research/audit_forage_pixels_matched_time.py build/sprite-forage-pixels-matched-5060-20261015
```

Read [audit.json](audit.json). Do not restart or extend this completed
allocation automatically. Any next comparison must preserve these results,
including the weak state control, rather than substituting a better seed.
