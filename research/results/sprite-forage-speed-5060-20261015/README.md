# Full-resolution spatial-head and batch experiment, RTX5060, October15

All declared work completed: fresh independent encoder checks, three short
native training canaries, one predeclared conditional longer training and
seven evaluation processes. No failure, retry, seed search or checkpoint
selection. Full640x480 RGB input, same game/drawing/rewards, seed7,8 agents,
H8, H128/L1, LR.001 and native Muon throughout. No main core/kernel changes.

The candidate activates existing depth3 controls:7x7/s4 C16,3x3/s2 C16,
3x3/s2 C16, flatten/linear64, linear128. Spatial features are30x40x16;
full parameters1,294,112 versus19,721,472 before. The projection falls from
19,660,800 weights to1,228,800. The visible recipe is
`research/recipes/sprite_forage_spatial_cnn.ini`.

|8192-decision canary|Parameters|Minibatch|Process seconds|Process SPS|Final eval perf, world420|
|---|---:|---:|---:|---:|---:|
|Old depth1|19,721,472|8|23.520|348.29|12.5%|
|New depth3|1,294,112|8|7.431|1,102.46|7.8125%|
|New depth3|1,294,112|64|6.529|1,254.63|3.125%|

The same-batch head change gives about3.17x process SPS. Increasing MB8 to64
adds about1.14x in these short canaries; it also changes optimization from an
update every8 decisions to every64. Startup/checkpoint costs are material in
these few-second runs. Sixteen serial episodes per canary at420 are execution
and development measurements, not a basis for quality-winner selection.

## Matched-decision follow-up

The batch64 canary projected221,184 decisions in176.29s, satisfying the
predeclared<=300s resource condition. No score was used for this decision.
The fresh longer run completed221,184 decisions in165.048s (2m45s),
1,340.12 process SPS,1,344.18 native adjusted-average SPS and1,335.73 native
final-bin SPS. Total post-math training/evaluation campaign217.667s.

The old depth1/MB8 run at the same decision budget took612.087s at361.36
process SPS. The new configuration is3.71x faster in this one comparison;
both architecture and minibatch changed. Same native binary and input/game
sources; output paths and checkpoint cadence are captured. The optimizer,
precision, other learner settings, rollout size and training seed stay fixed.

|Development world|New architecture, untrained|New architecture, final|
|---|---:|---:|
|420|7.8125%|4.6875%|
|1337|10.9375%|1.5625%|
|Mean,32 episodes|9.375%|3.125%|

**Learning regressed.** Both development worlds decline. Final training-bin
perf is5%. The previous slower network's final eval was9.375%. Keep all of
these outcomes: improved throughput is not improved learning, and no Pareto
dominance, reliable learning or generalization claim follows. This is one
independent training seed; evaluation episodes are not training replicas.

Initial/final follow-up evaluations use16 sequential episodes per world,
one slot, fixed action/policy seed7, drawing0 and512-step cap. These are
existing development world offsets and the legacy native serial aggregate
evaluator, not new held-out confirmation or per-ID action-trace acceptance.
All112 assigned episodes across the seven processes finish. Final scheduled
checkpoints only are evaluated; no best-checkpoint selection.

## Numerical and runtime checks

Before training the new graph: six fresh workers,18 cases at B1/8/64,
72 native eager/graph calls pass independent CUDA float64/actual-float32-
branch comparisons, all gradients, device parameters, exact repeats/process
checks and selected finite differences, with unchanged tolerances. No CPU
neural reference. Existing full-image native/reference libraries are reused
with fresh config/bundle/allocations. Their owned cnn/nature/flex files match
the actual training snapshot. Full learner/optimizer derivatives remain
unqualified; encoder acceptance is not publication acceptance.

All seven checkpoint files have expected byte length, finite values and
captured SHA256. The new graph's files are5,176,448 bytes. All requested and
resolved env/vec/policy/train values match. Batch64 completes both short and
long full-policy runs without OOM; no peak-VRAM measurement is claimed.
Shared GPU lock and idle checks guard native execution. No heavy CPU work
overlaps timed training; the separate raw numerical allocation is about4.4GiB.

Raw packet: `build/sprite-forage-speed-5060-20261015/`. Compact results,
plan, training/evals, conditional decision, math report/request and raw hashes
are copied here. Source snapshots, binaries, full arrays and weights remain
ignored under build. Reused tools are read-only in the retained research tree;
all new edits and packets are under ~/cnn. Existing CUDA/NCCL only; no system,
Torch/dependency change, new dataset,5090 run, commit or push. No automatic
extension follows. Kinvert's next requested architecture idea is recorded in
`research/AGENT_CONTROLLED_ATTENTION.md`; it is not implemented here.
