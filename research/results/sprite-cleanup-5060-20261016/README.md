# Six-game sprite cleanup: compatibility and short GPU comparison

October16, RTX5060/G240, primary checkout `/home/claude/cnn`. The six game
directories and sprite_common are real local copies, with matching configs.
Earlier worktree sources and completed evidence remain unchanged.

Implemented a requested-skin atlas cache, per-environment tile/texture stamps
that skip unchanged/repeated pixel copies, and readable struct fields/resets.
Game rules, world RNG, rewards, state/pixel observation bytes and normal
movement-only action contracts are unchanged by this cleanup. Later optional
attention work has separate builds/tests; this packet does not test it.

## Measured state training

| Game | Before process SPS | After process SPS | Change | Final collection/task perf |
|---|---:|---:|---:|---:|
| Catch | 632,680 | 639,956 | +1.15% | 100% |
| Platform | 967,621 | 973,569 | +0.61% | 100% |
| Maze | 956,377 | 970,603 | +1.49% | 95.31% |
| Runner | 1,037,111 | 1,027,488 | -0.93% | 97.01% |
| Key | 940,364 | 938,348 | -0.21% | 100% |
| Push | 909,704 | 897,935 | -1.29% | 100% |

These small timing differences do not establish a state-training speedup.
The optimization targets sprite construction/copying, largely inactive in
headless state mode. Each game has four fresh8Mi-decision runs, ordered
before/after/after/before, fixed seed7,256 agents/H64/MB2048, same per-game
learner/core. Catch keeps its existing H128/L4 core; the others H128/L1.
Timings include process startup and four checkpoint writes, excluding eval.
Native timing/SPS are retained separately.

**Every scheduled state checkpoint matches byte-for-byte across all four
runs within each game.** The two timing repeats are one independent training
seed, not four replicas. Each version's first final checkpoint is evaluated
on32 exact assigned episodes at each development suite seed420/1337,
cap2048/16 slots. All per-episode evaluation CSV rows, including action hashes,
match exactly before/after. Quality is identical, not an improvement claim.
Catch/Runner perf is item accuracy, while other tasks use binary completion;
these percentages must not be averaged into an unlabeled global score.

## Pixel execution checks

Twelve separate full640x480 Nature pixel canaries ran128 decisions each
(before/after in all six games), with finite full-policy checkpoints and
matching-policy exact evaluation (four assigned episodes per policy).
**All six before/after pixel checkpoint pairs also match byte-for-byte.**
These are execution/compatibility checks, not meaningful pixel learning or
steady-state throughput measurements. Their training processes total15.808s;
raw per-run process/native timing and checkpoint identities are retained.
Do not compare these startup-dominated canaries with longer state training
and call the ratio the intrinsic perception cost.

The complete allocation took291.444s:24 state trainings/24 state evaluations,
12 pixel trainings/12 pixel evaluations,816 assigned evaluation episodes.
No failed/weak seed was replaced, and old allocations were not restarted.

## Scalar tests and evidence

All twelve original host/sanitizer tests and twelve updated game/render
tests pass. Updated checks include first-frame full copy, zero additional
copies for an unchanged frame and independent uncached redraw equality.
Six old/new state trace files match exactly over three fixed seeds and6144
decisions/game. No CPU CNN or neural reference was used.

Raw source/build/config/process/checkpoint/CSV receipts:
`build/sprite-cleanup-20261016/`. The new allocation freezes before/after
sources and native binaries; later main attention edits do not change it.
Native launch/audit glue: [run_sprite_cleanup.py](../../run_sprite_cleanup.py).
The actual cleanup directly edited main; compatible frozen native core was
copied only for builds. Shared GPU lock/idle queries precede serial native
work. No system/CUDA/Torch change, upstream push or Pareto claim.

Next learning comparison: matched local/gaze state and tile CNN pilots in
[SPRITE_ATTENTION.md](../../SPRITE_ATTENTION.md). Those outcomes must come
from their own completed packet, not this compatibility report.
