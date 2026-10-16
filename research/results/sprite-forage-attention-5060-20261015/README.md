# Sprite Forage attention prototype, RTX5060, October15

The requested native attention prototype is implemented. All four INI modes
train/evaluate successfully. This is functional/mathematical smoke evidence,
not learned attention, game mastery or Pareto dominance. The original one-head
baseline stays available in normal builds.

## Actual four-mode canary

Each row is one fresh seed7 native float32 training,8,192 decisions,8 agents,
H8/MB8, H128/L1, same starting learner/rewards/drawing0. Every row here uses
the research build's two action heads (movement5 + absolute gaze300). In
full modes, gaze does not affect game/observations; the extra head remains
part of the policy/loss and differs from original one-head baselines.

|Visibility/input|Obs floats|Full parameters|Process seconds|Process SPS|Eval perf|
|---|---:|---:|---:|---:|---:|
|Full state|141|106,368|1.117|7,337|0%|
|Full pixels, spatial Flex|921,600|1,332,512|7.582|1,080|21.875%|
|Partial state|21|91,008|1.016|8,062|6.25%|
|Partial pixels, shared tile CNN|52,228|137,896|1.818|4,505|3.125%|

Partial pixels are about4.17x the full-pixel process SPS in these short runs.
Observability and architecture differ, and startup/checkpoint cost is material.
No learning-efficiency or equal-quality speedup claim is justified. Full pixel
control uses the supplied depth3 spatial Flex graph, not Nature or the old
19.7M graph. Scores are collection-based perf, not win rates. All losses and
weak scores remain; there is no untrained baseline or convergence test here.

All four checkpoints have the exact expected size and finite float32 values.
Requested and resolved env/vec/policy/train settings agree. Six8-episode native
eval processes complete (48 episode executions), using one slot, world420,
seed7 and512-step cap. Partial-pixel graph/repeat/eager aggregate scores match
exactly. This is summary equality, not independent episode/action traces.
The overall native train/eval campaign takes19.514s. No seeds were swept.

## Observation and game checks

Seven ASan/UBSan suites pass. New attention tests compare24,000 actual game
steps across all four modes and five skins: player, world RNG/grid, rewards,
terminals and logs agree given the same movement and different gaze actions.
All1,500 skin/gaze-center pairs compare exact pixels to the full-frame raster,
including overlaps, edges/corners, player overlay and wall padding. State and
pixel view geometry match. Modifying all currently hidden cells changes
neither partial observation. Waiting/look-only and autoreset gaze timing pass.
Original game/raster/RNG/asset/viewer tests remain passing.

Partial observations contain eight player-neighbor cells and nine gaze cells.
Only selected cached sprite tiles are copied; no full frame or global object
list is built for the partial policy. Both modes append the same four explicit
player/gaze geometry floats. The current simple implementation copies all17
attention slots each observation; unchanged-slot caching is future work.

## Actual new encoder CUDA checks

The encoder has49,576 parameters. The production shared tile CNN uses the
existing Nature convolution/GEMM/derivative helpers with new descriptors;
three new packing kernels handle observation metadata and feature gradients.
The test uses independently stated topology and the existing direct float64
CUDA reference kernels, with no production convolution/GEMM in the oracle.
CPU work only constructs fixtures, converts/copies arrays and compares values.

Eight cases pass: B1/3/8/64 crossed with smooth-positive and zero-weight
fixtures. Each compares output and every encoder parameter gradient with
predeclared rtol8e-4/atol6e-5. One eager and two graph executions per case
(24 production calls) match exactly, with exact device-weight preservation.
The reference dyadic self-test also passes. The native test process takes
about0.69s; this is a correctness check, not a performance comparison.

These fixtures do not qualify arbitrary mixed ReLU branches, selected finite
differences, fresh-process numerical repeats, input derivatives or full
head/core/optimizer math. No publication/full-policy-math waiver follows.
An initial test build was retained unexecuted, then corrected before the
only actual CUDA test allocation. The production encoder source is unchanged.

## Sources, use and evidence

[Exact interface/build instructions](../../FORAGE_ATTENTION.md).
[Recorded architecture proposal](../../AGENT_CONTROLLED_ATTENTION.md).
All edits are in `/home/claude/cnn`, including environment hooks and the
research encoder, dispatch/build tools, canary runner and INI recipes.
Native core integration is copied from a retained compatible build and patched
only inside the fresh build tree; main core files are not overlaid. This
local dependency must be packaged before a future5090 handoff. This is not
the eventual minimal clean Sprite Forage PR.

Raw game tests: `build/sprite-forage-attention-tests-20261015/` and its log.
Native build/math: `build/sprite-forage-attention-build-20261015/`.
Four-mode canary: `build/sprite-forage-attention-canary-20261015/`.
Compact result/plan/receipts and hashes are retained here; large/raw files stay
under ignored build. Existing CUDA/NCCL, shared lock and idle checks only.
No CPU CNN, system/Torch/dependency change, dataset,5090 job, commit or push.
No completed allocation is restarted or extended; further learning remains
unqualified and unscheduled.
