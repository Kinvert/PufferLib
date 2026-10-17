# Seven sprite games: matching partial state and tile-CNN pilots

Completed October16 on RTX5060/G240. **All seven games now have working
matching local/gaze state and pixel paths**, with actual native training,
checkpoint reload and exact assigned-episode evaluation. These are short,
single-training-seed development results, not a Nature/Pareto/generalization
comparison or proof that learned gaze beats fixed gaze.

All final policies train1,048,576 decisions from scratch at fixed seed7.
Each receives32 exactly assigned episodes at each development suite420/1337
(64 episodes/policy). Same worlds/policy RNG assignments within each game,
complete quotas independent of episode completion order. Every final policy
is the last scheduled checkpoint, never chosen by score. State/CNN use
identical learner/core/action/visibility settings within a game.

| Game | Partial-state perf | Pixel perf | State process SPS | Pixel process SPS | Pixel train seconds |
|---|---:|---:|---:|---:|---:|
| Catch | 59.06% | 58.13% | 107,634 | 12,369 | 84.77 |
| Platform | 100% | 100% | 144,014 | 12,840 | 81.67 |
| Maze | 37.50% | 34.38% | 145,004 | 12,872 | 81.46 |
| Runner | 69.66% | 69.66% | 143,003 | 12,824 | 81.77 |
| Key | 46.88% | 85.94% | 139,215 | 12,778 | 82.06 |
| Push | 96.88% | 82.81% | 138,227 | 12,840 | 81.66 |
| Lane Dodge | 63.77% | 68.71% | 141,092 | 12,840 | 81.66 |

Perf definitions differ: Catch/Runner measure item accuracy; Platform/Maze/
Key/Push measure successful completion; Lane Dodge is half survival plus
half coin fraction. **Do not average these raw percentages or call every
row a win rate.** This experiment matches decisions, not wall-clock time:
partial state takes about7–10s, pixels about81–85s. It does not demonstrate
that pixels beat state at equal time. Key's positive gap is one observation
from one training seed, not evidence of general pixel superiority.

Full-state references are separate: the compatible cleanup panel reached
100% in Catch/Platform/Key/Push,95.31% Maze and97.01% Runner. Partial views
remove global target/semantic information, so lower partial-state scores
are not evidence that cleanup broke the original games.
[Full-state cleanup results](../sprite-cleanup-5060-20261016/README.md).

## Interfaces and recipes

The six original sprite games see18 tiles: centered3x3 player and3x3 chosen
gaze, overlapping cells allowed. Movement retains its original choices,
with additional absolute cellX20/Y15 categorical heads. Four geometry values
give player worldX/Y and relative gaze offsets. State supplies one-hot tile
identities; pixels supply cached32x32 RGB tiles. No hidden object lists,
nearest-target hints or auxiliary velocity/timing values are added to the
partial path. The ordinary one-head full-state/full-image builds remain.

Shared CNN: valid4x4/s2/C8 →3x3/s2/C16 →learned784-to-16 per tile,
ordered features plus geometry →H128 →the game's existing MinGRU/core and
decoder. No pretraining, full-map image construction, runtime image resize,
graphics context or Python learner. Captured Forage CUDA callbacks implement
the shared encoder directly via research dimension aliases.

The common six-game pilot batch is A32/H32/MB256/two threads. Each game's
existing LR/gamma/entropy/core depth remains fixed between state and pixels;
Catch keeps H128/L4, the others H128/L1. Per-game config is frozen rather
than swept. Larger learner tuning/longer training remains separate from CNN
architecture comparisons.

Lane Dodge intentionally retains its road-specific eighteen-tile view:
the player window spans two rows ahead through the player row, then3x3 gaze;
movement3/gaze300 heads and three absolute geometry values. New mode4 is
111-value one-hot state matching existing mode3 RGB. A32/H64/MB2048 with
its retained LR/rewards, H128/L1. State has102,272 parameters, pixel139,560.
Its metadata/action contract is not silently equated with the other games.
The new mode3 final checkpoint SHA256
`a81a456ff5a20cb0d89753a2ea2b98b0e3b7325c737a0352938ef6426c1fc218`
matches Qwen's earlier retained d1/d2 final weights exactly. Separate native
reload works in this packet; no in-process or retry/debug workaround is used.

## Verification actually executed

- All seven native game/sanitizer suites pass. Six-game partial checks
  cover3072 paired steps/game, all three skins, exact full-image crop parity,
  one-hot/geometry bounds, gaze/reset timing and hidden-cell mutation.
  Lane adds3072 matching mode3/4 steps and a complete192x192 enlarged-view
  regression that catches the previous tile-overlap rendering bug.
- Independent float64 CUDA forward/all-parameter gradient checks pass
  B1/3/8/64/256, positive and zero fixtures, exact eager/graph repetitions,
  device parameter preservation:51,624 encoder parameters for the shared
  four-geometry view,51,496 for Lane's three-geometry view.20 cases total.
  This does not qualify arbitrary ReLU branch gradients/full learner math.
- Twenty-eight fresh8192-decision canary trainings form14 same-seed repeat
  pairs. Every pair's checkpoint bytes match. Forty-two exact evaluations
  pass graph/repeat/eager episode-row equality, including every action head
  and a three-episode/two-slot partial final wave.
- Fourteen1Mi-decision pilots produce56 finite, shape-checked scheduled
  checkpoints. Twenty-eight final exact evaluations cover896 episode rows.
  The canary checks add126 rows. Total:114 GPU processes,1,022 assigned
  episode executions, with monotonic process start/end and serial idle/lock
  receipts. No failed native process was retried.
- All seven ordinary sprite targets plus golden `g2048` also compile from
  the actual main checkout's normal build path. This is separate from the
  captured compatible research-core builds used for the timed pilots.

The six-game campaign took795.410s including reporting recovery; Lane's
separate campaign took102.202s. Timing tables use decisions/monotonic process
duration, including startup/checkpoint writes and excluding evaluation.
Native uptime/dashboard SPS remain distinct in receipts. Hardware results
are local5060, not5090. Shared lock/idle queries do not certify all possible
external contention or intrinsic kernel speed.

## Retained failures and limits

The first six-game reader stopped after a successful episode-free Maze
canary because it incorrectly required `env/perf`. Missing canary perf is
now null. The original runner/failure/process receipts are preserved;
explicit continuation verified existing commands/configs/logs/checkpoints,
reused all successful native work, and launched only missing work. Both
COMPLETE and the historical failure.json remain; status is complete after
reader recovery, not a claim of a failure-free supervisor history.

An earlier attention build rejected a core already patched for Forage.
That build failure is preserved; a fresh explicit compatible-core build
compiled the suite. Production native sources were not waived/modified to
pass an independent math gate. No system/CUDA/Torch/dependency change,
dataset, CPU neural execution, commit or push occurred.

No independent multi-seed training confirmation, useful-gaze ablation,
held-out generalization or matched full-image Nature curve is established.
The smaller observation method must not be labeled a CNN-only speedup over
full-image Nature. Next meaningful research: fixed/random gaze controls,
precomputed smaller sprite tiles, and matched pixel-candidate curves with
the seven fixed replicas7/42/69/123/173/2027/9001. Seeds are never optimized.

## Reproduction and raw packets

- [Six-game protocol/build commands](../../SPRITE_ATTENTION.md)
- [Lane-specific contract](../../LANE_DODGE_MATCHED_STATE.md)
- Six-game raw: `build/sprite-attention-pilots-20261016/`;
  captured source/binaries: `build/sprite-attention-20261016-v2/`.
- Lane raw: `build/sprite-lanedodge-pilots-20261016/`;
  captured source/binary: `build/sprite-lanedodge-matched-20261016/`.
- Final normal/sanitizer/main-build receipts: `build/sprite-cleanup-20261016/`.

These allocations are complete. Do not restart/extend them automatically or
overwrite original weights/configs/source identities. New experiments need
new packets and explicit scope under the active research task.
