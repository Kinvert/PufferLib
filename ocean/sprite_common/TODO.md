# Sprite game improvements

Requested by Kinvert after watching the first six trained policies, October15.
Keep the existing per-game folders and normal build conventions. Environment
code stays simple native C; no build-system/core refactor for organization.

## Active solid-benchmark goal, October15

- [x] Every game must have a real lethal failure, including Catch skulls and
  Runner hazards. Forage already has lethal toadstools; preserve that behavior.
  Record death separately from timeout, missed rewards and successful completion.
- [x] Catch/Runner early deaths must not change the next assigned world. Give
  each episode its own object RNG stream; the reset/world RNG advances only
  on reset. Never score only surviving episodes or divide perf by only the
  few items attempted before death. Keep failed/incomplete item budgets visible.
- [x] Spread Maze traps among safe leaf passages throughout the maze, removing
  the fixed-right-edge shortcut without blocking the safe solution.
- [x] Dial in Runner and the changed Catch task with bounded native GPU
  development runs. Freeze per-game learner recipes before confirmation.
- [ ] Verify state learning across fixed seeds7/42/69 on separate fixed world
  streams, with final checkpoints, paired episode quotas, deaths, perfection,
  actual process SPS/times and every failed attempt retained.
  Current evidence:17 completed jobs, Catch42 timed out and remains missing;
  Runner42 is weak. Final-world metrics are audited for all17 policies. Never
  replace those replicas. The next larger matched comparison preassigns seven
  fixed seeds7/42/69/123/173/2027/9001; existing results remain three-seed evidence.
- [x] Repeat correctness/render/state-pixel/world-assignment checks on the final
  code; inspect individual changes for simple C, no per-step allocation and no
  unrelated core/build edits. Keep images at640x480 and cached backgrounds.
- [ ] Establish actual pixel-learning measurements and a clean reproducible
  path to compare Nature/IMPALA/custom encoders under matched learners and
  deterministic evaluation. Plumbing/state success alone is not CNN evidence.
  Runner's8192-decision full-resolution Nature pilot now supplies first learning
  measurements (almost no improvement), with all four checkpoints retained.
  The opt-in multi-encoder adapter now compiles all21 game/family targets.
  Nature and locked Flex pass full-resolution encoder math at H128/B1/2/8;
  Runner state/Nature checkpoint receipts match the original implementation
  byte-for-byte across14 new native eval processes/66 episode rows. Larger
  learner shapes, IMPALA mathematics/runtime and actual matched training remain.
  Read research/SPRITE_ENCODER_ADAPTER.md; this is temporary research integration.
- [x] Publish a scoped readiness report and instructions identifying exactly
  what is verified and what remains before a Pareto/generalization/paper claim.
  See research/results/sprite-solid-5060-20261015/README.md and its scoped audit.

Later augmentation work: hero1/hero2/hero3/hero4 and alternate item/terrain art,
with held-out sprite families and independent visual/gameplay seeds. Record the
design now; don't build an asset framework or new dataset during this cleanup.
Also separate visual recognition from geometric shortcuts: Maze traps occupy
leaves, and Push's fake crate starts farther away than the real crate. Future
held-out layouts/placements should challenge these cues; current completion
doesn't prove that the model recognized the lookalike artwork.

Per-episode native export is implemented: IDs/world seeds, per-episode policy
seeds, starting hashes, perf/success/death/return and action hash in one pass.
All seven games/state-pixel pairs pass102 native processes/482 executions across
all23 drawing IDs; repeat/eager and the tested partition rows match. No new
training or quality claim. See research/results/sprite-exact-5060-20261015/README.md.
This is temporary research tooling, outside the clean game PR. Old summary
replays remain preserved and do not increase the independent training sample n.

Build-all routing fixed: a four-line generic entry-file filter skips helpers
without their matching `.h`/`.cu`/`.c`. Actual dispatcher tests cover all three
entry types and the real ocean inventory: only `osrs`/`sprite_common` are skipped;
all seven sprite games remain, each dispatched for both precisions. No per-env
special cases or individual-build behavior changed. See
research/results/sprite-build-routing-20261015/README.md. Other comparison/PR
gates remain; this fix alone doesn't make the whole suite PR-ready.

## Preserve the first release

- [x] Keep the first release's sources, binaries, configs, checkpoints, weaker
  attempts and audited results under `build/sprite-night-20261015/`.
- [x] Distinguish state learning from CNN execution checks: first-release
  Platform state policies completed100% of the assigned evaluations. Its CNN
  only passed128-decision train/reload checks; learned pixel quality is unknown.
- [x] Reopen the original trained six-game tour on DISPLAY=:0 for GIF capture.
  One window at a time; close/Esc advances to the next game. Use frozen binaries,
  checkpoints and matching configs, irrespective of ongoing source edits.
  Tour: `build/sprite-gif-tour.FEae97/`, supervisor session14486.

## Gameplay changes

- [x] **Platform:** vary ground height instead of always three tiles high.
  Include height changes and raised sections that require jumping onto them,
  along with gaps. Construct reachable levels and verify with actual integer
  movement/collision rules. Success must require reaching the visible flag.
- [x] **Maze:** add actual branching corridors, dead ends and visible lethal
  traps, with a guaranteed safe route. Reference the native `ocean/maze` growing-
  tree implementation; the first sprite maze was a separate corridor task,
  not a reskin of that environment. Keep maze construction/reset work off the
  per-step path and retain local state observations as a semantic reference.
- [x] **Runner:** place some beneficial objects in the air so collecting them
  requires jumping. Keep ground collectibles and hazards; expose object height
  in state. Preassigned item kinds/heights and spawn times remain independent
  of actions, deaths, drawing and policy architecture.
- [x] **Key:** replace the exposed key with a good treasure chest and a lethal
  mimic. They must have the same silhouette and differ by only one or two
  source-art pixels, with positions randomized so location is not the answer.
  Opening the good chest gives the key, unlocks the door and allows escape;
  touching/opening the mimic kills the player. Add varied floor textures.
- [x] **Push:** add obstacles that allow bad pushes to strand the real crate.
  Add a visually similar lethal fake crate. Add a slow pursuing enemy that
  moves toward the player on every second turn; contact in either direction
  kills. Add varied floor textures. Generate solvable starts and check the
  enemy/collision ordering, blocked pushes and corner deadlocks explicitly.
- [x] **All backgrounds/floors:** Kinvert clarified this applies to every game,
  including Catch's backdrop, Platform's sky, Maze's paths and Runner's desert.
  Bake four textured variants into the immutable atlas once at pixel init.
  Select them by episode/cell hash without drawing from gameplay RNG. Texture
  empty pixels behind sprites too; no background tile may be a solid color.
  Sprite Forage already has its separately cached grass/background textures.
- [x] Further maze variation: Maze v3 distributes three traps among safe leaf
  passages across the whole maze. Tests cover both halves, degree-one placement,
  explicit deaths and oracle-solvable starts. Leaf placement itself is still
  a geometric avoidance cue; don't claim sprite recognition is the only solution.

## Determinism, observations and validation

- [x] Version every changed task; old weights/results keep their original
  meaning even if a new state vector happens to have the same length.
- [x] Use identical rules, rewards and collision logic in state/pixel modes.
  State may directly identify object roles; pixels must show the distinguishing
  details. Do not accidentally give identical scalar labels to fatal objects.
- [x] Floor texture choices are deterministic, independent of semantic roles
  and stable while moving objects uncover cells. Never consume gameplay RNG
  from rendering or observation generation; cache tiles and copy dirty rows.
- [x] Extend ASan/UBSan parity, observation guards, dirty/full redraw and viewer
  tests. Check next-episode world assignment under different action sequences.
- [x] Verify generated safe solutions, explicit lethal paths and the new
  height/jump/chest/enemy mechanics. Test more than the visually obvious case.
- [x] Build fresh binaries and run bounded GPU state-learning checks and RGB
  train/reload smokes as GPU availability permits. Keep new results separate
  from the first release; don't present its100% results as new-task performance.

Snake and the lower-priority Hexic-style game remain candidates in
`research/SPRITE_GAMES_OVERNIGHT.md`; these improvements come first.

First follow-up state check (before the all-background texture-only change):
16Mi decisions, fixed seed7,32 episodes each at worlds420/1337. Platform, Maze,
Key and Push score1; Runner item accuracy.75651. These are one-seed development
results, not three-seed confirmation or CNN quality. Keep raw
`build/sprite-upgrades-20261015/state-a` and the earlier pixel check unchanged.
After the background change verify state/world parity, rebuild all six games
and run fresh all-background RGB smokes. Do not label the original first-release
GIFs or its state learning table as results for the changed environments.

Completed: all12 textured ASan/render logs, six native float builds,6,144 exact
before/after scalar state/world steps per game, six RGB trainings/30 evaluations
and20 rebuilt native state replays (both worlds/repeat/eager) pass. The file-only
audit and scoped findings are in
`research/results/sprite-upgrades-5060-20261015/README.md`.

- [x] Calibrate Runner state learning with airborne items; the first seed7
  result is75.651% item accuracy. Preserve it and use a newly declared allocation
  rather than silently continuing/restarting this completed packet. New v3 uses
  clearer absolute object-height state and lethal hazards; development A reaches
  99.93% full-budget accuracy/99.22% perfect episodes at seed7,16Mi decisions.
- [ ] Confirm harder tasks on additional fixed training seeds and separate
  evaluation streams before describing state mastery as robust.
- [ ] Train pixel policies long enough to measure learning, then compare CNNs
  under matched per-game learners, budgets, draws and deterministic assignments.
