# Agent-controlled spatial attention

Kinvert's architecture idea, October15. The first opt-in native prototype is
implemented and its four-mode RTX5060 canary is complete. Read
[the implementation/run instructions](FORAGE_ATTENTION.md) and
[the exact test scope/results](results/sprite-forage-attention-5060-20261015/README.md).
No long learning allocation follows automatically.

The subsequently requested short and five-minute seed7 learning pilots are
also complete. At approximately five minutes, restricted state scores61.33%
and restricted pixels5.47% collection perf; the latter falls to3.52% at its
final377-second checkpoint. Full state scores97.66% at~11 seconds. This is
not a useful pixel policy or demonstrated learned attention advantage. Read
[the full timing/config/evaluation receipts](results/sprite-forage-attention-five-minute-5060-20261015/README.md).

## The requested idea

Run convolution kernels on just the **eight cells surrounding the player**.
Also maintain a separate center square **the agent chooses through actions**,
and convolve the **nine cells in its3x3 pixel neighborhood**. Learn both what
to do in the game and where to look. Call this architecture **attention**.

This is selective visual computation, not a full-map CNN followed by a mask.
The inputs are the actual rendered sprite pixels, not object IDs, reward
labels or privileged nearest-target coordinates. The two views provide local
safety information plus a deliberately chosen remote glimpse. The recurrent
policy can retain information after its gaze moves elsewhere.

The first view literally has eight neighbors; it does not silently become
nine. Including the player's center tile is a separate optional variant.
At32x32 pixels per cell,8+9 tiles cover17,408 pixel positions versus307,200
on the20x15 board: about5.7%, before overlap or border handling. This is a
pixel-count argument, not a measured17.6x training speedup. CNN launch/batching,
policy, optimizer and learning costs remain to be measured.

## Required matching state mode

Kinvert also requires a state-observation version with exactly the same
visibility: only the eight neighboring cells and the nine cells around the
agent-selected gaze center expose their state. Hidden cells must not appear
in a full-board grid, coordinate list, nearest-object helper or other state
feature. The agent must use the same gaze actions to inspect them.

Keep the state and pixel modes paired: identical window geometry, action
timing, border rules, permitted position/gaze metadata, world RNG, rewards
and episode limits. The state mode can encode the visible cells' semantic
values directly; the pixel mode has to recognize their rendered appearances.
Both receive the same locations and must remember things that leave view.
This makes a controlled state-versus-pixels comparison under partial vision,
alongside the existing full-state and full-pixel baselines.

Before training, test that changing hidden cell contents leaves either
mode's current observation unchanged, and that matched state/pixel views
refer to exactly the same cells. This is required prototype scope, not an
optional later ablation. These visibility/crop tests now pass for the prototype.

## First prototype design

The requested state action/observation ablations are now complete. Read
[the matched results and exact-repeat evidence](results/sprite-forage-state-controls-5060-20261015/README.md).
Factored X/Y gaze + visible-cell one-hot identities + signed ROI offsets
reach93.55% collection/90.23% grape exits in97.3 seconds at the same83.886M
decisions as the original64.26%/123.8s partial control. All eight checkpoint
hashes and256 episode rows repeat exactly from scratch. This is one fixed
training seed; independent confirmation and gaze-use controls remain.
Pixel resolution and downsampling have not changed.

### Kinvert's revised controls and smaller pixel input — October15

The following design was implemented after the earlier five-minute pilots.
It supersedes the proposed eight-direction/recenter gaze actions. Preserve
the original absolute300 build and all its evidence as a historical control.

Kinvert's implementation order: **get the action/observation changes working
and run the initial partial-state ablations first. Only afterward work on
sprite downsampling.** Do not change image resolution while comparing those
action/observation variants.

Test exactly two new gaze interfaces, independent of the unchanged five
player movement actions:

1. **Directional5:** noop/right/left/up/down. A direction shifts the existing
   gaze center one board cell; noop holds that center in world coordinates.
   Clamp x/y at the board edge, without wrapping rows. No diagonal, recenter
   or extra free observation action is added.
2. **Factored absolute XY:** one20-choice discrete X head and one15-choice
   discrete Y head choose the ROI center directly. The policy can jump its
   gaze anywhere in one game step. These35 gaze logits replace the original
   300-way head; they are two categorical cell-index choices, not continuous
   control or pixel coordinates. Normalized labels such as-1 through1 can
   describe the positions, but each axis has exactly its board dimension's
   number of choices. Native action-head shapes are currently compile-time
   constants, so implementations must bind each binary/checkpoint to its
   explicit interface rather than pretend an INI changes an old binary.

Give both state/pixel policies explicit signed ROI-to-player displacement:
`dx=(gaze_x-player_x)/(width-1)`, `dy=(gaze_y-player_y)/(height-1)`.
Keep normalized player world X/Y. These four geometry values retain absolute
world context (gaze position is reconstructible), while making the relation
direct. Compute them after movement and gaze update; reset still starts gaze
on the player. Geometry changes alone must not consume RNG or expose hidden
objects. Ablate relative versus the existing absolute geometry separately.

**Massively reduce actual CNN input**, rather than merely masking a large
image. The prototype already reads only17 selected tiles, but32x32 RGB per
tile still costs52,224 pixel floats. At8x8 RGB this becomes3,264 (16x less);
at4x4 RGB,816 (64x less), plus the same four geometry values. These ratios
describe input size, not guaranteed end-to-end speedup.

After the initial action/observation ablations, compare deterministic sprite
reduction methods such as max pooling and average pooling (mean pooling is
the same operation as average pooling). Mathematically reduce each sprite
ahead of stepping, during asset preparation or one-time initialization, and
cache the small pixel arrays. There is **no per-step/per-frame resampling or
pooling preprocessing**. Each step only copies selected precomputed tiles.
Do not build640x480 and resize it per step, copy large tiles to the GPU then
shrink them, or replace sprite pixels with semantic IDs in pixel mode.
Preserve the full-resolution human viewer. Start with8x8; adapt the encoder
for4x4 before testing it, since the
current4x4-then3x3 valid convolutions do not fit a4x4 input. Check that
lookalike apples/hazards retain their distinguishing marks after downsampling.

First use approximately120-second **state** ablations: old absolute300
control, directional5, factored XY, then the relative-coordinate observation
change isolated from action changes. Keep learner/rewards/core and assigned
train/eval seeds fixed, record every result and true monotonic training time,
and never optimize seeds. A time deadline/checkpoint boundary should avoid
repeating the previous large fixed-step runtime overshoot. Numerical/world/
visibility tests and short GPU canaries precede the learning allocation.
Afterward compare reduced-pixel candidates under the same chosen interface.

- Keep a game-movement action and add a separate gaze action. A direct grid
  cell choice (300 possible centers here) is the simplest initial candidate;
  relative gaze movement is another option. The agent must actually choose
  the center, without an oracle target picker or random look-window sweep.
- At time t, pixels reflect the current game and previous gaze choice. The
  policy outputs movement and its next gaze center. Apply movement and gaze,
  then expose the next local/remote windows. Decide this ordering explicitly;
  do not let a gaze action reveal future pixels in the same observation.
- Keep every game action's timestep, rewards, hazards and episode limits.
  Movement can stay still while the policy looks; looking gets no automatic
  extra free game steps or label-based reward bonus.
- Reuse cached sprite pixels to fill only selected tiles, without building
  the full observation frame first. Keep the full human viewer separate.
- A simple implementation candidate is a shared small tile CNN, batched over
  the eight local and nine gaze tiles, followed by features in fixed spatial
  order and the recurrent core. Compare with contiguous cropped windows if
  cross-tile convolution is useful. No giant full-map flatten projection.
- Define border padding and overlapping-view behavior. For the prototype,
  duplicate overlap is acceptable; deduplication is optional later work.
- Centering the local view uses the player's location as requested. Any
  policy-visible gaze coordinates, relative offsets or padding masks must be
  documented as geometry metadata. Do not silently pass semantic state.

## Determinism and comparisons

Given the same seed and movement actions, changing gaze must leave game state,
rewards, spawns and world RNG unchanged. Gaze selection/pixel generation must
not consume world RNG. Test exact crop pixels against the corresponding full
render, including borders, pickups, resets and every skin. New checkpoints
need an explicit observation/action contract; old same-sized files are not
automatically compatible.

Preserve current state and full-frame pixel modes. Keep an opt-in research
prototype separate from the eventual minimal Sprite Forage PR integration.
The local crop and active glimpse change observability and the action space;
compare them honestly with full vision, local-only, fixed-gaze and a declared
random-gaze control. Report inference as well as training time, parameter
count, input/copy volume, all scheduled checkpoints and complete evaluations.
Do not assume faster learning or a Pareto win from fewer pixels alone.

Use fixed paired training seeds, never sweep seeds, and different fixed eval
worlds. Memory becomes relevant because previously seen distant objects can
leave the current windows. First examine a small deterministic functional
prototype; establish related work before making novelty or paper claims.

## Motivation from the current run

The spatial-downsampling/batch fix keeps full640x480 input and improves the
matched221,184-decision run from612.09s to165.05s (about3.71x). However, its
final evaluation perf is3.125%, below its9.375% untrained control. This is a
speed improvement with a learning regression, not a winning architecture.
See [the complete experiment](results/sprite-forage-speed-5060-20261015/README.md).
