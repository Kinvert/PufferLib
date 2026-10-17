# Sprite Forage research roadmap

The purpose is a controlled state-versus-pixels benchmark for improving game
CNNs. A useful baseline does not require solving every episode or reaching
100% perf. Prioritize deterministic rules, paired evaluation worlds, stable
metrics and clear learning/time curves.

## Attention architecture — October15 prototype complete

- [x] Revised controls requested by Kinvert: exactly five directional ROI
  actions (noop/right/left/up/down), or separate absolute cell-X/cell-Y
  categorical actions (20 and15 choices). Add explicit signed ROI-to-player
  offsets while retaining player world coordinates. Compare controls and
  geometry separately in120-second fixed-seed state ablations. No diagonal
  or recenter action. See the revised design in
  [agent-controlled spatial attention](../../research/AGENT_CONTROLLED_ATTENTION.md).
- [x] Implement/test the requested controls and initial matched state ablations.
  Six scalar action/geometry cases plus full-state control, three one-hot
  cases and factored one-hot/relative interaction are complete on5060.
  The selected partial policy reaches93.55% collection/90.23% grape exits
  versus original64.26%/64.84%, same83.886M decisions and256 assigned worlds.
  It uses factored X/Y actions,13-way categories per visible cell and signed
  ROI-to-player offsets; no hidden-target helper or new reward. All eight
  checkpoints and256 episode rows reproduce exactly in a fresh same-seed
  training. Training n=1, not paper proof. See
  [the complete state ablations](../../research/results/sprite-forage-state-controls-5060-20261015/README.md).
- [ ] Confirm the selected partial-state candidate across the fixed seven
  training replicas and new evaluation worlds before claiming robust gains.
  Keep losses/failures and separately test whether learned gaze beats fixed
  or random gaze. No automatic follow-up allocation is implied.
- [ ] Shrink pixel observations at their source: precompute8x8/4x4 RGB
  sprite caches and directly copy the17 selected small tiles. Keep the
  human viewer/full-board game unchanged. Adapt the CNN to the actual input
  dimensions; verify lookalike marks survive. **Do this after the initial
  action/observation ablations work.** Compare max versus average/mean
  reduction ahead of stepping; no per-frame runtime downsampling. This is
  not implemented yet.
- [x] Implement the first CNN that processes the eight neighboring pixel cells around
  the player and nine cells around a separate center selected by the agent's
  actions. The policy learns where to look. Avoid full-map convolution and
  full-map pixel work when producing these observations. Add a matching state
  mode that exposes state only in those same cells, with identical gaze actions
  and timing; no hidden-cell coordinate lists, grids or target helpers. Compare
  state versus pixels with the same restricted visibility. Preserve game rules,
  world RNG and original state/full-pixel modes. Full idea, action timing and
  comparison requirements: [agent-controlled spatial attention](../../research/AGENT_CONTROLLED_ATTENTION.md).
  The opt-in two-head build, both partial modes and both full controls now pass
  the bounded native canary. [Build/use and test scope](../../research/FORAGE_ATTENTION.md).
- [ ] Qualify/tune restricted-state learning, inspect gaze behavior, then compare
  pixel learning at matched budgets/seeds. Smokes are not learning evidence.
  First seed7 five-minute pilots are complete: partial state61.33% collection
  perf at301.8s; partial pixels5.47% at303.1s, falling to3.52% at377.1s.
  Full state remains97.66% at~11s. Pixel learning is poor and useful learned
  gaze is unproven. Preserve the decline; timestamps alone select the
  near-five-minute checkpoints, never scores. See
  [the full report](../../research/results/sprite-forage-attention-five-minute-5060-20261015/README.md).

## Current state results and cleanup candidates — October15

Current observation version is v6, with v5 game rules: neutral objects give zero
reward and disappear on pickup;
perf is clamp((apples + grape - bad pickups) /4,0,1). The v4 reward/metric notes
below are historical, not the current contract. State layout2 currently uses
141 values: row-ordered object coordinates/playerXY and a local5x5 grid.

Completed six-case, three-seed matched ablation:
`research/results/sprite-forage-observation-ablation-20261015/README.md`.
Full observations reach99.61% mean final perf; removing the local grid falls
to58.66%. PlayerXY + nearest apple + grape + local grid also reaches99.61%.
Keeping all coordinates but removing sorting reaches99.48%.

- Retain the local grid. Its two5-wide loops read only25 cells and carry useful
  information under the tested recipe.
- Coordinate sorting removed at Kinvert's request; the complete object list and
  local grid remain. V6 identifies the changed semantics; old sorted weights
  have the same byte length but expect a different ordering.
- Optionally compare a compact33-input target-plus-local layout against the full
  baseline. This has not been trained: the ablation zeroed unused141-input slots
  to hold model size/initialization fixed. Shrinking needs fresh matched training.
- Profile observation construction separately if needed; whole training SPS also
  depends on the policy's movement/reset pattern. No automatic GPU follow-up.

## Apple/grape reward objective — implemented as v4

Kinvert clarifies that positive objects must not all have the same value. The
benchmark should test recognition of value as well as good/bad classification.
An ordinary apple should be highly valuable while its visually similar spotted
apple is penalized; berries/grapes can also be a high-value target. Preserve
the existing lookalike artwork and its spatial markings.

The implemented v4 defaults are:

| Category | Value |
|---|---:|
| Fatal mushroom | -1.0 |
| Penalty collectibles, including spotted apple | -0.5 |
| Empty/neutral | 0.0 |
| Pear, leaf, flower, beetle, ordinary mushroom | +0.5 |
| Apples | +1.0 |
| One grape cluster, ends the episode | +1.0 |

These are real values shared between state and pixels. `reward_good` is a common
scale for positives, preserving the half/full tier ordering. Penalty/death/step
controls remain numeric. Perf weights are independent of those learner rewards.
Trees block movement while empty cells do not: if both have scalar value0,
provide a separate wall mask. Keep the player position explicit too.

The grape contributes0.5 perf, all apples together0.5; each bad pickup loses
one apple's share. Collecting the last apple alone does not terminate, so the
policy can collect apples before choosing the grape exit. Other good items
give smaller rewards but do not increase this fixed objective. The historical
count-based diagnostic has been removed; see README for the current metrics.

The prior equal-positive/count-perf calibration is historical. Do not relabel
its checkpoints/results as current v4 learning; the historical launchers now
refuse new v4 sources. New state calibration and observation experiments remain
to be run under this revised fixed contract.

## State observation experiments — planned, not implemented

Compare these against the current116-float coordinate list, changing only the
observation scheme. Freeze rules, rewards, learner/core, decision budgets and
paired training/evaluation seeds. Input sizes change the encoder parameter
count and cost; record both rather than calling the resulting nets identical.
Keep the current encoding available and version incompatible checkpoints.

First candidate requested by Kinvert: a full20x15 scalar value grid plus
normalized player x/y. With a dedicated wall code this is302 floats. His newer
preference is the simple value palette above; using0 for both neutral/walls
requires a separate300-cell wall mask, making602 floats including player x/y.
The representation must distinguish ordinary/high-value positives. Observation
values stay fixed when learner reward scales are swept. This differs from
the earlier3,900-float identity-one-hot grid, which learned poorly.
The grape's terminal role must also remain identifiable: even though its reward
matches an apple, merging both into the same scalar code would hide which cell
ends the episode. Use a distinct grape code, goal mask or explicit goal coordinates.

Other useful candidates, in suggested test order:

1. Seven binary20x15 planes: ordinary good, apple, grape exit, nonfatal bad, fatal,
   walls and player (2,100 floats). Empty cells are implicit. This avoids arbitrary numeric
   category ordering and preserves the spatial layout. A flat state encoder
   does not gain convolutional structure merely from receiving grid planes.
2. Player-centered full-board category planes, so a neighboring item always
   appears at the same relative offset. A39x29 canvas covers every displacement
   of a20x15 board; distinguish positive tiers/the grape exit and bad/fatal/walls, with outside-
   board positions as blocked. Record its much larger size/cost. A smaller
   local crop alone hides distant objects and changes the information available.
3. Stable coordinate slots: keep each object's spawn slot for the episode,
   zeroing that slot on collection instead of compacting later objects into
   earlier slots. Same116-float budget at defaults, but requires explicit
   slot bookkeeping. Test whether reducing discontinuities helps learning.
4. A full coarse grid plus a small player-centered detail window. This keeps
   distant targets visible while making adjacent hazards explicit. Treat it
   as an engineered state representation, and measure the added input cost.

Optional diagnostics: a nearest-good direction/distance or path-distance field
can help distinguish navigation difficulty from learner problems. These add
task-specific processing, so label them privileged diagnostic observations,
not the default state baseline or evidence that pixels solve navigation.
Likewise, remaining episode time may help with the512-step cutoff, but adding
it changes observability; handle it as a separate experiment and explicitly
decide how the pixel policy receives equivalent timing information.

After freezing the revised item values, do not combine observation redesign with
further reward/game changes or assume any
candidate improves perf before matched runs demonstrate it. Keep all results,
including losses, and use fresh final evaluation seeds after development.

## Grayscale observations — low priority, not implemented

- [ ] Add an optional one-channel grayscale pixel mode; keep RGB as the default.
  Update pixel generation, observation sizing and CNN input handling together.
  Keep rules, rewards and world RNG unchanged. Verify lookalike sprites remain
  distinguishable, deterministic pixel/viewer agreement, and memory/SPS changes
  under matched recipes and fixed seeds. Removing color is a separate benchmark
  condition; fewer input channels do not imply a threefold whole-network speedup.

## Optional sprite animation — planned, not implemented

Add an INI-controlled option for selected sprites to use two or three frames.
For example, the wasp's wings could alternate positions; other suitable objects
could flutter or sway. Keep the static mode available as the control.

Animation should change appearance only: object identity, position, collision,
rewards, spawn RNG and perf remain identical. State observations stay unchanged.
The extra visual variation should challenge the pixel encoder without changing
the underlying task. Describe this as an intended research question, not an
assumed improvement in benchmark quality.

Derive frame selection deterministically from episode tick and object/cell
identity, without consuming world RNG. Cache the frames once, and copy only
changed tiles. Animation makes some tiles change even while the player waits;
account for that explicitly rather than relying only on the existing two-player-
cell update. The policy pixels and human viewer must display the same frame.

When implemented, verify repeatable frame sequences, unchanged state-mode
trajectories, dirty/full redraw equality and viewer/policy agreement. Record
animation settings in every resolved training/evaluation recipe, keep them
fixed across compared policies, and measure raster cost and training SPS.
Treat static-to-animated transfer as a separate condition. Preserve the
current static benchmark and its receipts.
