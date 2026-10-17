# Sprite Key

V2 has two almost identical treasure chests in the left chamber. Touch the good
chest to obtain the key and open the central gate, then reach the jewel exit.
Touching the mimic chest kills the player with-1. Their silhouettes match; two
8x8 source pixels differ, enlarged4x like the rest of the artwork. Chest roles
and positions vary; the good chest is not always above or left of the mimic.
Gate/exit rows also vary. The20x15 game ends on success, death or192 decisions;
perf is binary exit completion. Construction guarantees a safe route.

Key pickup earns.25, exit1 and steps cost.001. Two reset-only distance fields
provide.02 progress shaping for the current stage; they are not observations.
The37 state values describe player/good-chest/mimic/gate/exit coordinates, key possession
and local5x5 tile categories. There is no nearest-object sorting. Pickup changes
the chest and gate sprites as well as the player position; pixel mode updates all
four affected tiles from its immutable cache.

Three cached floor textures are assigned by an episode seed and cell hash.
Texture generation consumes no gameplay RNG; a cell retains its texture when
uncovered. State labels all three as neutral floor. Old v1 policies/results
describe an exposed key without a mimic and use35 inputs; train fresh for v2.

Set observation_mode0/1 in `config/sprite_key.ini`. See
[suite instructions](../sprite_common/README.md) for training/evaluation/tests.
