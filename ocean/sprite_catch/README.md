# Sprite Catch

Current rules are v4: catching a skull kills immediately. Missing a crystal
loses its reward but does not kill. Each episode has its own object RNG stream;
the world RNG advances only on reset, so early deaths cannot shift subsequent
assigned worlds. Perf divides correct decisions by the full configured drop
budget, including drops not reached before death. Success requires completing
every drop correctly; deaths is a separate metric. Old v3 results/checkpoints
retain their nonlethal rules and must not be presented as v4 results.

Move a basket to catch falling crystals and avoid skulls. Objects fall one cell
every two decisions; the basket moves one cell per decision. There are24
decisions per drop, enough to cross the20-column board. Each episode has10
drops by default and240 decisions; object position/type sequences are independent
of actions. A correct catch/avoid earns+1, an incorrect decision-1. Crystal
approach shaping adds.02 times the decrease in horizontal distance. Skull
avoidance adds.02 times its increase. Backtracking cancels this potential
shaping; it does not accumulate reward from repeated left/right movement.

Perf is correct decisions/drops; successes counts perfect episodes. A96% perf
score means96% item accuracy, not96% perfect games. The six state values are
basket x, object x/y, good/bad identity, horizontal offset and fall-clock phase.
V3 adds skull-distance shaping after v1/v2 learners settled for missing a rare
boundary case. V2's extra boundary flags did not solve it across all seeds and
were removed. V3 restores v1's six observations; physics, spawns and perf are
unchanged. Preserve the reward version in every training receipt. V1 weights
can still be evaluated, but were trained under the older reward function.

The calibrated recipe uses the standard H128/L4 MinGRU and entropy annealing
from.03 to.00003, with learning rate.003/gamma.99. Core depth and entropy changed
together; this is a validated recipe, not an isolated claim about either choice.
For the fast state allocation use256 agents,H64,MB2048 and33,554,432 decisions.
Other suite games use L1; keep each game's core/learner fixed in future CNN
comparisons. The INI keeps8 agents/H8/MB8 so switching to pixels is safe.
The floor/one-cell basket matches the collision grid exactly.

Use `config/sprite_catch.ini`: observation_mode0 for state,1 for640x480 RGB.
See [suite instructions](../sprite_common/README.md) for native training,
evaluation, drawings, deterministic assignment and correctness checks.
