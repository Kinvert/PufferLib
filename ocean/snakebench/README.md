# SnakeBench state control

Matched float32 11x11x8 local one-hot state for
[SnakeCNN's separately versioned episodic game](../snakecnn/README.md).
Both include the same `../snakecnn/game.h`; neither modifies original stock
Snake. State adds no global food direction, length, time or whole-board channel.
The default recurrent policy uses H128/L1 with the ordinary native state
encoder. This is an optional information/control comparison, not the target
for the compact CNN's performance claim.

Compilation and CPU game/observation checks pass. GPU training/reload and
exact assigned-episode evaluation remain pending under the current holds.
