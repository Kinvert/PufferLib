# Sprite Maze

V3 removes the fixed-right-edge trap shortcut. Three distinct traps are chosen
from leaf passages throughout the maze, excluding start/goal. Removing a leaf
cannot cut their safe path; every generated maze still has a safe solution.
Tests check safe solvability and trap placement in both halves across100
assigned worlds. Original v2 trap-placement results remain frozen.

Navigate a scout through a branching woodland maze with dead ends and three
lethal potion traps. Growing-tree carving follows the approach in `ocean/maze`,
using this game's own deterministic RNG and a reset-only stack. It is a separate
task, not a drop-in visual reskin or a claim of parity with the original Maze.
Seven additional leaf passages guarantee enough dead ends. Traps
never cut the safe route to the gold. The board is20x15, pixels640x480, and the
limit512 decisions.
Success earns1, steps cost.001, and reset-only shortest-path distances supply
.02 potential shaping. Distances and planned actions are absent from observations.
Touching a trap ends the episode with-1. Perf is binary treasure completion.

The29-float state includes player/treasure relative coordinates and a5x5 local
occupancy window, with distinct labels for walls, traps and treasure. These are engineered semantic
state features corresponding to visible geometry. Every move uses constant-time
physics; carving and BFS run only at reset. Pixel movement updates only two cached tiles.
The original three-hedge v1 task,32-input policies and results remain frozen;
its100% completion does not describe this harder task. Train fresh for v2.
Current v3 also changes trap assignment; retain the matching version with weights.

Select state/pixels in `config/sprite_maze.ini`. See
[suite instructions](../sprite_common/README.md) for native commands/tests.
