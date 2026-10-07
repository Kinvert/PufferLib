# Snake local episodic protocol v1

October 5, 2026. Rules identity `snake-local-episodic-v1`, INI
`env.rule_version=1`. This is a new native benchmark derived from PufferLib
Snake's movement rules, **not an unchanged stock-Snake reproduction**.
Original `ocean/snake/snake.h` and its configuration remain untouched.
Only environment/raster checks and compilation are authorized now; both GPU
execution holds and encoder-5 gates remain.

## Frozen game definition

One agent, four absolute actions: up/down/left/right. One grid move per
decision. A five-cell wall margin surrounds a rectangular playable interior;
default total board 26x26, interior 16x16. A fresh snake starts at length one
and randomly placed food occupies distinct empty cells. There are no other
agents or corpses. Stock's neck reversal behavior is retained: requesting the
own neck reverses the requested direction to continue away from the neck.
Walls and any occupied body cell, including the currently occupied tail, kill
the snake before movement. This follows the stock collision check.

Food gives the declared food reward and grows the snake up to
`max_snake_length-1`, the stock circular-buffer limit. At that limit it still
gives reward and replaces food, while the tail advances. Normal moves give
zero reward; collision gives the declared death reward. A declared finite
`max_steps` is part of this game's horizon, not an evaluation-only timeout.
Death takes precedence if it occurs on that decision; otherwise the horizon
ends after movement/reward/food replacement without an extra penalty.
Both endings emit a terminal and automatically reset the game. The ending
reward/terminal and explicit ending counters survive that reset.

Metrics: ending snake length (`score`), clipped length/120 (`perf`, retained
stock scale), collected food, episode return and decision length. These are
not win rates. Death/horizon counts distinguish the endings. The arbitrary
120 normalizer is descriptive, not a solved-task threshold. Evaluation must
eventually allocate exact episode identities and retain both end reasons.

## Determinism and safety

Each native environment owns a uint32 RNG, initially its native slot seed.
Portable LCG recurrence is `rng = 1664525*rng + 1013904223` modulo 2^32;
zero is a valid seed. Bounded choices use rejection to avoid modulo bias.
Spawns select a rank among all current empty cells, traversed in row-major
order. This is deliberately different from stock global `rand()` rejection
placement. Work order and pixel generation must not consume another game's
RNG. Appearance assignment has its own existing hash/seed.

Every reset clears the grid, ring slots and live episode counters before
placing walls, snake and food. Slots cannot inherit food/body/corpses from a
previous episode. Game arrays allocate only at initialization; stepping and
resetting allocate nothing. The ring capacity is at most playable cells minus
food plus one, guaranteeing space to replace food even at maximum live length.
Invalid geometry, population, capacity, vision, rule version, horizon, reward
or appearance settings fail before allocation. Invalid actions fail rather
than becoming no-ops or reading out of bounds.

## Equal local information

Both `snakebench` (state control) and `snakecnn` (pixels) use this same game
implementation and a centered 11x11 local crop, vision five. There is no
whole-board policy observation, global food direction, length, timestep, RNG
or episode-history channel. The head is always the center crop cell.

State retains the stock eight-category one-hot layout, float32, 11x11x8;
only empty/food/wall/own-body categories occur in this one-agent protocol.
Pixels are float32 CHW 1x36x44. Each crop tile occupies a 3x3 cell on a 33x33
canvas, at x=5/y=1. Empty=0, food=.25, wall=.5, body=.75; the known center
head=1. No screenshot, graphics context or Raylib rendering is needed.
Distinguishing the fixed center head adds no location information. Rasters
cannot reveal objects outside the same state crop.

| Representation | Pixel preset |
|---:|---|
| 0 | Filled 3x3 tiles |
| 1 | Five-pixel disks inside each 3x3 cell |
| 2 | Filled 2x2 tiles with one-pixel row/column gaps |
| 3 | Full-canvas grayscale inversion of preset 0 |
| 4 | Swap food/body intensities in preset 0 |
| 5 | Full-canvas horizontal reflection of preset 0 |

IDs 1/2 alter tile shapes; 3/4/5 are reversible transforms. Padding, image
shape and convolution FLOPs are fixed. Reflection leaves actions unchanged.
Mode zero fixes a preset; mode one hashes appearance seed/native slot once,
independently of game RNG, and retains that preset across resets. Do not let
an optimizer's easiest preset stand in for robustness.

## Required evidence before comparisons

Independent reset/spawn fixtures and array-based movement reference; ring
wrap, neck reversal, tail/body/wall collision, growth limit and food capacity;
dirty-reset and ending-reward/terminal/counter preservation; slot/work-order
RNG independence; state/pixel identical game trajectories; literal all-preset
pixels, overwrite/guard/boundary/hidden-state checks; repeat and ASan/UBSan.
Then native float32 compilation with shared encoders. GPU math, actual
train/reload/recurrent-reset behavior, exact assigned-episode evaluation and
recipe/horizon calibration remain distinct later gates. No comparison or
SOTA claim follows from host tests or build success.
