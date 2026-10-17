# Sprite Push

V2 pushes one wooden crate three cells onto a target rug. Direction, crate position
and starting player position vary; construction starts with room to stand behind
the crate and guarantees a short safe solution. The20x15 room has a solid border,
scattered obstacles and a real corner trap: a bad sideways push strands the crate.
Crates can be pushed but not pulled. A second crate is lethal on contact; its
sprite differs in only two8x8 source pixels, enlarged4x. It cannot be pushed.

A golem pursues the player on turns2,4,6,..., moving one cell along the largest
coordinate difference first (horizontal on ties). If blocked it tries the other
axis, then waits if both are blocked. Walls and either crate block it. Walking
into it or being caught adds a-1 death penalty, including on a turn the crate
reaches its goal: death takes precedence. Chasing consumes no RNG. Its start
is far from the constructed short solution; delayed play can still get caught.

Reaching the rug earns1. Steps cost.001, crate progress earns.1 per Manhattan
cell, and approaching the crate supplies.02 distance shaping. Perf is binary
completion before the128-decision limit. The36-float state contains player
position, relative real/fake crate, target and enemy coordinates, movement phase
and a5x5 local window. Pixels use the same grid at640x480. Each step updates at
most six cached tiles. Three deterministic cached floor textures remain stable
when uncovered; they are neutral in state and consume no gameplay RNG.

Original v1 policies/results describe the empty room and31 state inputs. Train
fresh for v2; its old100% state completion is not a result for these new hazards.

This is a constructed small pushing task, not a general Sokoban benchmark.
The fake crate starts farther from the player than the real crate; that is a
geometric cue, so solving it does not prove recognition of the two-pixel artwork
difference. The golem is greedy and can be blocked by walls. Future held-out
layout/placement variation should test these shortcuts separately from new art.

Set observation_mode0/1 in `config/sprite_push.ini`. See
[suite instructions](../sprite_common/README.md) for native build/train/eval/tests.
