# Sprite Platform

V2 moves a robot across two randomly placed one/two-cell gaps, a raised middle
section and a higher final platform. Ground thickness varies from two to five
tiles, and the final rise is one or two tiles. The board stays20x15. Gravity and
jumping use integer cells and swept vertical collision; there is no scrolling
or floating-point physics. Touching the visible flag wins; flying over its
column does not. Falling into a gap loses.
Timeout is128 decisions. Forward progress earns.05 per column, each step costs
.001, reaching the flag adds1, falling loses1. Perf is binary completion.

Forty-one state values describe player x/y, vertical velocity, grounded status,
horizontal/vertical goal offsets and a7x5 local window. Pixel mode shows the same robot,
gaps and flag at640x480. Velocity is explicit in state; a pixel recurrent policy
must infer motion from its observation history. This is an intentional state
reference, not an information-equivalent feedforward comparison.

Old v1 checkpoints describe the flat-ground task and use40 state inputs. Its
reported100% state completion is historical; no learned CNN quality was measured.
Use fresh training for v2. The original trained version remains frozen under
`build/sprite-night-20261015/` for comparisons and recordings.

Set observation_mode in `config/sprite_platform.ini`. See
[suite instructions](../sprite_common/README.md) for build/train/eval/tests.
