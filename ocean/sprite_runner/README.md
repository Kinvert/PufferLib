# Sprite Runner

Current rules are v3: touching a cactus/boulder kills immediately. Missing a
beneficial object is an error but is not lethal. The world RNG advances once
per reset; a separate episode-local stream produces its fixed object schedule.
An early death does not shift the next assigned world's objects. Perf divides
correct decisions by the full configured item budget; unattempted items count
against it. Perfect completion and deaths are separate metrics. State feature7
is absolute item height above ground, normalized by the three-cell jump range.
Old v2 state weights share the input length but have different feature semantics.

A fixed-screen lizard runner meets stars, berries, cacti and boulders. Stay on
the ground to touch low stars/berries, jump to collect elevated ones, and jump
to miss cacti/boulders. V2 places beneficial objects at rows11,9 or8; hazards
remain at ground level. These are distinct
original sprites. Items move one cell per decision from column19 toward the
player at column3. Each item gets16 decisions;12 items form an episode. The
sequence and timing are fixed by the world RNG and independent of actions.

Actions are wait/jump. A jump follows integer vertical motion; it can only start
on the ground. Correct collect/avoid earns1, an error-1; perf is full-budget
correctness, while successes counts perfect episodes. Only hazard contact ends
the episode early. The seven state values describe player
x/y, vertical velocity, grounded status, item offset, good/bad identity and
absolute item height. Rendering and observation generation consume no world RNG.
The old six-input v1 policies/results remain frozen; train fresh for current v3.

Set observation_mode0/1 in `config/sprite_runner.ini` for state/640x480 RGB.
See [suite instructions](../sprite_common/README.md) for native commands/tests.
