# Sprite Forage

This environment benchmarks learning from state versus pixels, with identical
rules and paired worlds. It need not reach100% perf to be useful: reproducible
learning and time curves are the main comparison. Planned extensions, including
optional two/three-frame sprite animation, are in [TODO.md](TODO.md).

State/pixel game determinism, actual state-policy repeatability and comparison
rules are documented in `research/SPRITE_FORAGE_BENCHMARK.md`.
Current version is `sprite-forage-state-v6`: apples and a single grape exit
give positive reward; neutral collectibles disappear without item reward.
V6 removes nearest-object sorting from state layout2; game rules and pixels
are unchanged from v5. Old sorted layout2 checkpoints have the same size but
different observation semantics. Use weights trained with the unsorted layout.
Earlier v4 and equal-positive/count-perf calibration results
are historical and do not describe this revised task.

One agent on a20x15 board. In `config/sprite_forage.ini`, select:

| `env.observation_mode` | Observation | Native policy encoder |
|---:|---|---|
|0 (default)|Row-ordered labeled coordinates and a local5x5 grid:141 floats|Standard PufferLib linear encoder|
|1|480x640x3 RGB, flattened to921,600 floats|Nature CNN:32/64/64 channels, kernels8/4/3, strides4/2/1, then linear|

Both use float32 observations. `env.state_layout` selects the state encoding:

- `0`:116-float row-ordered coordinates, retained as a control.
- `1`:300-float scalar grid; player1, apple0.75, grape0.5, neutral collectible0.25,
  empty0, penalty-0.25, fatal-0.5, wall-1. No separate player x/y.
- `2` (default): coordinates in board row order within each type, plus a
  player-centered5x5 grid using those same scalar codes, including player1.

Scalar codes identify categories; they are not reward values. Layouts0/2
start with player `(x / 20, y / 15)`.
Then each object slot contains `(present, (x - player_x) / 20, (y - player_y) / 15)`.
Slots group the ten collectible types in `sprites.h` order, with
`copies_per_type` slots each, then `trees` interior-tree slots. Within each group,
objects are row-ordered in both layouts0/2.
Unused slots are zero. The grape group
has only one present object; its other reserved slots are zero. The constant
border/ground are implicit; every interior object and the player are represented.
Coordinate size is `2 + 3 * (10 * copies_per_type + trees)`, plus25 for layout2,
with values in[-1,1]. Layout1 always has300 floats.
Pixels are HWC in[0,1]. Neither includes a shortest path, recommended action, reward map
or hidden objects. Changing mode changes only observations and the encoder.
Spawns, rules, rewards, actions and episode endings are shared.
The earlier3,900-float one-hot state is retained in historical experiment source
archives; its checkpoints are incompatible with the new coordinate encoder.

`env.seed` offsets the native environment-slot RNG. Use the same offset and slot
for paired worlds; change this offset for held-out evaluation worlds. `base.seed`
controls the learner and does not replace the explicit world seed. Never sweep
either seed. `representation` chooses one of five skins; it does not consume RNG.

Each32x32 cell contains16x16 art enlarged2x. Each requested skin is scaled and
composited once, then shared across environments. Unused skins are not built;
the fixed cache still reserves space for all five. Both modes show the same
scene in `puf_render`.
Headless simulation needs no window, GPU graphics context or image decoder.

## Rules

Actions:0 wait,1 up,2 down,3 left,4 right. Movement is one cell; trees block it.
Each reset scatters one grape, three copies of each other collectible, eight
interior trees and the player without overlaps. Nonlethal cells remain reachable
without passing through the grape exit: apples cannot require crossing the
grape or a fatal chokepoint. Nonfatal penalty objects can still be traversed.
Objects disappear on collection; there is no respawn until the next episode.

| Object | Reward | Effect |
|---|---:|---|
|Apple|`env.reward_good` (1)|Collect; earns apple perf credit|
|Grape cluster: exactly one|`env.reward_grape` (1)|Positive terminal/exit|
|Leaf, pear, flower, beetle, striped mushroom|`env.reward_other` (0)|Disappear; no item reward or perf credit|
|Spotted apple, wasp|`env.reward_bad` (-0.5)|Negative collectible|
|Spotted toadstool|`env.reward_death` (-1)|Death|

Every decision adds `env.step_cost` (-0.001). Reward controls remain configurable.
Episodes end on the grape, death or `env.max_steps` (512 by default).
Collecting the last apple does not end an episode. Same-step autoreset preserves
terminal reward. Defaults give three apples, one grape,15 neutral collectibles
and nine negative collectibles.

Perf is fixed independently of reward settings:

`perf = clamp((apples + grape - bad_pickups) / 4, 0, 1)`

These are per-episode counts; grape is1 on a grape exit and0 otherwise.

Each apple and the grape contributes0.25. An immediate grape exit scores0.25.
All three apples followed by the grape, with no bad pickups, scores1.
Each bad pickup loses0.25, including death items. The denominator stays4 even
if `copies_per_type` is changed; perf remains clamped to[0,1].
Timeout/death retain any remaining apple credit but give no grape credit.
Neutral collectibles give no item reward and do not increase this objective.
Perf is clipped per episode before averaging. `successes` counts grape exits,
including partial-score exits.

`score`/`episode_return` report actual reward return including step costs, and
can exceed1. PROTEIN maximizes `perf`, not reward return. The old
weighted prototype and equal-positive v2/v3 results have different metric/rule
contracts and must not be compared as unchanged experiments.

The completed v4 sweep learned weakly. Its retained report is in
`research/results/sprite-forage-v4-state-sweep-5060-20261014/README.md`.
The v5 diagnosis and new state-only sweep are documented in
`research/SPRITE_FORAGE_V5_STATE.md`. Historical launchers reject current rules.
Keep game versions and state layouts attached to all weights and results.
The completed three-seed observation ablation supports removing sorting:
unsorted coordinates plus local grid achieved99.48% mean perf versus99.61%
with sorting. See `research/results/sprite-forage-observation-ablation-20261015/`.

Drawings:0 original meadow,1 swapped global palette,2 mirrored sprites,
3 dense grass texture,4 imported Kenney Tiny Dungeon sprites. Drawing never
advances RNG or changes collisions/spawns/rewards. Representation4's explicit
object-role mapping is in `assets/SOURCES.md`. It is not assumed to preserve
human semantic identities such as apple→apple.

## Rendering cost

A simple initialization flag builds the shared immutable atlas once, during
serial `puf_init` before environment workers start. State-only human rendering
builds it lazily on the rendering thread. Each move copies the old
cell and new cell using32 contiguous row copies per tile. Wait/blocked movement
does not redraw. The first reset draws the entire frame. Later resets compare
the new tile IDs against a300-byte per-environment cache and copy only changed
cells, keeping unchanged ground and borders. There are no per-step image
loads, transforms, allocations or graphical rendering calls. The normal loop
still sends a full observation to the trainer; reduced raster work does not
establish a training-speed advantage.

At32x32 RGB float32, an ordinary move copies24,576 bytes across the two tiles;
waits/blocked moves copy0 pixel bytes. The first reset writes3,686,400 image
bytes; later resets write12,288 bytes per changed tile.
The row copy uses a tile origin and row offsets, with a constant384-byte copy
per row. There is no dirty mask, list or new allocation. Existing
dirty/full redraw tests cover movement/collection/reset; sentinel-image checks
also verify waits and blocked moves do not rewrite any pixels. Test-only copy
counters check two tiles per ordinary move, zero for unchanged resets, and
exactly the changed tiles for new worlds across all five skins. The skin is
fixed at initialization; explicit `compute_observations` still rebuilds the
whole image and synchronizes the cache.

October15 initial row-copy check against environment commit `49bb8f9f`: all six
ASan/UBSan suites pass. Three paired 5,000-step raster timings measure about
0.89M environment SPS before and 0.88M after; forced full redraw measures about
8.6K in both. The dirty path already existed in that commit. These millisecond
dirty-path samples establish no benefit from the row-address cleanup and are
not GPU training measurements. Raw receipts:
`build/sprite-forage-dirty-cells-20261015/`. This predates the reset tile cache.

The reset-cache follow-up measures928,636 →1,638,668 environment SPS in two
paired50,000-step native game/raster runs per version (about1.76x). This is
headless pixel generation, not CNN training SPS. Each run has35,879 ordinary
moves and278 resets; dirty and forced-full paths produce equal final worlds
and pixels. All six ASan/UBSan suites pass, including state/pixel parity across
five skins and fixed seeds. Copy counters also check the first reset writes
the full image. Sources, logs and timings:
`build/sprite-forage-reset-dirty-20261015/`.

Two original pairs (apple/spotted apple, striped/spotted mushroom) retain the
same silhouette and color counts. Their distinguishing signal is the spatial
arrangement of markings. All ten classes use recognizable, non-solid artwork.

## State throughput record — October15, RTX5060

| Source | Process SPS | Native adjusted-average SPS | Evaluation perf | Grape exits |
|---|---:|---:|---:|---:|
| Before metric/name/cache cleanup |742,328|782,101|97.65625%|100%|
| Reviewed cleanup |745,912|770,347|97.65625%|100%|

Two fresh timing repeats per version, before/after/after/before, each8,388,608
decisions with the same fixed seed7, H128/L1,141 state values,256 agents,
H64/MB2048 and previously tuned learner. Each version's first final checkpoint
gets32 sequential episodes at each development world420/1337, one slot and a
512-step cap. Independent training n=1. All four saved checkpoints per run
match byte-for-byte across all four runs. Process SPS changed+0.48%; native
adjusted SPS changed-1.50%. These short timings establish no speedup.
Process timing is monotonic and includes startup/checkpoint writes; the native
adjusted timer has a different scope and uses CLOCK_REALTIME.

Local raw receipts: `build/sprite-forage-cleanup-sps-20261015/`; the detailed
local report is `research/results/sprite-forage-cleanup-sps-5060-20261015/`.
Neither raw research packet is part of this environment-only commit. Keep
these state results separate from pixel raster or CNN training measurements.

## Pixel training pilot — October15, RTX5060

Built-in Nature CNN, random initialization, full640x480 RGB float32, H128/L1,
fixed training seed7, drawing0. One32,768-decision run completes in147.219s:
222.58 process SPS,223.38 native adjusted-average SPS,224.71 final-bin SPS.
All four scheduled checkpoints are finite and have distinct hashes;
34,991,136 parameters each. The changed-cell reset code is included.

Using16 serial episodes each at development world offsets420/1337, the same
evaluation settings give6.25% mean perf before training and9.375% afterward.
Final training-bin perf is0. This establishes execution and weight updates,
not convincing learning or convergence. This is the built-in Nature encoder,
not our Flex architecture.8 agents/H8/MB8 and starting LR.001 differ from the
much larger-batch tuned state run; these SPS numbers do not isolate pixel cost.
Raw evidence: `build/sprite-forage-pixel-train-5060-20261015/`; report:
`research/results/sprite-forage-pixels-5060-20261015/README.md`.

Our locked Flex quality CNN also completes the matched32,768-decision run:
89.842s,364.73 process SPS,366.85 native adjusted-average SPS,19,721,472
parameters. Mean eval perf is2.34375% untrained and4.6875% final. It runs
about1.64x Nature's process SPS in this one short paired setup, but its final
perf is lower than Nature's9.375%. Neither result establishes reliable
learning. All four Flex checkpoints pass size/finiteness checks and differ.
The learner, seed, environment, input size, rollout and evaluation settings
match Nature; only CNN coordinates/output paths differ. Report:
`research/results/sprite-forage-flex-5060-20261015/README.md`.

The requested roughly ten-minute fresh Flex run completes221,184 decisions
in612.087s (10m12s),361.36 process SPS,361.77 native adjusted-average SPS.
Same graph/seed/recipe, with the longer total budget and LR annealing schedule.
Mean final development eval perf is9.375% (world420:6.25%,1337:12.5%), versus
the reused2.34375% untrained control. The earlier short Flex run scored4.6875%.
This remains weak one-seed performance; no equal-budget Nature result exists.
All four checkpoints pass size/finiteness checks. Report:
`research/results/sprite-forage-flex-ten-minute-5060-20261015/README.md`.
The speed investigation finds99.69% of this full-resolution model's parameters
in its flatten-to64 projection, with a full Muon update every8 decisions.
Retained timing counters put most work in GPU training/model execution and
roughly0.15% in the environment; these are indicative downsampled buckets.
See `research/results/sprite-forage-flex-ten-minute-5060-20261015/SLOWDOWN.md`.

The depth3 spatial-head/batch64 follow-up completes the same221,184 decisions
in165.048s,1,340.12 process SPS (about3.71x faster), with1,294,112 parameters.
However, eval perf falls from its9.375% untrained control to3.125%. It is a
speed gain with a learning regression. All18 encoder numerical cases/72 CUDA
calls at B1/8/64 and the bounded native canaries pass; full learner math and
quality claims remain unqualified. Details:
`research/results/sprite-forage-speed-5060-20261015/README.md`.

## Build and inspect

The opt-in **attention research prototype** now supports full/partial state
and full/partial pixels through observation_mode and partial_observation.
The latter requires the separate two-action-head attention build. It uses a
shared tile CNN over the eight player-neighbor cells and nine agent-selected
gaze cells, with the same visibility for state mode. The normal one-head
baseline stays unchanged. [Interface/build instructions](../../research/FORAGE_ATTENTION.md)
and [actual four-mode5060 results](../../research/results/sprite-forage-attention-5060-20261015/README.md).
This is an implemented/tested prototype, not learning or publication acceptance.

This environment-only checkpoint keeps native core integration separate. The
visible checkout still needs the INI-selected observation-size/shared-Nature
support before its trainer build works. The measured state run above used a
frozen compatible research core; the standalone game tests work here directly.

Reuse the existing runtime and toolchain. Native trainer build:

```bash
mkdir -p build/sprite_forage
NVCC_ARCH=sm_120 bash build.sh sprite_forage build/sprite_forage/puffer --float
```

Use your usual PufferLib CUDA/NCCL setup and the architecture for your GPU.
The normal CUDA trainer supports both modes
without rebuilding; neither mode imports the experimental CNN/sweep files.
The fixed pixel encoder lives in `sprite_forage.cu`. There is no Python learner.

```bash
# Separate run IDs/checkpoint directories: state and pixel weights are incompatible.
build/sprite_forage/puffer train --env.observation_mode=0 --base.run_id=forage-state
build/sprite_forage/puffer train --env.observation_mode=1 --base.run_id=forage-pixels
```

These commands use the small-batch starting learner. A historical v3 state-only
calibration completed two20-run native PROTEIN sweeps, then three frozen
H128/L4 training replicas at134,217,728 decisions each on the RTX5060.
Final perf on two new world streams averages0.297681, versus0.133371 for the
untrained control. This is improved collection, not a solved task or pixel result.
These weights have the same length as v4 state policies but belong to a different
task; file length alone does not identify game rules. Do not reuse them as a v4
baseline or describe their score as current v4 perf. Historical recipe details are in
`research/SPRITE_FORAGE_STATE_SWEEP.md`; evidence is in
`research/results/sprite-forage-coordinates-5060-20261014/README.md`.
The main INI intentionally keeps the small batch. Keep16 agents initially for
pixels; a state-sized vector/batch can exhaust VRAM at640x480. For paired learning
comparisons keep learners, seeds and decision budgets identical. A state-only
throughput experiment can use a larger vector, but must record that difference.

Evaluate a final checkpoint using the **native CUDA** executable, the matching
mode/core size, and a held-out `env.seed`. Use one evaluation agent to give each
policy the same sequential episode stream without batch-completion bias:

```bash
build/sprite_forage/puffer eval PATH_TO_FINAL.bin --headless \
  --env.observation_mode=0 --env.seed=420 \
  --base.eval_agents=1 --base.eval_episodes=256
```

Run from the training runtime directory with its matching config; for the
measured H128/L4 recipe, the main small-batch H128/L1 INI cannot load those weights.

Repeat with a predeclared set of learner/world seeds and compare against initial
and random policies before claiming the state task is learned. Serial evaluation
is deliberately simple; this environment is not registered in the six-game
research evaluator or frozen search objective. The standalone CPU player supports
state checkpoints; pixel checkpoints require the native CUDA evaluator.

Game/observation correctness, without a CNN or GPU:

```bash
bash ocean/sprite_forage/tests/run.sh build/forage-tests-FRESH
```

This runs ASan/UBSan, state/pixel trajectory equivalence, complete coordinate identities,
compact-buffer guards, dirty/full redraw checks, viewer upload versus policy RGB,
and imported-asset checks. `tests/test_policy.cu` separately registers the real
native encoders at H64/H128/H256 on the host; it does not initialize or execute
neural weights. Successful host tests/builds are not evidence of learned quality.

Optional human play (WASD/arrows, Escape to close), without a model:

```bash
clang -std=c11 -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -O2 \
  -I. -Isrc -Iraylib-5.5_linux_amd64/include ocean/sprite_forage/play.c \
  raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lm -lpthread -ldl -lrt \
  -o build/sprite_forage/play
build/sprite_forage/play 0
```

Pass0..4 to choose a drawing and optionally0/1 as a second argument to choose
state/pixels. WASD/arrows control this manual helper. In the normal policy viewer,
hold Left Shift with WASD/arrows to take control.

Rebuild imported arrays after changing selected assets:

```bash
clang -std=c11 -O1 -Iraylib-5.5_linux_amd64/include \
  ocean/sprite_forage/import_sprites.c raylib-5.5_linux_amd64/lib/libraylib.a \
  -lGL -lm -lpthread -ldl -lrt -o build/sprite_forage/import-sprites
build/sprite_forage/import-sprites ocean/sprite_forage/imported_sprites.h \
  ocean/sprite_forage/assets/kenney/tile_{0114,0115,0116,0113,0101,0122,0123,0124,0120,0110,0004,0112}.png
```

The tool also supports `--scale INPUT OUTPUT.png INTEGER_SCALE` for offline
asset previews, and `--cell PIXELS OUTPUT.h ...` (default32, up to96) when
compiling a different-sized tile atlas. A12x16 source scales to24x32 at cell32
or72x96 at cell96, without stretching or a per-step transform. Keep proportions
and transparent padding; do not introduce
bilinear blur or per-step image decoding. Original artwork is in `sprites.h`.
