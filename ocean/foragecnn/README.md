# Sprite Forage

`forage-sprites-v1`: one agent,20x15 cells,640x480 RGB float32 NHWC observations.
Each32x32 cell contains original16x16 pixel artwork enlarged2x with nearest
neighbor sampling. Scaled/composited tiles are shared across environments.
The game draws directly into the policy observation; `puf_render` displays
that same buffer without adding privileged state.

## Rules

Actions:0 wait,1 up,2 down,3 left,4 right. Movement is one cell; trees block it.
Each reset scatters three copies of every object, eight interior trees and the
player without overlaps. All nonlethal cells remain connected, including after
placing lethal hazards, so a required collectible cannot be trapped behind a
fatal chokepoint. Objects are removed
when collected; there is no respawn until the next episode.

| Object | Reward | Effect |
|---|---:|---|
|Leaf|+0.05|Collect|
|Apple|+0.10|Collect|
|Pear|+0.15|Collect|
|Berries|+0.20|Collect|
|Flower|+0.25|Collect|
|Beetle|+0.30|Collect|
|Striped mushroom|+0.40|Collect|
|Spotted apple|-0.20|Negative collectible|
|Wasp|-0.50|Negative collectible|
|Spotted toadstool|-1.00|Death|

Every decision also costs0.001. Episodes end on death, all21 positive objects
collected, or512 decisions. Same-step autoreset preserves terminal reward and
the outcome receipt. `score`/`episode_return` include all rewards and costs;
`perf` is the positive reward value collected divided by the initial4.35.
This is an explicitly defined new benchmark, not an Atari score.

Drawings:0 original meadow,1 swapped global palette,2 mirrored sprites,
3 dense grass texture,4 imported Kenney Tiny Dungeon sprites. Drawing never
advances RNG or changes collisions/spawns/rewards. Representation4's explicit
object-role mapping is in `assets/SOURCES.md`. It is not assumed to preserve
human semantic identities such as apple→apple.

## Rendering cost

`pthread_once` builds a shared immutable atlas once. Each move copies the old
cell and new cell using32 contiguous row copies per tile. Wait/blocked movement
does not redraw. Reset redraws the entire frame. There are no per-step image
loads, transforms, allocations or graphical rendering calls. The normal loop
still sends a full observation to the trainer; reduced raster work does not
establish a training-speed advantage.

Two original pairs (apple/spotted apple, striped/spotted mushroom) retain the
same silhouette and color counts. Their distinguishing signal is the spatial
arrangement of markings. All ten classes use recognizable, non-solid artwork.

## Build and inspect

Reuse the existing runtime and toolchain. Native trainer build:

```bash
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash build.sh foragecnn build/foragecnn/puffer --float
```

Create the parent output directory first if needed. This compiles the normal
PufferLib CUDA trainer and selects existing Flex (encoder4); it does not train.
`config/foragecnn.ini` contains an uncalibrated small learner, not a published
recipe. Do not launch a long run or insert this game into frozen six-game
campaigns just because compilation succeeds. It has no dedicated `eval_exact`
policy adapter or accepted CNN numerical/learning/frontier result yet.

Game/raster correctness, without a CNN:

```bash
mkdir -p build/foragecnn
clang -std=c11 -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -O1 -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I. -Isrc -Iraylib-5.5_linux_amd64/include \
  ocean/foragecnn/tests/test_environment.c \
  raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lpthread -ldl -lrt -lm \
  -o build/foragecnn/test-environment
build/foragecnn/test-environment
```

Preview actual progressed frames through the shared gallery:

```bash
.venv/bin/python research/observation_gallery/make_gallery.py \
  --games foragecnn --out build/forage-gallery-FRESH
.venv/bin/python research/observation_gallery/audit_gallery.py \
  --gallery build/forage-gallery-FRESH
.venv/bin/python -m http.server 8875 --bind 127.0.0.1 \
  --directory build/forage-gallery-FRESH
```

Use a fresh output each time. The preview controller sees state to visit
interesting objects, but the exported tensor contains only native pixels.

Optional human play (WASD/arrows, Escape to close), without a model:

```bash
clang -std=c11 -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -O2 \
  -I. -Isrc -Iraylib-5.5_linux_amd64/include ocean/foragecnn/play.c \
  raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lm -lpthread -ldl -lrt \
  -o build/foragecnn/play
build/foragecnn/play 0
```

Pass0..4 to choose a drawing. The Raylib window displays the policy buffer;
it does not rebuild a different scene from state.

Rebuild imported arrays after changing selected assets:

```bash
clang -std=c11 -O1 -Iraylib-5.5_linux_amd64/include \
  ocean/foragecnn/import_sprites.c raylib-5.5_linux_amd64/lib/libraylib.a \
  -lGL -lm -lpthread -ldl -lrt -o build/foragecnn/import-sprites
build/foragecnn/import-sprites ocean/foragecnn/imported_sprites.h \
  ocean/foragecnn/assets/kenney/tile_{0114,0115,0116,0113,0101,0122,0123,0124,0120,0110,0004,0112}.png
```

The tool also supports `--scale INPUT OUTPUT.png INTEGER_SCALE` for offline
asset previews, and `--cell PIXELS OUTPUT.h ...` (default32, up to96) when
compiling a different-sized tile atlas. A12x16 source scales to24x32 at cell32
or72x96 at cell96, without stretching or a per-step transform. Keep proportions
and transparent padding; do not introduce
bilinear blur or per-step image decoding. Original artwork is in `sprites.h`.
