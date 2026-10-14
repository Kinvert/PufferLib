# Sprite Forage implementation and visual inspection

G240, October14. Temporary Kinvert research; no new GPU policy execution,
training result, Pareto gain or SOTA claim. Existing six-game discovery inputs
remain unchanged. The only trainer-core change appends `PUFFER_FORAGECNN` to
the two existing pixel-encoder dispatch conditions in `src/ocean.cu`.

## Implemented

- Native20x15 foraging game,640x480 RGB,32x32 cells. Ten object classes with
  different rewards, one lethal class, trees and five visual skins. Reset
  guarantees every class appears and nonlethal cells remain connected.
- Original16x16 pixel-art objects plus12 author-distributed Kenney Tiny Dungeon
  CC0 sprites. Source PNGs, license, mappings/hashes and generated C arrays are
  retained in `ocean/foragecnn`. No copyrighted Nintendo sprite bytes are in
  that game/default asset pack or this committed report.
- Offline C image import chooses the largest integer scale fitting a cell,
  preserves source aspect ratio/RGBA and pads transparently. Shared native
  float32 tiles are built once. Each ordinary move redraws two cells via row
  memcpy; no image decoder, allocation, scaling or graphics context per step.
- Reusable observation-gallery source and maintenance instructions are now
  tracked under `research/observation_gallery/` and `research/OBSERVATION_GALLERY.md`.

## Actual checks

`test_environment.c` passes under Clang ASan/UBSan. It checks all ten reward
classes, lethal/success/horizon autoreset receipts, blocked movement, spawn
coverage, reachable maps across32 seeds, threaded atlas initialization, all five
skin-equivalent worlds,1000 shared random actions with unrelated libc RNG
interference, guards/range and dirty-update pixels versus a full native redraw.
The apple and mushroom lookalikes have equal glyph/color counts but different
spatial texture arrays. This does not certify CNN learning or mathematical
correctness of the learner/encoder.

`test_imported.c` compares every generated RGBA pixel against the decoded source
images, including scale and padding. Both the bundled12-sprite32-cell pack and
a separate local96-cell reference pack pass. The latter uses the actual
NES small Mario12x16 PNG and Goomba16x16 GIF as two inputs: Mario's image is
scaled6x to72x96 in a96x96 cell. These Nintendo reference bytes/generated pack
remain ignored under `build/retro-sprite-reference`, not in the bundled game.
See `research/SPRITE_SIZES_AND_COMPLEXITY.md` for public source URLs and sizes.

The normal public `build.sh foragecnn ... --float` completes using existing
CUDA12.8 and sm120, without GPU discovery/execution or dependency changes.
The final `puffer-current` build has before/after SHA256 matches for the owned
dispatch/header/config sources. The manual-play helper also compiles; no
Raylib window was launched for the headless validation. Exact-policy evaluator,
GPU numerical checks and useful-learning acceptance remain unimplemented or
unexecuted for this new game, as documented in its README.

Fresh full-gallery export/audit passes7 games,46 drawings,261 PNGs and261
lossless float32 buffers. All21 game/moment world receipts are paired across
drawings/formats;13 drawing-zero mode/game pairs also match fresh processes.
Actual Chrome headless navigation verifies every image, all game/mode/variant
controls, zoom, moments, keyboard, pixel inspector and RGB/history comparison.
The temporary gallery is `/home/claude/cnn/build/observation-gallery-20261014-current`,
served on localhost8875; the older six-game gallery remains separate on8874.
Small exact buffers, manifests, browser proof and screenshot remain with the
gallery. It is not model/episode inference validation.

## Raster-only timing diagnostic

One native5000-step paired game/raster run reports dirty updates0.004903699s
and an additional full redraw on every decision0.604314318s (ratio123.24).
Both modes perform3614 moves and19 autoresets, ending with identical world,
logs, RNG and full observation bytes. Initialization is outside the clocks.
This was a short, single-process CPU diagnostic during development, including
other host build activity; its very short dirty interval is noisy. It is
**not GPU training SPS, an isolated publication benchmark or an end-to-end
speedup claim**. The structural bytes-written saving for one two-cell move is
24,576 versus3,686,400 (150x); full observation transport still occurs.

Local commands/logs/binaries are under `build/foragecnn-development-20261014`.
`ocean/foragecnn/README.md` supplies clean build, test, play, asset-import and
gallery commands for future agents. No long run or automatic insertion into
the frozen PROTEIN campaign was launched.
