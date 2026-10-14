# Maintaining the native-observation HTML gallery

The reusable implementation is `research/observation_gallery/`. The first gallery
was temporary under `/home/claude/cnn/build/observation-gallery-20261014`; do not
edit only that ignored copy and lose the implementation on another checkout.
The new gallery includes Sprite Forage without changing any frozen six-game
learning campaign. It is a visual inspection artifact, not training evidence.

## Create or update a gallery

Run from the desired research checkout, reusing the existing uv Python3.12 venv
with NumPy. No Torch, new embedding database, GPU, model weights or dependency
installation is needed. Existing Clang and Raylib's static library are reused
for native **game/raster** helpers; no CPU CNN runs.

```bash
.venv/bin/python research/observation_gallery/make_gallery.py \
  --out build/observation-gallery-FRESH
.venv/bin/python research/observation_gallery/audit_gallery.py \
  --gallery build/observation-gallery-FRESH
.venv/bin/python -m http.server 8875 --bind 127.0.0.1 \
  --directory build/observation-gallery-FRESH
```

Open `http://localhost:8875/index.html`. If the audit worktree reuses another
checkout's venv, pass that venv's Python path directly; do not resolve its
interpreter symlink. Choose an unused port if another gallery is still running.
Never kill someone else's server or overwrite a gallery somebody is inspecting.
The generator refuses an existing output directory. Use `--games foragecnn`
(or a list of game names) to regenerate a smaller selection quickly.

Each manifest records exact dimensions, mode, representation ID, seed, episode,
decision/tick, state/RNG receipt, native source/config hashes, binary hashes and
raw memory/image hashes. It saves three progressed moments per drawing/mode.
The frames are normalized **native float32 NHWC policy observations**, before
CNN execution. Snake/Maze retain their local crop. No Raylib screenshots or
debug-only overlays are substituted. A deterministic state-reading controller
only helps reach interesting states; it is not a trained policy or a source
of privileged channels in the exported image.

PNG displays nearest-rounded8-bit colors. `memory/*.f32.gz` retains the exact
little-endian float32 arrays losslessly. The generator proves its display-color
palette maps unambiguously back to the original floats; hovering shows those
values and their memory offsets. No smoothing is applied during canvas zoom.

## Add a game or drawing

1. Update `GAMES` in `make_gallery.py`: native game name, display title, actual
   H/W, three nonzero progress targets, and names in exact numeric drawing order.
   Declare `modes=['rgb']` for an RGB-only game; do not fabricate a historical
   grayscale input. Legacy names can differ from modern names.
2. In `capture.c`, add a native controller and a pointer-independent world receipt
   to `action_for`/`describe_state` if needed. Use ordinary `puf_init`, `puf_reset`
   and `puf_step`; do not plant objects or fabricate a board for a screenshot.
   The Maze capture chooses quarters of the actual route length instead of
   pretending the spawn neighborhood is mid-episode.
3. Initialize/replay **each drawing separately from seed7**. Do not simply change
   `env->representation`: Connect4 caches glyph masks during `puf_init`, and new
   games can have similar caches. This capture seed has no effect on frozen
   training/evaluation seeds or the optimizer.
4. Keep the exact-byte repeated-write assertion. The generator also repeats
   drawing0 in a fresh process for each mode and checks matching world/RNG/
   progress receipts across all drawings/formats. A mismatch stops generation.
5. Run the raw/PNG audit. It checks every frame's shape, SHA256, PNG CRCs and
   conversion from the saved exact floats, uniqueness/coverage and world pairing.
   If a new geometry changes visibility or gameplay, label/version it separately
   rather than weakening the pairing assertion.

`gallery.html` contains only the UI/template. Change controls there and regenerate
a fresh HTML; the manifest is embedded, while PNGs and compressed buffers remain
small separate files. Retain incomplete outputs as failures until understood.
Compiled snapshot helpers are removed after successful capture; no large array,
checkpoint or browser profile archive is required. Do not refresh LanceDB or
recreate retired GPU evidence for this task.

## Browser verification

`check_browser.cjs` uses an existing Chrome DevTools endpoint and the host's
existing Node `ws` module. It does not install packages. Start a fresh temporary
Chrome headless profile with `--disable-gpu --no-sandbox
--remote-debugging-port=9234`, then, for a full gallery containing Connect4:

```bash
node research/observation_gallery/check_browser.cjs \
  "$PWD/build/observation-gallery-FRESH" \
  http://127.0.0.1:8875/index.html 9234
```

The check loads every PNG and exercises actual controls for all games/modes,
variant navigation, zoom, moment selection, keyboard navigation, raw-float pixel
inspection and RGB/history comparison. It saves `browser-check.json` and a
preview PNG, then closes its own browser. Remove only that browser's temporary
profile afterward. Leave the HTTP server alive while the owner is inspecting.
On this WSL workstation, Windows can open the URL with PowerShell
`Start-Process 'http://localhost:8875/index.html'` when a terminal file link fails.

## Artwork and speed

See `SPRITE_SIZES_AND_COMPLEXITY.md` and `ocean/foragecnn/README.md` for dimensions
and asset import. Decode sprite files and scale/composite them **offline or once
at initialization**, preserving proportions. The gameplay path copies cached
pixel rows directly into the observation buffer, never PNG files or a Raylib
framebuffer. Profile raster/transport/training separately; a reduction in raster
writes is not an end-to-end SPS result. Preserve attribution/license/source
hashes for imported assets and check downloaded bytes really are images.
