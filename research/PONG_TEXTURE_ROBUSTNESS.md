# Pong texture robustness preparation

October 6, 2026. Adds two native **information-preserving** pixel appearances
without changing Pong physics, opponent, actions, rewards, score rows or image
shape. CPU environment/raster/reference/ASan/UBSan checks and four normal native
float32 builds pass. Native host-start checks pass; **GPU learning/inference
remain unvalidated and both holds remain**.

## New fixed appearances

All original IDs 0–4 retain their implementations. New IDs:

- **5 — checker texture:** each court pixel's original class (background,
  left paddle, right paddle, ball) chooses a disjoint intensity band. The low/
  high value alternates with `(x+y) mod 2`.
- **6 — contour texture:** uses the same bands; high values indicate that a
  four-neighbor pixel has a different original class. Outside-court neighbors
  are background. Background pixels adjacent to sprites also receive the
  high background-band value. This retains filled sprite interiors and isn't
  a lossy outline-only rendering.

| Original court value | Meaning | New low/high |
|---:|---|---|
| 0 | Background | 0.125 / 0.1875 |
| 0.5 | Left paddle | 0.375 / 0.4375 |
| 0.75 | Right paddle | 0.625 / 0.6875 |
| 1 | Ball | 0.875 / 0.9375 |

Decode class index as `floor(4*pixel)`, then map indices 0/1/2/3 back to
0/0.5/0.75/1. This recovers every original court pixel, including clipping,
occlusion, thin sprites and background. The two score rows are copied
unchanged and excluded from texturing. The implementation works from the
already generated raster; it doesn't introduce state/velocity/episode-ID inputs.

One fixed 1,584-byte stack array holds classes and two court passes apply the
texture. No heap allocation, graphics context, random draw or system dependency
is added. Original appearances bypass this path. Encoder tensor/FLOP shapes
stay unchanged; extra observation work has **not been timed**, so do not claim
zero overhead or a speed advantage.

These test a spatially varying palette, local contours and high-frequency
background variation beyond the old uniform palette/reflection/inversion
panel. They aren't new games, natural-image generalization or an independent
proof of robustness. Retain worse learning curves too.

## Preserve historical deterministic mixing

Increasing a catalog count normally changes `hash(seed,slot) % count`. That
would silently change old mixed runs. Pong therefore adds:

```ini
[env]
representation = 0
representation_mode = 1
representation_seed = 12345
representation_mix_catalog = 0
```

Catalog **0** is the default legacy alphabet (IDs 0–4). Catalog **1** opts
into all seven IDs. Fixed mode 0 accepts IDs 0–6 independently of the mix
catalog; mixed legacy mode also validates its otherwise inactive fixed ID
against 0–4. Selection doesn't consume game RNG and stays fixed per slot/reset.
The original 16-slot assignment receipt at seed 12345 is unchanged. Shared
helper tests check 4,096 slots, round-trip catalog changes and all new-category
coverage. Other environments keep their existing helper/count semantics.

Record catalog alongside seed/mode/slot count and preserve it on reload. The
sweep report groups mixed fronts by **catalog and seed**, so the same numeric
seed with different alphabets is not silently pooled. Fixed-ID fronts remain
grouped by actual ID. Architecture fingerprints exclude appearance settings.

## Verification and next gates

`ocean/pongcnn/tests/run_all.sh FRESH_OUTPUT` refuses existing directories and
retains world traces for all seven IDs, both catalogs and seeds
0/12345/4294967295. Each panel matches the original 49,152-transition world/
reward/terminal trace. Independent corner/edge expectations, score rows,
dirty-buffer/RNG/velocity invariance, invertible decode and invalid controls
are checked under ASan/UBSan. The exact episode CPU suite checks world starts,
terminal/cap/rally counters and all seven drawings against original Pong.
These are environment tests, not CPU neural testing.

The exact adapter advertises its appearance count. Old binaries without that
field retain the legacy five-ID interpretation; new appearance suites require
a compatible binary before preparing/running. Existing suite/rule versions and
historical fixed-ID image hashes remain separate. GPU reset/reload/quota/
repeatability/cap calibration is still required for exact policy results.

Fourteen host/glue tests pass against both retained five-ID binaries and new
seven-ID binaries, including rejection of new suites on old builds. The
retained verifier checks 1,000 assigned starts per appearance across three
pixel build families with H512/L4, matching state worlds and independent tail
starts. All five historical full CSV manifests are byte-identical. These are
initialization receipts; no policy or 1,000-episode rollout ran. Twenty-one
sweep/sidecar checks and two complete multi-task preparation tests pass.

[Durable source, logs, raw CPU traces and host-start receipts](results/pongcnn/texture-preparation-20261006/README.md)
preserve the measured build state separately from later documentation edits.
An initial host-test attempt failed to load existing NCCL because the runtime
helper wasn't sourced; that failure remains archived. Using the documented
process-local helper fixed loading without system changes.

The fixed multi-task preparer now offers 32 appearances across five games:
10 Connect4 + 7 Pong + 4 Flappy + 5 Breakout + 6 Snake. One training seed's
all-appearance panel is 128 unexecuted jobs; two seeds give 256. It preserves
one architecture and matched within-condition learner/budget/seed allocation.
Small default preparation budgets are plumbing, not successful learning or
publication confirmation. No long sweep/campaign is authorized by these files.

Later October 6 addition: MazeCNN expands the **current** panel to six tasks/
38 drawings (24 default, 152 all-drawing jobs per seed). The 32-condition
texture packets above remain historical unexecuted five-task evidence; use
[the current multi-task plan](MULTI_ENV_ROBUSTNESS.md) and fresh source receipts.

Next: explicitly scheduled native exact/learning acceptance, then matched full
learning curves across the fixed panel and paired seeds. No GPU performance,
Pareto advantage or SOTA claim follows from these CPU checks.
