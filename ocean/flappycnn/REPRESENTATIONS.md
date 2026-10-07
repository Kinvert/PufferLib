# FlappyCNN drawings

All IDs generate float32 `[1,36,44]` directly in the native observation buffer.
Game rules, actions, physics, rewards, spawns and game RNG are unchanged. Pipes
use value 0.5 and the bird uses 1 against a zero background, except inversion.

| `env.representation` | Drawing |
|---|---|
| 0 | Original filled pipes and rectangular bird |
| 1 | Horizontal reflection of ID 0 |
| 2 | Vertical reflection of ID 0 |
| 3 | Grayscale inversion of ID 0 |
| 4 | Rounded bird with filled pipes |
| 5 | Rectangular bird with outlined pipes |
| 6 | Rounded bird with outlined pipes |

IDs 0–3 retain their original raster implementation. IDs 4–6 are geometric
variations, not reversible transformations of ID 0. They change visible pixel
information without changing the collision geometry. They require separate
learning/transfer measurements; a simpler raster does not reduce encoder FLOPs.

The rounded bird projects the world-space circle to an ellipse under the
nonuniform viewport scale. A pixel is filled when its center lies inside that
ellipse. The visible center pixel is always filled, including when a tiny bird
misses every pixel center. Pipes are clipped first; outlines follow the boundary
of each visible rectangle, including the image edge. The bird is drawn last.
No hidden velocity, score or offscreen object labels enter the image.

`representation_mode=0` selects a fixed ID. Mode 1 assigns an ID once per native
slot from `representation_seed`, without consuming game RNG or changing the
assignment on reset. `representation_mix_catalog=0` preserves the original
four-ID mixture. Explicit value 1 opts into all seven IDs. It has no effect on
fixed-ID selection. Preserve mode, catalog, seed and slot count when reloading.

The shared robustness preparer explicitly chooses catalog 1 for mixed Flappy
panels. Evaluate each fixed ID separately; mixed assignment counts are not
necessarily equal. Historical packets retain their captured native catalog.

Exact evaluation keeps `flappy-fixed-suite-v1` restricted to IDs 0–3. New
seven-ID metadata produces `flappy-fixed-suite-v2` with explicit catalog size.
Reject stale binaries that cannot represent the requested catalog. Both suite
versions retain native terminal/crash behavior and exact assigned episode IDs.
GPU evaluator acceptance is still pending.

[Checks, receipts and current limits](../../research/FLAPPY_GEOMETRIC_ROBUSTNESS.md).
