# Cell sizes, retro sprites and the first recognition environment

These are the current realistic-image observation dimensions, not the historical
36x44 grayscale versions. Check native rasters, not the approximate web thumbnail.

| Game | Observation width x height | Cell or object size in observation pixels |
|---|---|---|
| Connect4 | 672x576 RGB | 7x6 board; each cell96x96. Piece sizes vary inside cells. |
| Snake | 616x616 RGB | 11x11 local crop; each cell56x56. |
| Maze | 616x616 RGB | 11x11 local crop; each cell56x56. |
| Breakout | 576x330 RGB | No movement grid; stock bricks32x12, ball32x32. |
| Flappy | 420x640 RGB | Continuous geometry; bird roughly28x28, pipe width58. |
| Pong | 620x640 RGB | Continuous geometry; ball roughly32x31 after the declared margins/HUD. |
| New Sprite Forage | 640x480 RGB | 20x15 grid; cell32x32; original16x16 art enlarged2x. |

The first six sizes derive from `image_v1.h`, configs and the frozen image
contract. Old Connect4 variants have much smaller occupied footprints; do not
confuse those with the new96x96 cells.

## Real sprite references

The extracted **small Mario standing image is12x16**, not a full16x16 drawing.
The public rip lists Super Mario16x32, Goomba16x16, Koopa16x24, coin10x14 and
the brick block16x16. These are extracted image bounds; poses and transparent
padding differ. [Original SMB sprite images and dimensions](https://www.nesmaps.com/maps/SuperMarioBrothers/sprites/SuperMarioBrothersSprites.html).

The original Zelda rip lists Link15x16 and his sword image16x27.
[Zelda sprite images](https://www.nesmaps.com/maps/Zelda/sprites/ZeldaSprites.html).
Hardware sprites are smaller pieces from which a character is assembled:
the NES supports8x8 or8x16 units. A hardware unit is not the character's visible
bounding box. [NESdev hardware-size reference](https://www.nesdev.org/wiki/Sprite_size).

Integer nearest-neighbor scaling preserves these shapes. Mario12x16 at6x is
72x96, centered horizontally in a96x96 cell. At2x it is24x32 and fits the new
game's32x32 cell. Enlarging a sprite adds display pixels, not new visual
information; internal edges, limbs, markings and textures must come from the art.

Downloaded size references are retained locally under
`build/retro-sprite-reference/`: Mario standing PNG12x16 and Goomba GIF16x16,
plus nearest-neighbor6x previews. One mushroom URL returned HTML despite
HTTP200; that response is not accepted as an image. The supplied DeviantArt
page returned403 and the Brave proxy was inaccessible to the browser tool.
Neither is silently substituted as verified artwork.

For reusable game assets, Kenney's author pages specify16x16 tiles and CC0:
[Tiny Dungeon](https://kenney.nl/assets/tiny-dungeon),
[Tiny Town](https://kenney.nl/assets/tiny-town).
The first environment includes12 selected Tiny Dungeon PNGs, the original
license, source mapping and their generated C arrays under `ocean/foragecnn`.

## First implemented environment

Read `ocean/foragecnn/README.md`. Ten score-bearing object types are scattered
without overlaps; the player moves through the grid, collects good objects,
can consume negative ones, and dies on a lethal toadstool. Trees block movement.
Every reset places all ten types and keeps the walkable terrain connected.

The original sprite set has stems, eyes, legs, veins, petals, cap stripes and
spots. Positive/spotted apples and striped/lethal mushrooms have identical
silhouettes and glyph histograms, but different texture placement. A classifier
using only their overall color counts cannot distinguish those pairs.
Representation4 uses imported Kenney loot/characters; its role mapping is
explicit and is a separate visual skin, not a claim of semantic transfer.

All scaling, alpha composition and float normalization happen once in a shared
immutable tile atlas. Each32x32 RGB tile contains12,288 bytes. A normal move
copies two tiles row by row:24,576 bytes, versus3,686,400 for a full640x480 RGB
float32 redraw. This is a150x reduction in observation bytes written for that
move, not a claim of150x training SPS. Reset still draws the complete frame;
the trainer still transports its complete observation. No PNG decoder, RNG,
allocation or Raylib graphics context runs in the normal drawing path.

Keep the existing six-game calibrated campaigns frozen. This new game is not
silently inserted into their optimizer objective or used to qualify any CNN.
The first delivery needs game/raster correctness and visual inspection; useful
learning, GPU numerical checks and fair training comparisons remain separate.
