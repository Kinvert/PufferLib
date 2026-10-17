# Imported sprites

Author/source: [Kenney, Tiny Dungeon](https://kenney.nl/assets/tiny-dungeon),
version1.0 (2022). License: CC0; verbatim `kenney/License.txt` is included.
Original tiles are16x16 RGBA. Only the12 selected PNGs are retained here.

Downloaded archive:
https://kenney.nl/media/pages/assets/tiny-dungeon/f8422efb44-1674742415/kenney_tiny-dungeon.zip

These assets form representation4. Numeric gameplay identities and rewards stay
fixed; this skin has its own visible objects rather than falsely labeling a
potion as a picture of an apple.

| Slot | PNG | Visible object | Reward |
|---:|---|---|---:|
|1|tile_0114.png|Green potion|+0.10|
|2|tile_0115.png|Red potion|+0.15|
|3|tile_0116.png|Blue potion|+0.20|
|4|tile_0113.png|Gray potion|+0.05|
|5|tile_0101.png|Shield|+0.25|
|6|tile_0122.png|Beetle|+0.30|
|7|tile_0123.png|Mushroom|+0.40|
|8|tile_0124.png|Rat|-0.20|
|9|tile_0120.png|Bat|-0.50|
|10|tile_0110.png|Demon|-1.00, lethal|
|11|tile_0004.png|Wall scenery|Blocks movement|
|12|tile_0112.png|Explorer|Player|

`../import_sprites.c` reads the selected files **offline** and emits
`../imported_sprites.h`. Pixel colors/alpha are preserved. The compiler chooses
the largest integer scale that fits the requested cell, centers horizontally
and aligns the feet to the bottom without stretching. Native16x16 tiles become
32x32 in this game; non-square sprites are supported too.
The runtime precomposes those already-scaled RGBA values with the ground once into the
same shared float32 cache used by the original artwork. No file decoding occurs
during training or observation updates. See the rebuild command in the game
README. PNG/source hashes are in `SHA256SUMS`.
