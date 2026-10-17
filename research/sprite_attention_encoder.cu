// Reuse the native Forage tile encoder; only view dimensions differ.
// Research-only aliases keep one implementation of the CUDA math.
#define FORAGE_VIEW_TILES SPRITE_VIEW_TILES
#define FORAGE_TILE_PIXELS SPRITE_TILE_PIXELS
#define FORAGE_VIEW_PIXELS SPRITE_VIEW_PIXELS
#define FORAGE_VIEW_OBS SPRITE_VIEW_OBS
#include "forage_attention_encoder.cu"
#undef FORAGE_VIEW_TILES
#undef FORAGE_TILE_PIXELS
#undef FORAGE_VIEW_PIXELS
#undef FORAGE_VIEW_OBS
