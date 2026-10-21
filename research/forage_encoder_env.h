// Research build adapter. The PR environment and its observation/action code stay unchanged.
#define puf_observation_size forage_original_observation_size
#include "../ocean/sprite_forage/sprite_forage.h"
#undef puf_observation_size
#undef PUF_NATURE_ENCODER
#define OBS_SIZE FORAGE_PIXEL_OBS_SIZE
#define AE_FORAGE
