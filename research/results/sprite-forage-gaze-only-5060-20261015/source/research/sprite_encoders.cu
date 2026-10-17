// Included only by the opt-in FORAGE_ATTENTION research build.
#define OBS_SIZE FORAGE_PIXEL_OBS_SIZE
#include "../ocean/connect4cnn/cnn.cu"
#include "../ocean/connect4cnn/nature.cu"
#include "../ocean/connect4cnn/flex.cu"
#undef OBS_SIZE
#include "forage_attention_encoder.cu"

static void create_sprite_research_encoder(Encoder* enc, Dict* policy) {
    if (enc->in_dim == FORAGE_VIEW_OBS) {
        create_forage_attention_encoder(enc);
    } else if (dict_get(policy, "encoder") == 4) {
        create_flex_encoder(enc, policy);
    } else {
        create_nature_encoder(enc);
    }
}
