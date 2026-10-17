// Temporary sprite-suite research callbacks: tile CNN or full-image Nature/Flex.
#define OBS_SIZE (OBS_HEIGHT * OBS_WIDTH * OBS_CHANNELS)
#include "../ocean/connect4cnn/cnn.cu"
#include "../ocean/connect4cnn/nature.cu"
#include "../ocean/connect4cnn/flex.cu"
#undef OBS_SIZE
#include "sprite_attention_encoder.cu"

static void create_sprite_research_encoder(Encoder* enc, Dict* policy) {
    if (enc->in_dim == SPRITE_VIEW_OBS) {
        create_forage_attention_encoder(enc);
        return;
    }
    DictItem* type = policy ? dict_find(policy, "encoder") : NULL;
    if (type && type->value == 4) {
        create_flex_encoder(enc, policy);
    } else {
        create_nature_encoder(enc);
    }
}
