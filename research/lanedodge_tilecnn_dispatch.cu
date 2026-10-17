// Research dispatch for the lane-dodge tile-CNN build. The shared tile CNN
// handles the 18-tile agent view (mode 3); the Nature CNN handles full-map
// pixels (mode 1). Replaces research/sprite_encoders.cu in the opt-in build.
// Included only by the lane-dodge tile-CNN research build.
#define OBS_SIZE (OBS_HEIGHT * OBS_WIDTH * OBS_CHANNELS)
#include "../ocean/connect4cnn/cnn.cu"
#include "../ocean/connect4cnn/nature.cu"
#include "../ocean/connect4cnn/flex.cu"
#undef OBS_SIZE
#include "lanedodge_tilecnn_encoder.cu"

static void create_sprite_research_encoder(Encoder* enc, Dict* policy) {
    if (enc->in_dim == LD_VIEW_OBS) {
        create_lanedodge_tilecnn_encoder(enc);
    } else {
        DictItem* type = policy ? dict_find(policy, "encoder") : NULL;
        if (type && type->value == 4) {
            create_flex_encoder(enc, policy);
        } else {
            create_nature_encoder(enc);
        }
    }
}
