// Temporary adapter to the existing research callbacks; no new kernels.
// Keep OBS_SIZE local: sprite state observations are resolved from the INI.
#define OBS_SIZE (OBS_HEIGHT * OBS_WIDTH * OBS_CHANNELS)
#include "../ocean/connect4cnn/cnn.cu"
#include "../ocean/connect4cnn/nature.cu"
#include "../ocean/connect4cnn/flex.cu"
#if defined(C4_IMPALA_CNN) || defined(C4_IMPOOLA_CNN)
#include "../ocean/connect4cnn/impala.cu"
#endif
#undef OBS_SIZE

static void create_sprite_research_encoder(Encoder* enc, Dict* policy) {
    DictItem* type = policy ? dict_find(policy, "encoder") : NULL;
#if defined(C4_IMPALA_CNN) || defined(C4_IMPOOLA_CNN)
    int fallback = 0;
#else
    int fallback = 2;
#endif
    double selected = type ? type->value : fallback;
    if (selected == 2) {
        create_nature_encoder(enc);
    } else if (selected == 4) {
        create_flex_encoder(enc, policy);
#if defined(C4_IMPALA_CNN) || defined(C4_IMPOOLA_CNN)
    } else if (selected == 0) {
        create_connect4_encoder(enc);
#endif
    } else {
        assert(0 && "Sprite research supports Nature2, Flex4 and residual-family0 only");
    }
}
