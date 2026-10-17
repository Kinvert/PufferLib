// Export the exact cached RGB observation without a graphics context.
#include <assert.h>
#include GAME_HEADER

int main(int argc, char** argv) {
    assert(argc == 2);
    Ini ini = {0};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, GAME_CONFIG);
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    dict_set(kwargs, "observation_mode", 1);
    Env env = {.rng = 7};
    float* obs = calloc(PIXEL_OBS_SIZE, sizeof(float));
    float action = 0, reward = 0, terminal = 0;
    env.agents[0] = (Agent){.observations = obs, .actions = &action,
        .rewards = &reward, .terminals = &terminal};
    puf_init(&env, kwargs);
    puf_reset(&env);
    Color* pixels = malloc(OBS_WIDTH * OBS_HEIGHT * sizeof(Color));
    for (int i = 0; i < OBS_WIDTH * OBS_HEIGHT; i++) {
        pixels[i] = (Color){obs[3 * i] * 255 + .5f, obs[3 * i + 1] * 255 + .5f,
            obs[3 * i + 2] * 255 + .5f, 255};
    }
    Image image = {pixels, OBS_WIDTH, OBS_HEIGHT, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
    assert(ExportImage(image, argv[1]));
    free(pixels);
    free(obs);
    puf_close(&env);
    puf_ini_free(&ini);
}
