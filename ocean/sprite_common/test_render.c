#include <assert.h>
#define InitWindow test_InitWindow
#define SetTargetFPS test_SetTargetFPS
#define LoadTextureFromImage test_LoadTextureFromImage
#define UpdateTexture test_UpdateTexture
#define BeginDrawing test_BeginDrawing
#define DrawTexture test_DrawTexture
#define EndDrawing test_EndDrawing
#define UnloadTexture test_UnloadTexture
#define CloseWindow test_CloseWindow
#include GAME_HEADER

Color captured[OBS_WIDTH * OBS_HEIGHT];
void test_InitWindow(int w, int h, const char* title) { assert(w == OBS_WIDTH && h == OBS_HEIGHT); }
void test_SetTargetFPS(int fps) { assert(fps == PUF_STEPS_PER_SEC); }
Texture2D test_LoadTextureFromImage(Image image) { return (Texture2D){.id = 1}; }
void test_UpdateTexture(Texture2D texture, const void* pixels) { memcpy(captured, pixels, sizeof(captured)); }
void test_BeginDrawing(void) {}
void test_DrawTexture(Texture2D texture, int x, int y, Color tint) {}
void test_EndDrawing(void) {}
void test_UnloadTexture(Texture2D texture) {}
void test_CloseWindow(void) {}

int main(void) {
    Ini ini = {0};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, GAME_CONFIG);
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    for (int rep = 0; rep < 3; rep++) {
        for (int mode = 0; mode < 2; mode++) {
            dict_set(kwargs, "observation_mode", mode);
            dict_set(kwargs, "representation", rep);
            Env env = {.rng = 7};
            float action = 0, reward = 0, terminal = 0;
            float* obs = calloc(puf_observation_size(kwargs), sizeof(float));
            float* expected = calloc(PIXEL_OBS_SIZE, sizeof(float));
            env.agents[0] = (Agent){.observations = obs, .actions = &action,
                .rewards = &reward, .terminals = &terminal};
            puf_init(&env, kwargs);
            puf_reset(&env);
            puf_render(&env);
            unsigned int rng = env.rng;
            puf_render(&env);
            sprite_observe(expected, env.grid, env.player, rep, env.floor_seed, NULL);
            assert(env.rng == rng);
            for (int i = 0; i < OBS_WIDTH * OBS_HEIGHT; i++) {
                assert(captured[i].r == (unsigned char)(expected[3 * i] * 255 + .5f));
                assert(captured[i].g == (unsigned char)(expected[3 * i + 1] * 255 + .5f));
                assert(captured[i].b == (unsigned char)(expected[3 * i + 2] * 255 + .5f));
            }
            if (mode) {
                assert(!memcmp(obs, expected, PIXEL_OBS_SIZE * sizeof(float)));
            }
            puf_close(&env);
            free(obs);
            free(expected);
        }
    }
    puf_ini_free(&ini);
    puts("PASS: viewer upload matches policy pixels in both modes/all three skins");
}
