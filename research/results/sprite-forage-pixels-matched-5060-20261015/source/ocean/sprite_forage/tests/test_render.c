// Exercise the real viewer's image writes with graphics calls intercepted.
// Compare its texture upload to actual policy pixels, without opening a window.
#include <assert.h>
#include <math.h>
#define InitWindow test_InitWindow
#define SetTargetFPS test_SetTargetFPS
#define LoadTextureFromImage test_LoadTextureFromImage
#define UpdateTexture test_UpdateTexture
#define BeginDrawing test_BeginDrawing
#define ClearBackground test_ClearBackground
#define DrawTexture test_DrawTexture
#define EndDrawing test_EndDrawing
#define UnloadTexture test_UnloadTexture
#define CloseWindow test_CloseWindow
#include "../sprite_forage.h"

Color captured[OBS_WIDTH * OBS_HEIGHT];
int uploads;
void test_InitWindow(int width, int height, const char* title) {
    assert(width == OBS_WIDTH && height == OBS_HEIGHT);
}
void test_SetTargetFPS(int fps) { assert(fps == PUF_STEPS_PER_SEC); }
Texture2D test_LoadTextureFromImage(Image image) {
    assert(image.width == OBS_WIDTH && image.height == OBS_HEIGHT);
    return (Texture2D){.id = 1};
}
void test_UpdateTexture(Texture2D texture, const void* pixels) {
    assert(texture.id == 1);
    memcpy(captured, pixels, sizeof(captured));
    uploads++;
}
void test_BeginDrawing(void) {}
void test_ClearBackground(Color color) {}
void test_DrawTexture(Texture2D texture, int x, int y, Color tint) {}
void test_EndDrawing(void) {}
void test_UnloadTexture(Texture2D texture) {}
void test_CloseWindow(void) {}

int main(void) {
    Ini ini = {0};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, "config/sprite_forage.ini");
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    for (int rep = 0; rep < FORAGE_REPRESENTATIONS; rep++) {
        Env envs[2] = {0};
        float *buffers[2], actions[2] = {0}, rewards[2] = {0}, terminals[2] = {0};
        for (int mode = 0; mode < 2; mode++) {
            dict_set(kwargs, "observation_mode", mode);
            dict_set(kwargs, "representation", rep);
            buffers[mode] = calloc(puf_observation_size(kwargs), sizeof(float));
            envs[mode].rng = 7;
            envs[mode].agents[0].observations = buffers[mode];
            envs[mode].agents[0].actions = &actions[mode];
            envs[mode].agents[0].rewards = &rewards[mode];
            envs[mode].agents[0].terminals = &terminals[mode];
            puf_init(&envs[mode], kwargs);
            puf_reset(&envs[mode]);
        }
        unsigned int sequence = 42;
        for (int step = 0; step < 100; step++) {
            sequence = sequence * 1664525u + 1013904223u;
            for (int mode = 0; mode < 2; mode++) {
                actions[mode] = (sequence >> 16) % 5;
                puf_step(&envs[mode]);
            }
        }
        for (int mode = 0; mode < 2; mode++) {
            unsigned int rng = envs[mode].rng;
            puf_render(&envs[mode]);
            puf_render(&envs[mode]);
            assert(envs[mode].rng == rng);
            for (int i = 0; i < OBS_WIDTH * OBS_HEIGHT; i++) {
                assert(captured[i].r == (unsigned char)lroundf(buffers[1][i * 3] * 255));
                assert(captured[i].g == (unsigned char)lroundf(buffers[1][i * 3 + 1] * 255));
                assert(captured[i].b == (unsigned char)lroundf(buffers[1][i * 3 + 2] * 255));
                assert(captured[i].a == 255);
            }
            puf_close(&envs[mode]);
            assert(envs[mode].client == NULL);
        }
        free(buffers[0]);
        free(buffers[1]);
    }
    assert(uploads == 20);
    puf_ini_free(&ini);
    puts("Viewer: both modes/all 5 skins exactly match policy RGB pixels; "
        "repeat rendering preserves RNG; 20 intercepted uploads PASS (no GPU/window)");
}
