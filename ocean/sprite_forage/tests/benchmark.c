// Headless world/raster comparison, not CNN training SPS.
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <time.h>
#include "../sprite_forage.h"
#define OBS_SIZE FORAGE_PIXEL_OBS_SIZE

static double clock_seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9;
}
typedef struct { Env env; float* obs; float action, reward, terminal; } Fixture;
static void setup(Fixture* f) {
    memset(f, 0, sizeof(*f)); f->obs = (float*)calloc(OBS_SIZE, sizeof(float)); assert(f->obs);
    f->env.rng = 7; f->env.agents[0].observations = f->obs; f->env.agents[0].actions = &f->action;
    f->env.agents[0].rewards = &f->reward; f->env.agents[0].terminals = &f->terminal;
    Ini ini = {0}; puf_ini_load_file(&ini, "config/default.ini"); puf_ini_load_file(&ini, "config/sprite_forage.ini");
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    dict_set(kwargs, "observation_mode", 1);
    puf_init(&f->env, kwargs); puf_reset(&f->env); puf_ini_free(&ini);
}
int main(void) {
    const int steps = 50000; double elapsed[2]; long moved[2] = {0}, resets[2] = {0}; Fixture f[2];
    for (int mode = 0; mode < 2; mode++) {
        setup(&f[mode]); double start = clock_seconds(); unsigned int sequence = 1337;
        for (int i = 0; i < steps; i++) {
            sequence = sequence * 1664525u + 1013904223u;
            f[mode].action = (sequence >> 16) % 5; int player = f[mode].env.player_idx;
            puf_step(&f[mode].env);
            resets[mode] += f[mode].terminal != 0;
            moved[mode] += !f[mode].terminal && f[mode].env.player_idx != player;
            if (mode) compute_observations(&f[mode].env);
            // Keep every observation write observable to the compiler.
            __asm__ __volatile__("" ::: "memory");
        }
        elapsed[mode] = clock_seconds() - start;
    }
    assert(f[0].env.player_idx == f[1].env.player_idx && f[0].env.rng == f[1].env.rng
        && !memcmp(f[0].env.grid, f[1].env.grid, FORAGE_AREA)
        && !memcmp(&f[0].env.log, &f[1].env.log, sizeof(Log))
        && !memcmp(f[0].obs, f[1].obs, OBS_SIZE * sizeof(float)));
    printf("{\"scope\":\"single-process CPU world/raster only; not training SPS\","
        "\"steps_per_mode\":%d,\"dirty_seconds\":%.9g,\"full_redraw_seconds\":%.9g,"
        "\"dirty_env_sps\":%.9g,\"full_redraw_env_sps\":%.9g,\"measured_ratio\":%.9g,"
        "\"moves\":%ld,\"resets\":%ld,\"final_world_pixels_equal\":true,\"gpu_used\":false}\n",
        steps,elapsed[0],elapsed[1],steps/elapsed[0],steps/elapsed[1],elapsed[1]/elapsed[0],moved[0],resets[0]);
    for (int i = 0; i < 2; i++) { puf_close(&f[i].env); free(f[i].obs); }
    return 0;
}
