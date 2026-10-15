// Human play without a model. Both modes display the same scene.
#include <assert.h>
#include "sprite_forage.h"

int main(int argc, char** argv) {
    Ini ini = {0};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, "config/sprite_forage.ini");
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    if (argc > 1) {
        dict_set(kwargs, "representation", atoi(argv[1]));
    }
    if (argc > 2) {
        dict_set(kwargs, "observation_mode", atoi(argv[2]));
    }
    Env env = {0};
    float action = 0, reward = 0, terminal = 0;
    float* observation = (float*)calloc(puf_observation_size(kwargs), sizeof(float));
    assert(observation);
    env.rng = 7;
    env.agents[0].observations = observation;
    env.agents[0].actions = &action;
    env.agents[0].rewards = &reward;
    env.agents[0].terminals = &terminal;
    puf_init(&env, kwargs);
    puf_reset(&env);
    puf_render(&env);
    while (!WindowShouldClose()) {
        action = 0;
        if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
            action = 1;
        } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
            action = 2;
        } else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
            action = 3;
        } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
            action = 4;
        }
        puf_step(&env);
        puf_render(&env);
    }
    puf_close(&env);
    puf_ini_free(&ini);
    free(observation);
    return 0;
}
