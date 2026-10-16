// Human play without a model. Mode 2 shows the fixed-gaze ROI view (cell 0).
#include <assert.h>
#include "sprite_lanedodge.h"

int main(int argc, char** argv) {
    Ini ini = {0};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, "config/sprite_lanedodge.ini");
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    if (argc > 1) {
        dict_set(kwargs, "representation", atoi(argv[1]));
    }
    if (argc > 2) {
        dict_set(kwargs, "observation_mode", atoi(argv[2]));
    }
    Env env = {0};
    float actions[2] = {0};
    float reward = 0, terminal = 0;
    float* observation = (float*)calloc(puf_observation_size(kwargs), sizeof(float));
    assert(observation);
    env.rng = 7;
    env.agents[0].observations = observation;
    env.agents[0].actions = actions;
    env.agents[0].rewards = &reward;
    env.agents[0].terminals = &terminal;
    puf_init(&env, kwargs);
    puf_reset(&env);
    puf_render(&env);
    while (!WindowShouldClose()) {
        actions[0] = 0;
        if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
            actions[0] = 1;
        } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
            actions[0] = 2;
        }
        puf_step(&env);
        puf_render(&env);
    }
    puf_close(&env);
    puf_ini_free(&ini);
    free(observation);
    return 0;
}
