// Human inspection of the exact policy buffer, without a model.
#include "foragecnn.h"
int main(int argc, char** argv) {
    Env env = {0}; float action = 0, reward = 0, terminal = 0;
    float* observation = (float*)calloc(OBS_SIZE, sizeof(float)); assert(observation);
    env.rng = 7; env.agents[0].observations = observation; env.agents[0].actions = &action;
    env.agents[0].rewards = &reward; env.agents[0].terminals = &terminal;
    Ini ini = {0}; puf_ini_load_file(&ini, "config/default.ini"); puf_ini_load_file(&ini, "config/foragecnn.ini");
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    if (argc == 2) dict_set(kwargs, "representation", atoi(argv[1]));
    puf_init(&env, kwargs); puf_reset(&env); puf_render(&env);
    while (!WindowShouldClose()) {
        action = 0;
        if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) action = 1;
        else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) action = 2;
        else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) action = 3;
        else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) action = 4;
        puf_step(&env); puf_render(&env);
    }
    puf_close(&env); puf_ini_free(&ini); free(observation); return 0;
}
