// Scalar game/state trace for checking visual-only changes. No neural inference.
#include <assert.h>
#include GAME_HEADER

int main(int argc, char** argv) {
    assert(argc == 2);
    FILE* out = fopen(argv[1], "wb");
    assert(out);
    Ini ini = {0};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, GAME_CONFIG);
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    dict_set(kwargs, "observation_mode", 0);
    unsigned int seeds[] = {7, 42, 69};
    int actions[] = ACT_SIZES;
    for (int i = 0; i < 3; i++) {
        Env env = {.rng = seeds[i]};
        float obs[STATE_OBS_SIZE], action = 0, reward = 0, terminal = 0;
        env.agents[0] = (Agent){.observations = obs, .actions = &action,
            .rewards = &reward, .terminals = &terminal};
        puf_init(&env, kwargs);
        puf_reset(&env);
        unsigned int sequence = 420;
        for (int step = 0; step < 2048; step++) {
            sequence = sequence * 1664525u + 1013904223u;
            action = (sequence >> 16) % actions[0];
            puf_step(&env);
            fwrite(&env.rng, sizeof(env.rng), 1, out);
            fwrite(&env.player, sizeof(env.player), 1, out);
            fwrite(&env.tick, sizeof(env.tick), 1, out);
            fwrite(env.grid, sizeof(env.grid), 1, out);
            fwrite(&env.log, sizeof(env.log), 1, out);
            fwrite(&reward, sizeof(reward), 1, out);
            fwrite(&terminal, sizeof(terminal), 1, out);
            fwrite(obs, sizeof(obs), 1, out);
        }
        puf_close(&env);
    }
    puf_ini_free(&ini);
    assert(!ferror(out));
    fclose(out);
}
