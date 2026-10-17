// Constant actions and deterministic random control; no trained policy.
#include <assert.h>
#include GAME_HEADER

int main(void) {
    Ini ini = {0};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, GAME_CONFIG);
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    dict_set(kwargs, "observation_mode", 0);
    dict_set(kwargs, "seed", 2027);
    const int actions[] = ACT_SIZES;
    puts("policy,episodes,perf,perfect_episodes,deaths");
    for (int policy = 0; policy <= actions[0]; policy++) {
        Env env = {0}; // Match the initial RNG of native evaluation slot0.
        float action = 0, reward = 0, terminal = 0;
        float* obs = calloc(STATE_OBS_SIZE, sizeof(float));
        env.agents[0] = (Agent){.observations = obs, .actions = &action,
            .rewards = &reward, .terminals = &terminal};
        puf_init(&env, kwargs);
        puf_reset(&env);
        unsigned int sequence = 69;
        int steps = 0;
        while (env.log.n < 128) {
            sequence = sequence * 1664525u + 1013904223u;
            action = policy == actions[0] ? (sequence >> 16) % actions[0] : policy;
            reward = terminal = 0;
            puf_step(&env);
            assert(++steps < 100000);
        }
        printf("%d,128,%.9g,%.9g,%.9g\n", policy, env.log.perf /128,
            env.log.successes /128, env.log.deaths /128);
        puf_close(&env);
        free(obs);
    }
    puf_ini_free(&ini);
}
