// Single-thread environment/raster throughput only; no neural computations.
#include <assert.h>
#include <time.h>
#include GAME_HEADER

double seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec * 1e-9;
}

int main(void) {
    Ini ini = {0};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, GAME_CONFIG);
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    const int actions[] = ACT_SIZES;
    puts("mode,repeat,steps,seconds,environment_sps,episodes");
    for (int mode = 0; mode < 2; mode++) {
        for (int repeat = 0; repeat < 3; repeat++) {
            dict_set(kwargs, "observation_mode", mode);
            dict_set(kwargs, "representation", repeat);
            Env env = {.rng = 7};
            float action = 0, reward = 0, terminal = 0;
            float* obs = calloc(puf_observation_size(kwargs), sizeof(float));
            env.agents[0] = (Agent){.observations = obs, .actions = &action,
                .rewards = &reward, .terminals = &terminal};
            puf_init(&env, kwargs);
            puf_reset(&env);
            unsigned int sequence = 42;
            int steps = mode ? 20000 : 2000000;
            double started = seconds();
            for (int step = 0; step < steps; step++) {
                sequence = sequence * 1664525u + 1013904223u;
                action = (sequence >> 16) % actions[0];
                reward = terminal = 0;
                puf_step(&env);
                // Keep all observation writes observable without adding neural work.
                __asm__ volatile("" : : "r"(obs) : "memory");
            }
            double elapsed = seconds() - started;
            assert(elapsed > 0 && env.log.n > 0);
            printf("%d,%d,%d,%.9f,%.3f,%.0f\n", mode, repeat, steps,
                elapsed, steps / elapsed, env.log.n);
            puf_close(&env);
            free(obs);
        }
    }
    puf_ini_free(&ini);
}
