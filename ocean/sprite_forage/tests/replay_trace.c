// Replay real GPU-policy actions through both observation modes/all five skins.
// No CPU neural inference. Full world/reward/log bytes must match every step.
#include <assert.h>
#include <inttypes.h>
#include "../sprite_forage.h"
#include "trace_types.h"

void match_snapshot(ForageSnapshot* actual, ForageSnapshot* expected, int step, int after) {
    const unsigned char* a = (const unsigned char*)actual;
    const unsigned char* b = (const unsigned char*)expected;
    for (int i = 0; i < (int)sizeof(*actual); i++) {
        if (a[i] != b[i]) {
            fprintf(stderr, "step=%d after=%d byte=%d actual=%02x expected=%02x "
                "rng=%u/%u player=%d/%d tick=%d/%d reward=%a/%a terminal=%a/%a\n",
                step, after, i, a[i], b[i], actual->rng, expected->rng,
                actual->player, expected->player, actual->tick, expected->tick,
                actual->reward, expected->reward, actual->terminal, expected->terminal);
            assert(a[i] == b[i]);
        }
    }
}

int main(int argc, char** argv) {
    assert(argc == 4);
    Ini ini = {0};
    puf_ini_load_file(&ini, argv[1]);
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    dict_set(kwargs, "seed", atoi(argv[3]));
    for (int mode = 0; mode < 2; mode++) {
        for (int rep = 0; rep < FORAGE_REPRESENTATIONS; rep++) {
            dict_set(kwargs, "observation_mode", mode);
            dict_set(kwargs, "representation", rep);
            Env env = {0};
            float action = 0, reward = 0, terminal = 0;
            int size = puf_observation_size(kwargs);
            float* obs = calloc(size, sizeof(float));
            assert(obs);
            env.agents[0].observations = obs;
            env.agents[0].actions = &action;
            env.agents[0].rewards = &reward;
            env.agents[0].terminals = &terminal;
            puf_init(&env, kwargs);
            puf_reset(&env);
            FILE* file = fopen(argv[2], "rb");
            assert(file);
            assert(fseek(file, 0, SEEK_END) == 0);
            long bytes = ftell(file);
            assert(bytes > 0 && bytes % sizeof(ForageTrace) == 0);
            rewind(file);
            ForageTrace row;
            int steps = 0, episodes = 0;
            uint64_t pixels = UINT64_C(14695981039346656037);
            while (fread(&row, sizeof(row), 1, file) == 1) {
                assert(row.index == steps++);
                // The native vector worker clears these outputs before each step.
                reward = terminal = 0;
                ForageSnapshot actual;
                forage_snapshot(&env, &actual);
                match_snapshot(&actual, &row.before, row.index, 0);
                action = row.action;
                puf_step(&env);
                forage_snapshot(&env, &actual);
                match_snapshot(&actual, &row.after, row.index, 1);
                if (!mode) {
                    assert(!memcmp(obs, row.observation, sizeof(row.observation)));
                }
                if (terminal) {
                    episodes++;
                    const unsigned char* bytes = (const unsigned char*)obs;
                    for (int i = 0; i < size * (int)sizeof(float); i++) {
                        pixels = (pixels ^ bytes[i]) * UINT64_C(1099511628211);
                    }
                }
            }
            assert(feof(file) && !ferror(file) && episodes == 32);
            assert(steps == bytes / sizeof(ForageTrace));
            assert(fclose(file) == 0);
            printf("mode=%d rep=%d steps=%d episodes=%d reset_observations=%016" PRIx64
                " exact_world_reward_log_bytes=PASS\n", mode, rep, steps, episodes, pixels);
            puf_close(&env);
            free(obs);
        }
    }
    puf_ini_free(&ini);
}
