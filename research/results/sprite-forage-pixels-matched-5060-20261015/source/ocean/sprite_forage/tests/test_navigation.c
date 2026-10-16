// Game feasibility only: privileged-state path planner, no neural computation.
#include <assert.h>
#include <math.h>
#include "../sprite_forage.h"

int main(void) {
    Ini ini = {0};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, "config/sprite_forage.ini");
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    dict_set(kwargs, "observation_mode", 0);
    const unsigned int seeds[] = {7, 42, 69, 420, 1337};
    const int delta[] = {0, -FORAGE_WIDTH, FORAGE_WIDTH, -1, 1};
    for (int seed = 0; seed < 5; seed++) {
        Env env = {0};
        float observation[puf_observation_size(kwargs)], action, reward, terminal;
        env.rng = seeds[seed];
        env.agents[0].observations = observation;
        env.agents[0].actions = &action;
        env.agents[0].rewards = &reward;
        env.agents[0].terminals = &terminal;
        puf_init(&env, kwargs);
        puf_reset(&env);
        int apples = 0, bad = 0;
        for (int step = 0; step < 512; step++) {
            int target = -1;
            int parent[FORAGE_AREA];
            // Prefer paths without bad items; allow nonfatal items if required.
            for (int allow_bad = 0; allow_bad <= 1 && target < 0; allow_bad++) {
                for (int cell = 0; cell < FORAGE_AREA; cell++) {
                    parent[cell] = -1;
                }
                int queue[FORAGE_AREA];
                int head = 0, tail = 1;
                queue[0] = env.player_idx;
                parent[env.player_idx] = env.player_idx;
                while (head < tail && target < 0) {
                    int cell = queue[head++];
                    for (int direction = 1; direction <= 4; direction++) {
                        int next = cell + delta[direction];
                        int limit = allow_bad ? FORAGE_TOADSTOOL : FORAGE_SPOTTED_APPLE;
                        if (next < 0 || next >= FORAGE_AREA || parent[next] >= 0
                            || env.grid[next] >= limit
                            || (env.apples < env.copies && env.grid[next] == FORAGE_GRAPE)) {
                            continue;
                        }
                        parent[next] = cell;
                        queue[tail++] = next;
                        int goal = env.apples < env.copies ? FORAGE_APPLE : FORAGE_GRAPE;
                        if (env.grid[next] == goal) {
                            target = next;
                            break;
                        }
                    }
                }
            }
            assert(target >= 0);
            while (parent[target] != env.player_idx) {
                target = parent[target];
            }
            action = 0;
            for (int direction = 1; direction <= 4; direction++) {
                if (env.player_idx + delta[direction] == target) {
                    action = direction;
                }
            }
            assert(action != 0);
            int kind = env.grid[env.player_idx + delta[(int)action]];
            apples += kind == FORAGE_APPLE;
            bad += kind >= FORAGE_SPOTTED_APPLE && kind <= FORAGE_TOADSTOOL;
            puf_step(&env);
            if (terminal) {
                break;
            }
        }
        assert(env.log.n == 1 && env.log.successes == 1 && env.log.deaths == 0);
        assert(apples == 3 && env.log.episode_length <= 512);
        float expected = fmaxf(0, 1 - bad / 4.f);
        assert(fabsf(env.log.perf - expected) < 1e-6f);
        printf("Privileged planner seed=%u steps=%.0f apples=%d bad=%d perf=%.6f PASS\n",
            seeds[seed], env.log.episode_length, apples, bad, env.log.perf);
        puf_close(&env);
    }
    puf_ini_free(&ini);
}
