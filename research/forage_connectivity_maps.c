// Scalar game-generation diagnostics only. No policy, CNN or GPU execution.
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include FORAGE_TEST_HEADER

typedef struct {
    Env env;
    float observations[141];
    float actions[1];
    float reward, terminal;
} Fixture;

void setup(Fixture* f) {
    f->env.rng = 7;
    f->env.agents[0] = (Agent){.observations = f->observations,
        .actions = f->actions, .rewards = &f->reward, .terminals = &f->terminal};
    Dict kwargs = {0};
    dict_set(&kwargs, "seed", 0);
    dict_set(&kwargs, "observation_mode", 0);
    dict_set(&kwargs, "representation", 0);
    dict_set(&kwargs, "state_layout", 2);
    dict_set(&kwargs, "copies_per_type", 3);
    dict_set(&kwargs, "trees", 8);
    dict_set(&kwargs, "max_steps", 676);
    dict_set(&kwargs, "reward_good", 1);
    dict_set(&kwargs, "reward_other", 0);
    dict_set(&kwargs, "reward_grape", 3.71950364);
    dict_set(&kwargs, "reward_bad", -1.68730032);
    dict_set(&kwargs, "reward_death", -.5);
    dict_set(&kwargs, "step_cost", -.001);
    puf_init(&f->env, &kwargs);
    dict_clear(&kwargs);
}

int main(void) {
    Fixture first = {0}, repeat = {0};
    setup(&first);
    setup(&repeat);
    int disconnected = 0, impossible = 0, missing_apples = 0, missing_grapes = 0;
    long long reset_ns = 0;
    for (int episode = 0; episode < 4096; episode++) {
        struct timespec begin, end;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        puf_reset(&first.env);
        clock_gettime(CLOCK_MONOTONIC, &end);
        reset_ns += (end.tv_sec - begin.tv_sec) * 1000000000LL + end.tv_nsec - begin.tv_nsec;
        puf_reset(&repeat.env);
        assert(first.env.rng == repeat.env.rng);
        assert(first.env.player_idx == repeat.env.player_idx);
        assert(!memcmp(first.env.grid, repeat.env.grid, FORAGE_AREA));
        assert(!memcmp(first.observations, repeat.observations, sizeof(first.observations)));
        disconnected += !forage_connected(&first.env);

        unsigned char visited[FORAGE_AREA] = {0};
        int queue[FORAGE_AREA], head = 0, tail = 1;
        queue[0] = first.env.player_idx;
        visited[queue[0]] = 1;
        int delta[] = {-FORAGE_WIDTH, FORAGE_WIDTH, -1, 1};
        while (head < tail) {
            int cell = queue[head++];
            if (first.env.grid[cell] == FORAGE_GRAPE) {
                continue;
            }
            for (int a = 0; a < 4; a++) {
                int next_idx = cell + delta[a];
                if (visited[next_idx] || first.env.grid[next_idx] >= FORAGE_TOADSTOOL) {
                    continue;
                }
                visited[next_idx] = 1;
                queue[tail++] = next_idx;
            }
        }
        int apples = 0, grapes = 0;
        for (int cell = 0; cell < FORAGE_AREA; cell++) {
            apples += first.env.grid[cell] == FORAGE_APPLE && !visited[cell];
            grapes += first.env.grid[cell] == FORAGE_GRAPE && !visited[cell];
        }
        missing_apples += apples;
        missing_grapes += grapes;
        impossible += apples > 0 || grapes > 0;
    }
    printf("{\"maps\":4096,\"seed\":7,\"repeat_exact\":true,"
        "\"disconnected\":%d,\"unreachable_objective_maps\":%d,"
        "\"unreachable_apples\":%d,\"unreachable_grapes\":%d,"
        "\"reset_mean_us\":%.6f}\n", disconnected, impossible,
        missing_apples, missing_grapes, reset_ns / 4096.0 / 1000.0);
    return 0;
}
