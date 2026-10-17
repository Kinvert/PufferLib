// Reuse the independent environment reference/fixtures; no neural policy.
#define main snake_environment_tests_main
#include "test_environment.c"
#undef main
#include "../exact_episode.h"

static void starts(void) {
    uint32_t seeds[] = {0, 56111, UINT32_MAX}, ids[] = {0, 64, UINT32_MAX};
    for (int s = 0; s < 3; s++) for (int i = 0; i < 3; i++) {
        Fixture a, b; setup(&a, 1, 26, 4, 128, 37); setup(&b, 999, 26, 4, 128, 37);
        SnakeExactEpisode ea, eb;
        memset(b.env.grid, 255, 26*26); b.env.tick = 999;
        b.env.length = 999; b.env.head_ptr = 999; b.env.food_collected = 99;
        b.env.episode_return = NAN; b.env.last_end_reason = 2; b.env.log.n = 100;
        b.action = 99; b.reward = NAN; b.terminal = 0;
        snake_exact_start(&a.env, &ea, seeds[s], ids[i]);
        snake_exact_start(&b.env, &eb, seeds[s], ids[i]);
        assert(snake_exact_valid(&a.env, &ea) && snake_exact_valid(&b.env, &eb));
        Reference r = reference(&a, c4_exact_env_seed(seeds[s], ids[i])); r.terminal = 1;
        compare(&a, &r); compare(&b, &r);
        assert(a.terminal == 1 && b.terminal == 1 && a.action == 0 && a.reward == 0);
        assert(memcmp(a.obs+1, b.obs+1, OBS_SIZE*sizeof(float)) == 0);
        assert(snake_exact_world_hash(&a.env) == snake_exact_world_hash(&b.env));
        assert(snake_exact_observation_hash(a.obs+1) == snake_exact_observation_hash(b.obs+1));
        assert(memcmp(&ea, &eb, sizeof(ea)) == 0);
        observation(&a); observation(&b);
        printf("exact-start %u %u %u %016" PRIx64 "\n", seeds[s], ids[i], a.env.rng, snake_exact_world_hash(&a.env));
        puf_close(&a.env); puf_close(&b.env);
    }
}
static void terminal_cases(void) {
    for (int horizon = 1; horizon <= 2; horizon++) {
        Fixture f; setup(&f, 0, 26, 1, 8, horizon);
        SnakeExactEpisode e; snake_exact_start(&f.env, &e, 56111, 0);
        int edge = 5*26+5; place(&f, &edge, 1, 0);
        Reference r; from_fixture(&r, &f);
        assert(snake_exact_step(&f.env, &e, 0)); ref_step(&r, 0); compare(&f, &r);
        assert(e.end_reason == 1 && e.decisions == 1 && e.foods == 0 && e.score == 1 && e.episode_return == -1);
        uint64_t hash = snake_exact_world_hash(&f.env);
        assert(!snake_exact_step(&f.env, &e, 3) && hash == snake_exact_world_hash(&f.env));
        f.env.log.score = 999; assert(!snake_exact_valid(&f.env, &e));
        printf("death-horizon %d %016" PRIx64 "\n", horizon, hash); puf_close(&f.env);
    }
    for (int zero_rewards = 0; zero_rewards <= 1; zero_rewards++) {
        Fixture f; setup(&f, 0, 26, 1, 8, 1);
        if (zero_rewards) f.env.reward_food = f.env.reward_death = 0;
        SnakeExactEpisode e; snake_exact_start(&f.env, &e, 56111, 1);
        int center = 13*26+13; place(&f, &center, 1, 0);
        for (int i = 0; i < 26*26; i++) if (f.env.grid[i] == 1) f.env.grid[i] = 0;
        f.env.grid[center+1] = 1; Reference r; from_fixture(&r, &f);
        assert(snake_exact_step(&f.env, &e, 3)); ref_step(&r, 3); compare(&f, &r);
        assert(e.end_reason == 2 && e.decisions == 1 && e.foods == 1 && e.score == 2);
        assert(e.episode_return == f.env.reward_food && f.reward == f.env.reward_food && f.terminal == 1);
        printf("rewarded-horizon %d %016" PRIx64 "\n", zero_rewards, snake_exact_world_hash(&f.env)); puf_close(&f.env);
    }
}
static void rounded_return(void) {
    Fixture f; setup(&f, 0, 26, 1, 8, 100);
    SnakeExactEpisode e; snake_exact_start(&f.env, &e, 56111, 2);
    int cell = 13*26+6; place(&f, &cell, 1, 0);
    Reference r; from_fixture(&r, &f);
    for (int count = 0; count < 7; count++) {
        for (int i = 0; i < 26*26; i++) if (f.env.grid[i] == 1) f.env.grid[i] = r.grid[i] = 0;
        int next = f.env.snake[f.env.head_ptr] + 1; f.env.grid[next] = r.grid[next] = 1;
        assert(snake_exact_step(&f.env, &e, 3)); ref_step(&r, 3); compare(&f, &r);
    }
    assert(e.foods == 7 && e.score == 7 && e.end_reason == 0); // Food can continue at the growth limit.
    for (int i = 0; i < 26*26; i++) if (f.env.grid[i] == 1) f.env.grid[i] = r.grid[i] = 0;
    f.env.grid[6*26+6] = r.grid[6*26+6] = 1;
    while (!e.end_reason) { assert(snake_exact_step(&f.env, &e, 0)); ref_step(&r, 0); compare(&f, &r); }
    assert(e.end_reason == 1 && e.foods == 7 && e.decisions == 16);
    assert(e.episode_return != 7*f.env.reward_food + f.env.reward_death);
    printf("rounded-return %a %016" PRIx64 "\n", e.episode_return, snake_exact_world_hash(&f.env));
    puf_close(&f.env);
}
static void exact_trajectories(void) {
    for (int board = 0; board < 3; board++) {
        int width = board == 0 ? 13 : board == 1 ? 26 : 38;
        int food = board == 0 ? 7 : 4, ring = board == 0 ? 3 : board == 1 ? 8 : 128;
        for (uint32_t id = 0; id < 32; id++) {
            Fixture f; setup(&f, 999, width, food, ring, 73);
            SnakeExactEpisode e; snake_exact_start(&f.env, &e, 56111, id);
            Reference r = reference(&f, c4_exact_env_seed(56111, id)); r.terminal = 1; compare(&f, &r);
            float bad[] = {-1, 4, 1.5f, NAN, INFINITY};
            uint64_t start = snake_exact_world_hash(&f.env);
            for (int k = 0; k < 5; k++) assert(!snake_exact_step(&f.env, &e, bad[k]) && start == snake_exact_world_hash(&f.env));
            while (!e.end_reason) {
                int action = ((id*7 + e.decisions*13) ^ ((e.decisions+3)/5)) % 4;
                assert(snake_exact_step(&f.env, &e, action)); ref_step(&r, action); compare(&f, &r); observation(&f);
            }
            assert(e.decisions <= 73 && e.score == f.env.last_score && e.foods == f.env.last_food);
            SnakeExactEpisode saved = e;
            e.episode_return = NAN; assert(!snake_exact_valid(&f.env, &e)); e = saved;
            e.foods = e.decisions+1; assert(!snake_exact_valid(&f.env, &e)); e = saved;
            e.score = 0; assert(!snake_exact_valid(&f.env, &e)); e = saved;
            printf("exact-trajectory %d %u %d %d %016" PRIx64 "\n", width, id, e.decisions, e.end_reason, snake_exact_world_hash(&f.env));
            puf_close(&f.env);
        }
    }
}
int main(int argc, char** argv) {
    representation = argc > 1 ? strtod(argv[1], NULL) : 0;
    starts(); terminal_cases(); rounded_return(); exact_trajectories();
    return 0;
}
