// Reuse independent game/pixel checks; no neural policy executes.
#define main maze_environment_tests_main
#include "test_environment.c"
#undef main
#include "../exact_episode.h"

static void exact_starts(void) {
    uint32_t seeds[] = {0, 56111, UINT32_MAX}, ids[] = {0, 64, UINT32_MAX};
    for (int s = 0; s < 3; s++) for (int i = 0; i < 3; i++) {
        Fixture a, b; setup(&a, 1, 32, -1); setup(&b, 999, 32, -1);
        MazeExactEpisode ea, eb;
        memset(&b.env.state, 99, sizeof(State)); b.env.tick = 999; b.env.log.n = 100;
        b.action = 99; b.reward = NAN; b.terminal = 0;
        assert(maze_exact_start(&a.env, &ea, seeds[s], ids[i], 7, 13));
        assert(maze_exact_start(&b.env, &eb, seeds[s], ids[i], 7, 13));
        assert(memcmp(&ea, &eb, sizeof(ea)) == 0 && ea.level >= 7 && ea.level < 20);
        assert(a.env.rng == c4_exact_env_seed(seeds[s], ids[i]) && b.env.rng == a.env.rng);
        assert(a.terminal == 1 && b.terminal == 1 && b.action == 0 && b.reward == 0);
        assert(memcmp(&a.env.state, &a.env.levels[ea.level], sizeof(State)) == 0);
        assert(maze_exact_world_hash(&a.env) == maze_exact_world_hash(&b.env));
        assert(maze_exact_observation_hash(a.env.agents[0].observations) == maze_exact_observation_hash(b.env.agents[0].observations));
        pixels(&a); pixels(&b);
        uint64_t start = maze_exact_world_hash(&a.env);
        assert(!maze_exact_start(&a.env, &ea, seeds[s], ids[i], 0, 0));
        assert(!maze_exact_start(&a.env, &ea, seeds[s], ids[i], 31, 2));
        assert(start == maze_exact_world_hash(&a.env));
        printf("start %u %u %u %016" PRIx64 "\n", seeds[s], ids[i], eb.level, start);
        puf_close(&a.env); puf_close(&b.env);
    }
    for (uint32_t count = 1; count <= 37; count++) {
        unsigned char seen[37] = {0};
        for (uint32_t id = 0; id < count; id++) seen[maze_exact_level(56111, id, 0, count)]++;
        for (uint32_t id = 0; id < count; id++) assert(seen[id] == 1);
    }
    // Original generator's shorter training table is the exact prefix of a larger eval table.
    State* train = make_maze_levels(17, -1); State* eval = make_maze_levels(32, -1);
    assert(memcmp(train, eval, 17 * sizeof(State)) == 0); free(train); free(eval);
}

static void exact_trajectories(void) {
    for (int regime = 0; regime < 3; regime++) for (uint32_t id = 0; id < 32; id++) {
        int size = regime == 0 ? 5 : regime == 1 ? 11 : -1;
        Fixture f; setup(&f, 999, 32, size);
        MazeExactEpisode e; assert(maze_exact_start(&f.env, &e, 56111, id, 3, 29));
        Reference r = {f.env.state, f.env.log, f.env.tick, f.env.rng};
        float bad[] = {-1, 5, 1.5f, NAN, INFINITY}; uint64_t start = maze_exact_world_hash(&f.env);
        for (int i = 0; i < 5; i++) assert(!maze_exact_step(&f.env, &e, bad[i]) && start == maze_exact_world_hash(&f.env));
        MazeExactEpisode saved = e; e.horizon++; assert(!maze_exact_step(&f.env, &e, 0)); e = saved;
        while (!e.end_reason) {
            int action = id & 1 ? next_action(&r.state) : 0;
            float reward, terminal; reference_step(&r, f.env.levels, f.env.num_levels, action, &reward, &terminal);
            assert(maze_exact_step(&f.env, &e, action));
            assert(f.reward == reward && f.terminal == terminal && f.env.rng == r.rng && f.env.tick == r.tick);
            assert(memcmp(&f.env.state, &r.state, sizeof(State)) == 0 && memcmp(&f.env.log, &r.log, sizeof(Log)) == 0);
            if (!(e.decisions % 32) || e.end_reason) pixels(&f);
        }
        assert(e.success == (int)(id & 1) && e.episode_return == e.success);
        assert(e.native_logs == 1 + (e.success && e.decisions == e.horizon));
        assert(e.end_reason == (e.success ? 1 : 2) && (e.success || e.decisions == e.horizon));
        uint64_t ending = maze_exact_world_hash(&f.env); assert(!maze_exact_step(&f.env, &e, 0));
        assert(ending == maze_exact_world_hash(&f.env));
        printf("trajectory %d %u %u %d %d %016" PRIx64 "\n", size, id, e.level, e.decisions, e.end_reason, ending);
        puf_close(&f.env);
    }
}

static void goal_at_timeout(void) {
    Fixture f; setup(&f, 999, 8, 5);
    MazeExactEpisode e; assert(maze_exact_start(&f.env, &e, 56111, 0, 0, 8));
    // Controlled environment fixture, not a policy/benchmark outcome.
    f.env.state.maze[49] = GOAL; f.env.tick = e.decisions = e.horizon - 1;
    unsigned int rng = f.env.rng; rand_r(&rng); rand_r(&rng);
    assert(maze_exact_step(&f.env, &e, ATN_EAST));
    assert(e.success == 1 && e.end_reason == 1 && e.native_logs == 2 && e.episode_return == 1);
    assert(f.env.log.n == 2 && f.env.log.perf == 2 && f.env.rng == rng);
    printf("goal-at-timeout %d %d %d\n", e.decisions, e.success, e.native_logs); puf_close(&f.env);
}

int main(int argc, char** argv) {
#ifdef TEST_PIXELS
    representation = argc > 1 ? strtod(argv[1], NULL) : 0;
#endif
    exact_starts(); exact_trajectories(); goal_at_timeout();
    puts("PASS: exact Maze environment/identity/counter checks"); return 0;
}
