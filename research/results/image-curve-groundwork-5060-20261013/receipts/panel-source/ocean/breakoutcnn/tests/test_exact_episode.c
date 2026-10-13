// CPU world/raster acceptance only. No policy, CUDA, or CPU neural execution.
#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include ENV_HEADER
#include "../exact_episode.h"

typedef struct {
    Env env;
    float observations[OBS_SIZE + 2];
    float action, reward, terminal;
} Fixture;

static void setup(Fixture* f, unsigned seed, int skip, int representation) {
    memset(f, 0, sizeof(*f));
    f->observations[0] = 12345; f->observations[OBS_SIZE + 1] = -12345;
    f->env.agents[0].observations = f->observations + 1;
    f->env.agents[0].actions = &f->action;
    f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal;
    f->env.rng = seed;
    Dict kwargs = {0};
    dict_set(&kwargs, "frameskip", skip);
    dict_set(&kwargs, "width", 576); dict_set(&kwargs, "height", 330);
    dict_set(&kwargs, "paddle_width", 62); dict_set(&kwargs, "paddle_height", 8);
    dict_set(&kwargs, "ball_width", 32); dict_set(&kwargs, "ball_height", 32);
    dict_set(&kwargs, "brick_width", 32); dict_set(&kwargs, "brick_height", 12);
    dict_set(&kwargs, "brick_rows", 6); dict_set(&kwargs, "brick_cols", 18);
    dict_set(&kwargs, "initial_ball_speed", 256); dict_set(&kwargs, "max_ball_speed", 448);
    dict_set(&kwargs, "paddle_speed", 620); dict_set(&kwargs, "continuous", 0);
#ifdef TEST_PIXELS
    dict_set(&kwargs, "representation", representation);
    dict_set(&kwargs, "representation_mode", 0);
#endif
    puf_init(&f->env, &kwargs); dict_clear(&kwargs);
    puf_reset(&f->env);
}

static void check_pixels(Fixture* f) {
    assert(f->observations[0] == 12345 && f->observations[OBS_SIZE + 1] == -12345);
    for (int i = 0; i < OBS_SIZE; i++) assert(isfinite(f->observations[i + 1]));
    float saved[OBS_SIZE]; memcpy(saved, f->observations + 1, sizeof(saved));
    unsigned rng = f->env.rng;
    compute_observations(&f->env);
    assert(memcmp(saved, f->observations + 1, sizeof(saved)) == 0 && rng == f->env.rng);
}

static void starts(int representation) {
    uint32_t seeds[] = {0, 56111, UINT32_MAX};
    uint32_t ids[] = {0, 64, UINT32_MAX};
    for (int s = 0; s < 3; s++) for (int i = 0; i < 3; i++) {
        Fixture a, b; setup(&a, 1, 3, representation); setup(&b, 999, 3, representation);
        BreakoutExactEpisode ea, eb;
        // Prior trajectory/log/counter contents must not affect the next start.
        b.env.score = 73; b.env.num_balls = -1; b.env.balls_fired = 7;
        b.env.ball_x = -100; b.env.ball_y = 999; b.env.hits = 999;
        b.env.paddle_width = 31; b.env.tick = 999; b.env.log.n = 5;
        for (int j = 0; j < b.env.num_bricks; j++) b.env.brick_states[j] = 1;
        breakout_exact_start(&a.env, &ea, seeds[s], ids[i]);
        breakout_exact_start(&b.env, &eb, seeds[s], ids[i]);
        unsigned expected_rng = c4_exact_env_seed(seeds[s], ids[i]);
        assert(a.env.rng == expected_rng && b.env.rng == expected_rng);
        assert(memcmp(&ea, &eb, sizeof(ea)) == 0 && ea.frames == 0 && ea.decisions == 0);
        assert(a.terminal == 1 && a.reward == 0 && a.action == 0 && a.env.log.n == 0);
        assert(breakout_exact_world_hash(&a.env) == breakout_exact_world_hash(&b.env));
        assert(breakout_exact_observation_hash(a.observations + 1)
            == breakout_exact_observation_hash(b.observations + 1));
        uint64_t start = breakout_exact_world_hash(&a.env);
        unsigned launch = rand_r(&expected_rng);
        assert(breakout_exact_step(&a.env, &ea, RIGHT, 1));
        assert(a.env.rng == expected_rng && ea.frames == 1 && ea.decisions == 1 && ea.capped);
        assert(a.env.ball_vx != 0 && (a.env.ball_vx < 0) == (launch % 2 == 0));
        check_pixels(&a); check_pixels(&b);
        printf("start %u %u %016" PRIx64 " %u %016" PRIx64 "\n",
            seeds[s], ids[i], start, a.env.rng, breakout_exact_world_hash(&a.env));
        puf_close(&a.env); puf_close(&b.env);
    }
}

static void last_loss(Fixture* f) {
    f->env.num_balls = 0; f->env.balls_fired = 1;
    f->env.ball_x = 0; f->env.ball_y = f->env.height + 100;
    f->env.ball_vx = 0; f->env.ball_vy = 1;
}

static void endings(int representation) {
    int skips[] = {1, 3, 8};
    for (int k = 0; k < 3; k++) {
        Fixture f, stock; setup(&f, 73, skips[k], representation); setup(&stock, 73, skips[k], representation);
        BreakoutExactEpisode e, reference;
        breakout_exact_start(&f.env, &e, 56111, 999);
        breakout_exact_start(&stock.env, &reference, 56111, 999);
        last_loss(&f); last_loss(&stock);
        unsigned rng = f.env.rng;
        assert(breakout_exact_step(&f.env, &e, NOOP, 1));
        assert(e.completed && !e.capped && e.frames == 1 && e.decisions == 1);
        assert(e.score == 0 && e.episode_return == 0 && f.env.log.episode_length == 1);
        assert(f.env.tick == 0 && f.env.num_balls == 5 && f.env.balls_fired == 0 && f.env.rng == rng);
        // Stock loops into the replacement game; exact evaluation must not.
        puf_step(&stock.env);
        assert(stock.terminal == 1 && stock.env.log.n == 1 && stock.env.log.episode_length == 1);
        assert(stock.env.tick == skips[k] - 1);
        assert(stock.env.balls_fired == (skips[k] > 1));
        assert((stock.env.rng == rng) == (skips[k] == 1));
        assert(!breakout_exact_step(&f.env, &e, RIGHT, 100));
        assert(f.env.rng == rng && e.frames == 1);
        check_pixels(&f);
        printf("ending %d %d %d %u\n", skips[k], e.frames, stock.env.tick, rng);
        puf_close(&f.env); puf_close(&stock.env);

        setup(&f, 73, skips[k], representation);
        breakout_exact_start(&f.env, &e, 56111, 1000);
        // Destroy one bottom-row brick (one point) on the final-score frame.
        int brick = f.env.num_bricks - 1;
        f.env.score = f.env.max_score - 1;
        e.episode_return = f.env.score;
        f.env.balls_fired = 1;
        f.env.ball_x = f.env.brick_x[brick];
        f.env.ball_y = f.env.brick_y[brick] + f.env.brick_height + 0.5f;
        f.env.ball_vx = 0; f.env.ball_vy = -1;
        assert(breakout_exact_step(&f.env, &e, NOOP, 1));
        assert(e.completed && !e.capped && e.score == f.env.max_score);
        assert(f.reward == 1 && e.episode_return == f.env.max_score && f.env.log.perf == 1);
        assert(f.env.log.score == e.score && f.env.score == 0 && f.env.tick == 0);
        Log saved = f.env.log;
        f.env.log.n = 2; assert(!breakout_exact_valid(&f.env, &e, 1)); f.env.log = saved;
        f.env.log.episode_return = NAN; assert(!breakout_exact_valid(&f.env, &e, 1)); f.env.log = saved;
        f.env.log.episode_length = 2; assert(!breakout_exact_valid(&f.env, &e, 1)); f.env.log = saved;
        assert(breakout_exact_valid(&f.env, &e, 1)); check_pixels(&f);
        printf("win %d %d %a %a\n", skips[k], e.score, f.reward, e.episode_return);
        puf_close(&f.env);
    }
}

static void caps(int representation) {
    Fixture f; setup(&f, 73, 3, representation);
    BreakoutExactEpisode e; breakout_exact_start(&f.env, &e, 56111, 1001);
    unsigned rng = f.env.rng;
    float invalid[] = {-1, 3, 0.5f, NAN, INFINITY};
    for (int i = 0; i < 5; i++) assert(!breakout_exact_step(&f.env, &e, invalid[i], 5));
    assert(!breakout_exact_step(&f.env, &e, NOOP, 0));
    assert(!breakout_exact_step(&f.env, &e, NOOP, 16777217));
    assert(e.frames == 0 && e.decisions == 0 && f.env.rng == rng);
    assert(breakout_exact_step(&f.env, &e, RIGHT, 5));
    assert(e.frames == 3 && e.decisions == 1 && !e.completed && !e.capped);
    assert(breakout_exact_step(&f.env, &e, LEFT, 5));
    assert(e.frames == 5 && e.decisions == 2 && e.capped && !e.completed);
    assert(f.env.tick == 5 && f.terminal == 0 && f.env.log.n == 0);
    uint64_t world = breakout_exact_world_hash(&f.env);
    assert(!breakout_exact_step(&f.env, &e, NOOP, 100));
    assert(breakout_exact_world_hash(&f.env) == world && e.frames == 5);
    e.episode_return = NAN; assert(!breakout_exact_valid(&f.env, &e, 5)); e.episode_return = e.score;
    e.capped = 0; assert(!breakout_exact_valid(&f.env, &e, 5)); e.capped = 1;
    e.decisions = 3; assert(!breakout_exact_valid(&f.env, &e, 5)); e.decisions = 2;
    assert(breakout_exact_valid(&f.env, &e, 5)); check_pixels(&f);
    printf("cap %d %d %d %016" PRIx64 "\n", e.frames, e.decisions, e.score, world);
    puf_close(&f.env);
}

static void trajectories(int representation) {
    int skips[] = {1, 3, 8};
    for (int k = 0; k < 3; k++) for (uint32_t id = 0; id < 32; id++) {
        Fixture a, b; setup(&a, 0, skips[k], representation); setup(&b, 999, skips[k], representation);
        BreakoutExactEpisode e, reference;
        breakout_exact_start(&a.env, &e, 56111, id);
        breakout_exact_start(&b.env, &reference, 56111, id);
        unsigned actions = 1000 + id;
        // Cap is a skip multiple: compare every nonterminal action to stock puf_step.
        int frame_cap = skips[k] * 512;
        while (!e.completed && !e.capped) {
            float action = rand_r(&actions) % 3;
            b.action = action; puf_step(&b.env);
            assert(breakout_exact_step(&a.env, &e, action, frame_cap));
            assert(a.reward == b.reward && a.terminal == b.terminal);
            if (!e.completed) {
                assert(breakout_exact_world_hash(&a.env) == breakout_exact_world_hash(&b.env));
                assert(memcmp(a.observations + 1, b.observations + 1, OBS_SIZE * sizeof(float)) == 0);
            } else {
                assert(a.env.log.n == b.env.log.n && a.env.log.n == 1);
                assert(a.env.log.score == b.env.log.score && a.env.log.episode_length == b.env.log.episode_length);
            }
            check_pixels(&a);
        }
        assert(e.completed != e.capped);
        printf("trajectory %d %u %d %d %d %d %d %016" PRIx64 "\n",
            skips[k], id, e.decisions, e.frames, e.score, e.completed, e.capped, breakout_exact_world_hash(&a.env));
        puf_close(&a.env); puf_close(&b.env);
    }
}

int main(int argc, char** argv) {
    int representation = argc > 1 ? atoi(argv[1]) : 0;
    starts(representation); endings(representation); caps(representation); trajectories(representation);
    return 0;
}
