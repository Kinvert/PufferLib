// CPU native world/raster fixtures only; no policy/model execution.
#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include ENV_HEADER
#include "../exact_episode.h"

typedef struct { Env env; float buffer[OBS_SIZE + 2]; float action, reward, terminal; } Fixture;

static void setup(Fixture* f, unsigned seed, int skip, unsigned target, int representation) {
    memset(f, 0, sizeof(*f)); f->buffer[0] = 12345; f->buffer[OBS_SIZE + 1] = -12345;
    f->env.agents[0].observations = f->buffer + 1;
    f->env.agents[0].actions = &f->action; f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal; f->env.rng = seed;
    Dict kwargs = {0};
    dict_set(&kwargs, "width", 500); dict_set(&kwargs, "height", 640);
    dict_set(&kwargs, "paddle_width", 20); dict_set(&kwargs, "paddle_height", 70);
    dict_set(&kwargs, "ball_width", 32); dict_set(&kwargs, "ball_height", 32);
    dict_set(&kwargs, "paddle_speed", 8); dict_set(&kwargs, "ball_initial_speed_x", 10);
    dict_set(&kwargs, "ball_initial_speed_y", 1); dict_set(&kwargs, "ball_max_speed_y", 13);
    dict_set(&kwargs, "ball_speed_y_increment", 3); dict_set(&kwargs, "max_score", target);
    dict_set(&kwargs, "frameskip", skip); dict_set(&kwargs, "continuous", 0);
#ifdef TEST_PIXELS
    dict_set(&kwargs, "representation", representation); dict_set(&kwargs, "representation_mode", 0);
#endif
    puf_init(&f->env, &kwargs); dict_clear(&kwargs); puf_reset(&f->env);
}

static void observation(Fixture* f) {
    assert(f->buffer[0] == 12345 && f->buffer[OBS_SIZE + 1] == -12345);
    for (int i = 1; i <= OBS_SIZE; i++) assert(isfinite(f->buffer[i]));
    float saved[OBS_SIZE]; memcpy(saved, f->buffer + 1, sizeof(saved));
    unsigned rng = f->env.rng; compute_observations(&f->env);
    assert(memcmp(saved, f->buffer + 1, sizeof(saved)) == 0 && f->env.rng == rng);
}

static void starts(int representation) {
    unsigned seeds[] = {0, 56111, UINT32_MAX}, ids[] = {0, 64, UINT32_MAX};
    for (int s = 0; s < 3; s++) for (int i = 0; i < 3; i++) {
        Fixture a, b; setup(&a, 1, 8, 21, representation); setup(&b, 999, 8, 21, representation);
        PongExactEpisode ea, eb;
        b.env.score_r = 20; b.env.score_l = 19; b.env.tick = 777;
        b.env.paddle_dir = 1; b.env.win = 1; b.env.n_bounces = 99;
        b.env.ball_vx = -999; b.env.ball_y = -100; b.env.log.n = 20;
        pong_exact_start(&a.env, &ea, seeds[s], ids[i]);
        pong_exact_start(&b.env, &eb, seeds[s], ids[i]);
        unsigned expected = c4_exact_env_seed(seeds[s], ids[i]);
        unsigned direction = rand_r(&expected);
        assert(a.env.rng == expected && b.env.rng == expected);
        assert(a.env.ball_vy == ((direction & 1) ? 1.0f : -1.0f));
        assert(a.env.win == 0 && b.env.win == 0 && a.env.paddle_dir == 0 && b.env.paddle_dir == 0);
        assert(a.terminal == 1 && a.action == 0 && a.reward == 0 && a.env.log.n == 0);
        assert(memcmp(&ea, &eb, sizeof(ea)) == 0 && ea.decisions == 0 && ea.rally_decisions == 0);
        assert(pong_exact_world_hash(&a.env) == pong_exact_world_hash(&b.env));
        assert(memcmp(a.buffer + 1, b.buffer + 1, OBS_SIZE*sizeof(float)) == 0);
        observation(&a); observation(&b);
        printf("start %u %u %u %016" PRIx64 "\n", seeds[s], ids[i], expected, pong_exact_world_hash(&a.env));
        puf_close(&a.env); puf_close(&b.env);
    }
}

static void force_point(Fixture* f, int right) {
    f->env.ball_y = f->env.height - f->env.ball_height;
    f->env.ball_vy = 0;
    f->env.ball_x = right ? -1 : f->env.width + 1;
    f->env.ball_vx = right ? -10 : 10;
}

static void scoring(int representation) {
    int skips[] = {1, 3, 8};
    for (int k = 0; k < 3; k++) for (int win = 0; win < 2; win++) {
        Fixture a, b; setup(&a, 0, skips[k], 3, representation); setup(&b, 0, skips[k], 3, representation);
        PongExactEpisode e, reference;
        pong_exact_start(&a.env, &e, 56111, 1000 + win);
        pong_exact_start(&b.env, &reference, 56111, 1000 + win);
        // Opposite point first, then three winning-side points; nonpoint actions
        // between them make whole-match length visibly differ from last rally.
        int sides[] = {!win, win, win, win};
        for (int point = 0; point < 4; point++) {
            for (int delay = 0; delay < 2; delay++) {
                b.action = 0; puf_step(&b.env);
                assert(pong_exact_step(&a.env, &e, 0, 12));
                assert(a.reward == 0 && b.reward == 0 && !e.completed);
                assert(pong_exact_world_hash(&a.env) == pong_exact_world_hash(&b.env));
            }
            force_point(&a, sides[point]); force_point(&b, sides[point]);
            unsigned rng = a.env.rng; rand_r(&rng);
            b.action = 0; puf_step(&b.env);
            assert(pong_exact_step(&a.env, &e, 0, 12));
            assert(a.reward == (sides[point] ? 1 : -1) && a.reward == b.reward);
            assert(a.env.rng == rng && b.env.rng == rng);
            assert(pong_exact_world_hash(&a.env) == pong_exact_world_hash(&b.env));
            assert(memcmp(a.buffer + 1, b.buffer + 1, OBS_SIZE*sizeof(float)) == 0);
            assert(e.decisions == (point+1)*3);
            if (point < 3) {
                assert(!e.completed && !e.capped && e.rally_decisions == 0 && a.env.tick == 0 && a.env.log.n == 0);
            } else {
                assert(e.completed && !e.capped && e.decisions == 12 && e.rally_decisions == 3);
                assert(a.env.log.episode_length == 3 && a.env.log.episode_length != e.decisions);
                assert(e.right == (win ? 3u : 1u) && e.left == (win ? 1u : 3u));
                assert(e.episode_return == (win ? 2 : -2) && a.env.log.episode_return == e.episode_return);
                assert(a.env.score_r == 0 && a.env.score_l == 0 && a.terminal == 1);
                Log saved = a.env.log;
                a.env.log.episode_length = 12; assert(!pong_exact_valid(&a.env, &e, 12)); a.env.log = saved;
                a.env.log.perf = NAN; assert(!pong_exact_valid(&a.env, &e, 12)); a.env.log = saved;
                assert(pong_exact_valid(&a.env, &e, 12));
            }
            observation(&a);
        }
        uint64_t world = pong_exact_world_hash(&a.env);
        assert(!pong_exact_step(&a.env, &e, 0, 100));
        assert(world == pong_exact_world_hash(&a.env));
        double low, high; assert(pong_exact_bounds(e.right, e.left, 3, 1, &low, &high));
        assert(low == high && low == (win ? 0.75 : 0.25));
        printf("complete %d %d %d %d %u %u %a %a\n", skips[k], win, e.decisions, e.rally_decisions,
            e.right, e.left, e.episode_return, low);
        puf_close(&a.env); puf_close(&b.env);
    }
}

static void caps(int representation) {
    for (int points = 0; points < 2; points++) {
        Fixture f; setup(&f, 0, 8, 21, representation); PongExactEpisode e;
        pong_exact_start(&f.env, &e, 56111, 2000);
        unsigned rng = f.env.rng;
        float bad[] = {-1, 3, 0.5f, NAN, INFINITY};
        for (int i = 0; i < 5; i++) assert(!pong_exact_step(&f.env, &e, bad[i], 2));
        assert(!pong_exact_step(&f.env, &e, 0, 0) && !pong_exact_step(&f.env, &e, 0, 16777217));
        assert(e.decisions == 0 && f.env.rng == rng);
        assert(pong_exact_step(&f.env, &e, 0, 2));
        if (points) force_point(&f, 1);
        assert(pong_exact_step(&f.env, &e, 0, 2));
        assert(e.capped && !e.completed && e.decisions == 2 && e.right == (unsigned)points && e.left == 0);
        assert(f.terminal == 0 && f.env.log.n == 0 && f.env.score_r == e.right);
        assert(e.rally_decisions == (points ? 0 : 2) && f.env.tick == e.rally_decisions);
        double low, high; assert(pong_exact_bounds(e.right, e.left, 21, 0, &low, &high));
        assert(low == (points ? 1.0/22 : 0) && high == 1);
        uint64_t world = pong_exact_world_hash(&f.env);
        assert(!pong_exact_step(&f.env, &e, 1, 100) && world == pong_exact_world_hash(&f.env));
        e.capped = 0; assert(!pong_exact_valid(&f.env, &e, 2)); e.capped = 1;
        e.episode_return = NAN; assert(!pong_exact_valid(&f.env, &e, 2)); e.episode_return = points;
        assert(pong_exact_valid(&f.env, &e, 2)); observation(&f);
        printf("cap %d %d %d %u %a %a\n", points, e.decisions, e.rally_decisions, e.right, low, high);
        puf_close(&f.env);
    }
}

static void bounds(void) {
    for (unsigned target = 1; target <= 21; target++) {
        for (unsigned r = 0; r < target; r++) for (unsigned l = 0; l < target; l++) {
            double low, high; assert(pong_exact_bounds(r, l, target, 0, &low, &high));
            // Independently enumerate every possible final score pair. No NN.
            double actual_low = 1, actual_high = 0;
            for (unsigned loser = l; loser < target; loser++) {
                double fraction = (double)target/(target+loser);
                actual_low = fmin(actual_low, fraction); actual_high = fmax(actual_high, fraction);
            }
            for (unsigned loser = r; loser < target; loser++) {
                double fraction = (double)loser/(target+loser);
                actual_low = fmin(actual_low, fraction); actual_high = fmax(actual_high, fraction);
            }
            assert(low == actual_low && high == actual_high);
            assert(!pong_exact_bounds(r, l, target, 1, &low, &high));
        }
        for (unsigned loser = 0; loser < target; loser++) {
            double low, high;
            assert(pong_exact_bounds(target, loser, target, 1, &low, &high) && low == high);
            assert(pong_exact_bounds(loser, target, target, 1, &low, &high) && low == high);
            assert(!pong_exact_bounds(target, loser, target, 0, &low, &high));
        }
        double low, high; assert(!pong_exact_bounds(target, target, target, 1, &low, &high));
    }
    double low, high;
    assert(!pong_exact_bounds(0, 0, 0, 0, &low, &high));
    assert(!pong_exact_bounds(22, 0, 21, 0, &low, &high));
    assert(!pong_exact_bounds(0, 0, 21, 2, &low, &high));
    printf("bounds targets=1..21 enumerated\n");
}

static void trajectories(int representation) {
    int skips[] = {1, 3, 8};
    for (int k = 0; k < 3; k++) for (unsigned id = 0; id < 32; id++) {
        Fixture a, b; setup(&a, 0, skips[k], 21, representation); setup(&b, 999, skips[k], 21, representation);
        PongExactEpisode e, reference; pong_exact_start(&a.env, &e, 56111, id); pong_exact_start(&b.env, &reference, 56111, id);
        unsigned actions = 1000 + id, right = 0, left = 0;
        while (!e.completed && !e.capped) {
            float action = rand_r(&actions) % 3; b.action = action; puf_step(&b.env);
            assert(pong_exact_step(&a.env, &e, action, 2048));
            right += b.reward == 1; left += b.reward == -1;
            assert(e.right == right && e.left == left && a.reward == b.reward && a.terminal == b.terminal);
            assert(pong_exact_world_hash(&a.env) == pong_exact_world_hash(&b.env));
            assert(memcmp(a.buffer + 1, b.buffer + 1, OBS_SIZE*sizeof(float)) == 0);
            observation(&a);
        }
        assert(e.completed != e.capped);
        printf("trajectory %d %u %d %u %u %d %d %016" PRIx64 "\n", skips[k], id, e.decisions,
            e.right, e.left, e.completed, e.capped, pong_exact_world_hash(&a.env));
        puf_close(&a.env); puf_close(&b.env);
    }
}

int main(int argc, char** argv) {
    int representation = argc > 1 ? atoi(argv[1]) : 0;
    starts(representation); scoring(representation); caps(representation); bounds(); trajectories(representation);
    return 0;
}
