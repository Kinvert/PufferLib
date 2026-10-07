// Environment/pixel fixtures, not CPU model training or CNN validation.
#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include ENV_HEADER
#include "../exact_episode.h"

static double representation, representation_mode, representation_seed, representation_mix_catalog;

typedef struct {
    Env env;
    float buffer[OBS_SIZE + 2];
    float action, reward, terminal;
} Fixture;

static void setup(Fixture* f, unsigned int seed, int max_steps) {
    memset(f, 0, sizeof(*f));
    f->buffer[0] = 12345; f->buffer[OBS_SIZE + 1] = -12345;
    f->env.agents[0].observations = f->buffer + 1;
    f->env.agents[0].actions = &f->action;
    f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal;
    Dict kwargs = {0};
    dict_set(&kwargs, "width", 420); dict_set(&kwargs, "height", 640);
    dict_set(&kwargs, "max_steps", max_steps);
    dict_set(&kwargs, "gravity", 0.45); dict_set(&kwargs, "flap_velocity", -7.5);
    dict_set(&kwargs, "pipe_speed", 3); dict_set(&kwargs, "pipe_gap", 190);
    dict_set(&kwargs, "pipe_width", 58); dict_set(&kwargs, "pipe_spacing", 220);
    dict_set(&kwargs, "first_pipe_x", 220); dict_set(&kwargs, "bird_x", 96);
    dict_set(&kwargs, "bird_radius", 14);
    dict_set(&kwargs, "alive_reward", 0.01); dict_set(&kwargs, "pass_reward", 1);
    dict_set(&kwargs, "crash_reward", -1); dict_set(&kwargs, "center_reward", 0.03);
#ifdef TEST_PIXELS
    dict_set(&kwargs, "representation", representation);
    dict_set(&kwargs, "representation_mode", representation_mode);
    dict_set(&kwargs, "representation_seed", representation_seed);
    dict_set(&kwargs, "representation_mix_catalog", representation_mix_catalog);
#endif
    f->env.rng = seed;
    puf_init(&f->env, &kwargs);
    assert(f->env.rng == seed);
    dict_clear(&kwargs);
    puf_reset(&f->env);
}

static void check_observation(Fixture* f) {
    assert(f->buffer[0] == 12345 && f->buffer[OBS_SIZE + 1] == -12345);
    obs_t* obs = f->env.agents[0].observations;
    for (int i = 0; i < OBS_SIZE; i++) {
        assert(isfinite(obs[i]));
#ifdef TEST_PIXELS
        assert(obs[i] == 0 || obs[i] == 0.5f || obs[i] == 1);
#endif
    }
    float saved[OBS_SIZE]; memcpy(saved, obs, sizeof(saved));
    unsigned int rng = f->env.rng;
    for (int i = 0; i < OBS_SIZE; i++) obs[i] = 99;
    flappy_compute_observations(&f->env);
    assert(memcmp(saved, obs, sizeof(saved)) == 0 && rng == f->env.rng);
}

static uint64_t state_hash(Fixture* f) {
    Env state = f->env;
    state.client = NULL; memset(state.agents, 0, sizeof(state.agents));
    const unsigned char* bytes = (const unsigned char*)&state;
    uint64_t hash = UINT64_C(14695981039346656037);
    // Original fields only, excluding the pixel appearance and tail padding.
    for (size_t i = 0; i < offsetof(Env, pipes) + sizeof(state.pipes); i++)
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    return hash;
}

static void event_fixtures(void) {
    Fixture f;
    setup(&f, 73, 4096);
    f.env.bird_y = f.env.height; f.action = 0;
    puf_step(&f.env);
    assert(f.terminal == 1 && f.reward < 0 && f.env.tick == 0 && f.env.log.n == 1);
    check_observation(&f);
    printf("boundary %016" PRIx64 " %a %a\n", state_hash(&f), f.reward, f.terminal);
    puf_close(&f.env);

    setup(&f, 73, 1);
    puf_step(&f.env);
    assert(f.terminal == 1 && f.env.tick == 0 && f.env.log.n == 1);
    check_observation(&f);
    printf("cap %016" PRIx64 " %a %a\n", state_hash(&f), f.reward, f.terminal);
    puf_close(&f.env);

    setup(&f, 73, 4096);
    f.env.pipes[0].x = f.env.bird_x;
    f.env.pipes[0].gap_y = 200;
    f.env.bird_y = 80;
    puf_step(&f.env);
    assert(f.terminal == 1 && f.reward < 0 && f.env.log.n == 1);
    check_observation(&f);
    printf("pipe-crash %016" PRIx64 " %a %a\n", state_hash(&f), f.reward, f.terminal);
    puf_close(&f.env);

    setup(&f, 73, 4096);
    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) f.env.pipes[i].gap_y = 320;
    f.env.pipes[0].x = f.env.bird_x - f.env.pipe_width + f.env.pipe_speed - 0.1f;
    puf_step(&f.env);
    assert(f.terminal == 0 && f.env.score == 1 && f.reward > 1);
    check_observation(&f);
    printf("pass %016" PRIx64 " %a %a\n", state_hash(&f), f.reward, f.terminal);
    puf_close(&f.env);

    setup(&f, 73, 4096);
    unsigned int rng = f.env.rng;
    f.env.pipes[0].x = -f.env.pipe_width - 1;
    puf_step(&f.env);
    assert(f.terminal == 0 && f.env.pipes[0].x > f.env.width && f.env.rng != rng);
    check_observation(&f);
    printf("respawn %016" PRIx64 " %a %a\n", state_hash(&f), f.reward, f.terminal);
    puf_close(&f.env);
}

#ifdef TEST_PIXELS
static float reference_pixel(Fixture* f, int x, int y) {
    // Independent pixel-space definition using long-double geometry. No
    // renderer helper or output buffer supplies the expected image.
    Env* e = &f->env;
    long double sx = (long double)OBS_WIDTH / e->width, sy = (long double)OBS_HEIGHT / e->height;
    float value = 0;
    for (int p = 0; p < FLAPPY_NUM_PIPES; p++) for (int half = 0; half < 2; half++) {
        int l = (int)fmaxl(0, fminl(OBS_WIDTH, floorl(e->pipes[p].x * sx)));
        int r = (int)fmaxl(0, fminl(OBS_WIDTH, ceill((e->pipes[p].x + e->pipe_width) * sx)));
        int t = half ? (int)fmaxl(0, fminl(OBS_HEIGHT, floorl((e->pipes[p].gap_y + e->pipe_gap / 2) * sy))) : 0;
        int b = half ? OBS_HEIGHT : (int)fmaxl(0, fminl(OBS_HEIGHT, ceill((e->pipes[p].gap_y - e->pipe_gap / 2) * sy)));
        if (x >= l && x < r && y >= t && y < b &&
                ((e->representation != 5 && e->representation != 6) || x == l || x == r-1 || y == t || y == b-1)) value = 0.5f;
    }
    long double cx = e->bird_x * sx, cy = e->bird_y * sy;
    long double rx = e->bird_radius * sx, ry = e->bird_radius * sy;
    if (e->representation == 4 || e->representation == 6) {
        long double dx = (x + 0.5L - cx) / rx, dy = (y + 0.5L - cy) / ry;
        if (dx*dx + dy*dy <= 1 || (cx >= 0 && cx < OBS_WIDTH && cy >= 0 && cy < OBS_HEIGHT && x == (int)cx && y == (int)cy)) value = 1;
    } else if (x >= floorl(cx-rx) && x < ceill(cx+rx) && y >= floorl(cy-ry) && y < ceill(cy+ry)) value = 1;
    return value;
}

static void expect_pixel(Fixture* f, int x, int y, float value) {
    int rep = f->env.representation;
    if (rep >= 4) value = reference_pixel(f, x, y);
    if (rep == 1) x = OBS_WIDTH - 1 - x;
    if (rep == 2) y = OBS_HEIGHT - 1 - y;
    if (rep == 3) value = 1 - value;
    assert(f->env.agents[0].observations[y * OBS_WIDTH + x] == value);
}

static void pixel_fixtures(void) {
    Fixture f; setup(&f, 73, 4096);
    assert(OBS_CHANNELS == 1 && OBS_HEIGHT == 36 && OBS_WIDTH == 44);
    f.env.width = 44; f.env.height = 36;
    f.env.pipe_width = 4; f.env.pipe_gap = 12;
    f.env.bird_x = 4; f.env.bird_y = 18; f.env.bird_radius = 1;
    f.env.pipes[0].x = 20; f.env.pipes[1].x = 40; f.env.pipes[2].x = 50;
    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) f.env.pipes[i].gap_y = 18;
    flappy_compute_observations(&f.env); check_observation(&f);
    for (int y = 0; y < 36; y++) for (int x = 0; x < 44; x++) {
        float expected = ((x >= 20 && x < 24) || x >= 40) && (y < 12 || y >= 24) ? 0.5f : 0;
        if (x >= 3 && x < 5 && y >= 17 && y < 19) expected = 1;
        expect_pixel(&f, x, y, expected);
    }
    // Dirty-buffer repeat and hidden velocity/score/tick independence.
    float saved[OBS_SIZE]; memcpy(saved, f.buffer + 1, sizeof(saved));
    f.env.bird_vy = -100; f.env.score = 999; f.env.tick = 1000;
    flappy_compute_observations(&f.env);
    assert(memcmp(saved, f.buffer + 1, sizeof(saved)) == 0);

    // Subpixel rectangle coverage and clipping; bird drawn over the pipe.
    for (int i = 1; i < FLAPPY_NUM_PIPES; i++) f.env.pipes[i].x = 50;
    f.env.pipes[0].x = -1.5f; f.env.pipe_width = 3;
    f.env.bird_x = 0.75f; f.env.bird_y = 3.25f; f.env.bird_radius = 0.25f;
    flappy_compute_observations(&f.env); check_observation(&f);
    for (int y = 0; y < 36; y++) for (int x = 0; x < 44; x++) {
        float expected = x < 2 && (y < 12 || y >= 24) ? 0.5f : 0;
        if (x == 0 && y == 3) expected = 1;
        expect_pixel(&f, x, y, expected);
    }
    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) f.env.pipes[i].x = 50;
    for (int corner = 0; corner < 4; corner++) {
        f.env.bird_x = corner & 1 ? 43 : 1;
        f.env.bird_y = corner & 2 ? 35 : 1;
        f.env.bird_radius = 2;
        flappy_compute_observations(&f.env); check_observation(&f);
        for (int y = 0; y < 36; y++) for (int x = 0; x < 44; x++) {
            int visible_x = corner & 1 ? x >= 41 : x < 3;
            int visible_y = corner & 2 ? y >= 33 : y < 3;
            expect_pixel(&f, x, y, visible_x && visible_y ? 1 : 0);
        }
    }
    puf_close(&f.env);
}

static void geometric_fixtures(void) {
    Fixture f; setup(&f, 73, 4096);
    const float positions[][3] = {{96,320,14},{96,90.25f,14},{96,639,14},
        {0.75f,3.25f,0.25f},{419,1,28},{-3,320,28},{423,320,28}};
    for (int rep = 4; rep <= 6; rep++) for (int k = 0; k < 7; k++) {
        f.env.representation = rep;
        f.env.bird_x = positions[k][0]; f.env.bird_y = positions[k][1]; f.env.bird_radius = positions[k][2];
        f.env.pipes[0].x = -17.5f; f.env.pipes[1].x = 75.5f; f.env.pipes[2].x = 420;
        for (int p = 0; p < FLAPPY_NUM_PIPES; p++) f.env.pipes[p].gap_y = 320;
        unsigned int rng = f.env.rng;
        flappy_compute_observations(&f.env); check_observation(&f);
        assert(f.env.rng == rng);
        for (int y = 0; y < OBS_HEIGHT; y++) for (int x = 0; x < OBS_WIDTH; x++)
            assert(f.env.agents[0].observations[y * OBS_WIDTH + x] == reference_pixel(&f,x,y));
    }
    puf_close(&f.env);
}
#endif

static void exact_episode_fixtures(void) {
    Fixture a, b;
    setup(&a, 7, 2); setup(&b, 99, 2);
    unsigned int seeds[] = {0, 56111, UINT32_MAX};
    unsigned int ids[] = {0, 64, UINT32_MAX};
    for (int s = 0; s < 3; s++) for (int e = 0; e < 3; e++) {
        unsigned int expected_rng = c4_exact_env_seed(seeds[s],ids[e]);
        float gaps[FLAPPY_NUM_PIPES];
        float margin = a.env.pipe_gap * 0.6f;
        for (int p = 0; p < FLAPPY_NUM_PIPES; p++) {
            float u = (float)rand_r(&expected_rng) / (float)RAND_MAX;
            gaps[p] = margin + u * (a.env.height - 2.0f * margin);
        }
        flappy_exact_start(&a.env,seeds[s],ids[e]);
        flappy_exact_start(&b.env,seeds[s],ids[e]);
        assert(a.env.rng == expected_rng && b.env.rng == expected_rng);
        assert(a.terminal == 1 && a.reward == 0 && a.action == 0);
        assert(flappy_exact_world_hash(&a.env) == flappy_exact_world_hash(&b.env));
        for (int p = 0; p < FLAPPY_NUM_PIPES; p++) {
            assert(a.env.pipes[p].gap_y == gaps[p] && a.env.pipes[p].passed == 0);
            assert(a.env.pipes[p].x == a.env.first_pipe_x + p*a.env.pipe_spacing);
        }
        check_observation(&a); check_observation(&b);
        uint64_t before = flappy_exact_observation_hash(a.buffer+1);
        a.env.tick = 999; a.env.bird_vy = 100; a.env.log.n = 99;
        for (int i = 1; i <= OBS_SIZE; i++) a.buffer[i] = 99;
        flappy_exact_start(&a.env,seeds[s],ids[e]);
        assert(before == flappy_exact_observation_hash(a.buffer+1));
        printf("exact-start %u %u %u %016" PRIx64 "\n",seeds[s],ids[e],expected_rng,
            flappy_exact_world_hash(&a.env));
    }
    flappy_exact_start(&a.env,56111,0);
    float sum = 0;
    for (int decision = 1; decision <= 2; decision++) {
        a.action = 0; puf_step(&a.env); sum += a.reward;
        assert(a.terminal == (decision == 2));
    }
    assert(flappy_exact_terminal(&a.env,2,sum));
    assert(a.env.log.score == 0 && a.env.log.perf == 0);
    assert(!flappy_exact_terminal(&a.env,1,sum));
    assert(!flappy_exact_terminal(&a.env,2,NAN));
    a.env.log.n = 2; assert(!flappy_exact_terminal(&a.env,2,sum)); a.env.log.n = 1;
    printf("exact-cap %016" PRIx64 " %a\n",flappy_exact_world_hash(&a.env),sum);
    flappy_exact_start(&a.env,56111,0);
    a.env.bird_y = a.env.height;
    a.action = 0; puf_step(&a.env);
    assert(a.terminal == 1 && flappy_exact_terminal(&a.env,1,a.reward));
    printf("exact-crash %016" PRIx64 " %a\n",flappy_exact_world_hash(&a.env),a.reward);
    flappy_exact_start(&a.env,56111,0);
    a.env.max_steps = 1;
    for (int p = 0; p < FLAPPY_NUM_PIPES; p++) a.env.pipes[p].gap_y = 320;
    a.env.pipes[0].x = a.env.bird_x - a.env.pipe_width + a.env.pipe_speed - 0.1f;
    a.action = 0; puf_step(&a.env);
    assert(a.terminal == 1 && a.env.log.score == 1 && a.env.log.perf == 0.05f
        && flappy_exact_terminal(&a.env,1,a.reward));
    printf("exact-pass-cap %016" PRIx64 " %a\n",flappy_exact_world_hash(&a.env),a.reward);
    puf_close(&a.env); puf_close(&b.env);
}

int main(int argc, char** argv) {
    if (argc > 1) representation = atof(argv[1]);
    if (argc > 2) representation_mode = atof(argv[2]);
    if (argc > 3) representation_seed = atof(argv[3]);
    if (argc > 4) representation_mix_catalog = atof(argv[4]);
    int pixels = argc > 5 && strcmp(argv[5], "pixels") == 0;
    if (!pixels) {
        event_fixtures();
        exact_episode_fixtures();
#ifdef TEST_PIXELS
        pixel_fixtures();
        geometric_fixtures();
#endif
    }
    int caps[] = {2, 64, 4096};
    for (int cap = 0; cap < 3; cap++) for (unsigned int seed = 1; seed <= 8; seed++) {
        Fixture f; setup(&f, seed, caps[cap]);
#ifdef TEST_PIXELS
        int assigned = f.env.representation;
#endif
        unsigned int rng = seed + 1000;
        for (int step = 0; step < 2048; step++) {
            rng = rng * 1664525u + 1013904223u;
            f.action = rng % 13 == 0;
            puf_step(&f.env); check_observation(&f);
#ifdef TEST_PIXELS
            assert(f.env.representation == assigned);
#endif
            if (pixels) {
                assert(fwrite(f.env.agents[0].observations, sizeof(obs_t), OBS_SIZE, stdout) == OBS_SIZE);
            } else {
                printf("%d %u %d %016" PRIx64 " %a %a\n", caps[cap], seed, step, state_hash(&f), f.reward, f.terminal);
            }
        }
        puf_close(&f.env);
    }
    return 0;
}
