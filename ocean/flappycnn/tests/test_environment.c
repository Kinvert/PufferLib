// Environment/pixel fixtures, not CPU model training or CNN validation.
#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include ENV_HEADER

static double representation, representation_mode, representation_seed;

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
static void expect_pixel(Fixture* f, int x, int y, float value) {
    int rep = f->env.representation;
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
#endif

int main(int argc, char** argv) {
    if (argc > 1) representation = atof(argv[1]);
    if (argc > 2) representation_mode = atof(argv[2]);
    if (argc > 3) representation_seed = atof(argv[3]);
    event_fixtures();
#ifdef TEST_PIXELS
    pixel_fixtures();
#endif
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
            printf("%d %u %d %016" PRIx64 " %a %a\n", caps[cap], seed, step, state_hash(&f), f.reward, f.terminal);
        }
        puf_close(&f.env);
    }
    return 0;
}
