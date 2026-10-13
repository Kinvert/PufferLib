// Compile the same checks against upstream Pong and PongCNN, then compare traces.
#include <assert.h>
#include <inttypes.h>
#include ENV_HEADER
#include <stddef.h>

static double representation = 0;
static double representation_mode = 0;
static double representation_seed = 0;
static double representation_mix_catalog = 0;

typedef struct {
    Env env;
    float buffer[OBS_SIZE + 2];
    float action, reward, terminal;
} Fixture;

static void setup(Fixture* f, unsigned int seed, int frameskip, int continuous) {
    memset(f, 0, sizeof(*f));
    f->buffer[0] = 12345;
    f->buffer[OBS_SIZE + 1] = -12345;
    f->env.agents[0].observations = f->buffer + 1;
    f->env.agents[0].actions = &f->action;
    f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal;
    Dict kwargs = {0};
    dict_set(&kwargs, "width", 500);
    dict_set(&kwargs, "height", 640);
    dict_set(&kwargs, "paddle_width", 20);
    dict_set(&kwargs, "paddle_height", 70);
    dict_set(&kwargs, "ball_width", 32);
    dict_set(&kwargs, "ball_height", 32);
    dict_set(&kwargs, "paddle_speed", 8);
    dict_set(&kwargs, "ball_initial_speed_x", 10);
    dict_set(&kwargs, "ball_initial_speed_y", 1);
    dict_set(&kwargs, "ball_speed_y_increment", 3);
    dict_set(&kwargs, "ball_max_speed_y", 13);
    dict_set(&kwargs, "max_score", 21);
    dict_set(&kwargs, "frameskip", frameskip);
    dict_set(&kwargs, "continuous", continuous);
#ifdef TEST_PIXELS
    dict_set(&kwargs, "representation", representation);
    dict_set(&kwargs, "representation_mode", representation_mode);
    dict_set(&kwargs, "representation_seed", representation_seed);
    dict_set(&kwargs, "representation_mix_catalog", representation_mix_catalog);
#endif
    f->env.rng = seed;
    puf_init(&f->env, &kwargs);
    assert(f->env.rng == seed);
    free(kwargs.items);
    f->env.rng = seed;
    puf_reset(&f->env);
}

static void check_observation(Fixture* f) {
    assert(f->buffer[0] == 12345 && f->buffer[OBS_SIZE + 1] == -12345);
    float* obs = f->env.agents[0].observations;
    for (int i = 0; i < OBS_SIZE; i++) {
        assert(isfinite(obs[i]));
#ifdef TEST_PIXELS
        assert(obs[i] >= 0 && obs[i] <= 1);
#endif
    }
#ifdef TEST_PIXELS
    float left = 0, right = 0;
    for (int x = 0; x < OBS_WIDTH; x++) {
        left += obs[x];
        right += obs[OBS_WIDTH + x];
    }
    assert(fabsf(left / OBS_WIDTH - (float)f->env.score_l / f->env.max_score) < 1e-6f);
    assert(fabsf(right / OBS_WIDTH - (float)f->env.score_r / f->env.max_score) < 1e-6f);
#endif
    // Observation generation consumes no RNG and fully overwrites a dirty buffer.
    float saved[OBS_SIZE];
    memcpy(saved, obs, sizeof(saved));
    unsigned int rng = f->env.rng;
    for (int i = 0; i < OBS_SIZE; i++) obs[i] = 99;
    compute_observations(&f->env);
    assert(memcmp(saved, obs, sizeof(saved)) == 0 && rng == f->env.rng);
#ifdef TEST_PIXELS
    if (f->env.representation >= 5) {
        int id = f->env.representation;
        f->env.representation = 0;
        compute_observations(&f->env);
        const float decoded[] = {0, 0.5f, 0.75f, 1};
        for (int i = 0; i < OBS_SIZE; i++) {
            if (i < PONGCNN_SCORE_ROWS * OBS_WIDTH) assert(saved[i] == obs[i]);
            else {
                int c = (int)floorf(4 * saved[i]);
                assert(c >= 0 && c < 4 && decoded[c] == obs[i]);
            }
        }
        f->env.representation = id;
        compute_observations(&f->env);
        assert(memcmp(saved, obs, sizeof(saved)) == 0 && rng == f->env.rng);
    }
#endif
}

#ifdef TEST_PIXELS
static float fixture_pixel(int x, int y, int bx, int by) {
    if (x < 0 || x >= 44 || y < 2 || y >= 36) return 0;
    if (x >= bx && x < bx + 2 && y >= by && y < by + 2) return 1;
    return x < 2 && y >= 33 ? 0.5f : x >= 42 && y < 5 ? 0.75f : 0;
}
#endif

static void fixtures(void) {
    Fixture f;
    setup(&f, 73, 1, 0);
    check_observation(&f);
    // Force both sides' point/terminal paths, including the same-step reset.
    for (int side = 0; side < 2; side++) {
        for (int terminal = 0; terminal < 2; terminal++) {
            puf_reset(&f.env);
            f.env.score_l = side && terminal ? 20 : 0;
            f.env.score_r = !side && terminal ? 20 : 0;
            f.env.ball_x = side ? 500 : -32;
            f.env.ball_vx = side ? 10 : -10;
            f.env.ball_y = 20;
            f.env.ball_vy = 0;
            f.env.paddle_yl = f.env.paddle_yr = 500;
            f.action = 0;
            puf_step(&f.env);
            assert(f.reward == (side ? -1 : 1));
            assert(f.terminal == terminal);
            assert(f.env.score_l == (unsigned int)(side && !terminal));
            assert(f.env.score_r == (unsigned int)(!side && !terminal));
            check_observation(&f);
        }
    }
#ifdef TEST_PIXELS
    assert(OBS_CHANNELS == 1 && OBS_HEIGHT == 36 && OBS_WIDTH == 44);
    // Unit pixel/world scale makes corner, orientation and clipping fixtures exact.
    f.env.width = 40;
    f.env.height = 34;
    f.env.paddle_height = 6;
    f.env.ball_width = f.env.ball_height = 2;
    f.env.paddle_yl = -3;
    f.env.paddle_yr = 31;
    f.env.score_l = 7;
    f.env.score_r = 14;
    float* obs = f.env.agents[0].observations;
    for (int corner = 0; corner < 4; corner++) {
        f.env.ball_x = corner & 1 ? 38 : 0;
        f.env.ball_y = corner & 2 ? 32 : 0;
        compute_observations(&f.env);
        check_observation(&f);
        int bx = corner & 1 ? 40 : 2;
        int by = corner & 2 ? 2 : 34;
        for (int y = 2; y < 36; y++) {
            for (int x = 0; x < 44; x++) {
                float expected = x < 2 && y >= 33 ? 0.5f
                    : x >= 42 && y < 5 ? 0.75f : 0;
                if (x >= bx && x < bx + 2 && y >= by && y < by + 2) expected = 1;
                int tx = f.env.representation == 2 ? 43 - x : x;
                int ty = f.env.representation == 3 ? 37 - y : y;
                if (f.env.representation == 1) {
                    if (expected == 0.5f) expected = 0.75f;
                    else if (expected == 0.75f) expected = 0.5f;
                } else if (f.env.representation == 4) expected = 1 - expected;
                else if (f.env.representation >= 5) {
                    int c = expected == 0 ? 0 : expected == 0.5f ? 1 : expected == 0.75f ? 2 : 3;
                    int bit = (x + y) & 1;
                    if (f.env.representation == 6) {
                        bit = fixture_pixel(x - 1, y, bx, by) != expected
                            || fixture_pixel(x + 1, y, bx, by) != expected
                            || fixture_pixel(x, y - 1, bx, by) != expected
                            || fixture_pixel(x, y + 1, bx, by) != expected;
                    }
                    expected = 0.125f + 0.25f * c + 0.0625f * bit;
                }
                assert(obs[ty * 44 + tx] == expected);
            }
        }
        float saved[OBS_SIZE];
        memcpy(saved, obs, sizeof(saved));
        f.env.ball_vx = -f.env.ball_vx;
        f.env.ball_vy = 13;
        compute_observations(&f.env);
        assert(memcmp(saved, obs, sizeof(saved)) == 0);
    }
    // Transient physics overshoot must stay inside the observation allocation.
    f.env.ball_x = -100;
    f.env.ball_y = 100;
    compute_observations(&f.env);
    check_observation(&f);
#endif
    puf_close(&f.env);
}

static uint64_t state_hash(Fixture* f) {
    Env state = f->env;
    state.client = NULL;
    memset(state.agents, 0, sizeof(state.agents));
    const unsigned char* bytes = (const unsigned char*)&state;
    uint64_t hash = UINT64_C(14695981039346656037);
    // Compare the unchanged game fields, excluding appearance and tail padding.
    for (size_t i = 0; i < offsetof(Env, rng) + sizeof(state.rng); i++) {
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

int main(int argc, char** argv) {
    if (argc > 1) representation = atof(argv[1]);
    if (argc > 2) representation_mode = atof(argv[2]);
    if (argc > 3) representation_seed = atof(argv[3]);
    if (argc > 4) representation_mix_catalog = atof(argv[4]);
    fixtures();
    int skips[] = {1, 3, 8};
    for (int continuous = 0; continuous <= 1; continuous++) {
        for (int k = 0; k < 3; k++) {
            for (unsigned int seed = 1; seed <= 8; seed++) {
                Fixture f;
                setup(&f, seed, skips[k], continuous);
#ifdef TEST_PIXELS
                int assigned = f.env.representation;
#endif
                unsigned int rng = seed + 1000;
                for (int step = 0; step < 1024; step++) {
                    rng = rng * 1664525u + 1013904223u;
                    f.action = continuous ? (float)(rng % 201) / 100 - 1 : (float)(rng % 3);
                    puf_step(&f.env);
                    check_observation(&f);
#ifdef TEST_PIXELS
                    assert(f.env.representation == assigned);
#endif
                    printf("%d %d %u %d %016" PRIx64 " %a %a\n", continuous, skips[k],
                        seed, step, state_hash(&f), f.reward, f.terminal);
                }
                puf_close(&f.env);
            }
        }
    }
    return 0;
}
