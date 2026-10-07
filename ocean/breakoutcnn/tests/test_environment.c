// Native environment/pixel checks only. No policy or CPU neural execution.
#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <string.h>
#include ENV_HEADER

static double representation, representation_mode, representation_seed;
typedef struct {
    Env env;
    float buffer[OBS_SIZE + 2];
    float action, reward, terminal;
} Fixture;

static void setup(Fixture* f, unsigned int seed, int frameskip) {
    memset(f, 0, sizeof(*f));
    f->buffer[0] = 12345; f->buffer[OBS_SIZE + 1] = -12345;
    f->env.agents[0].observations = f->buffer + 1;
    f->env.agents[0].actions = &f->action;
    f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal;
    f->env.rng = seed;
    Dict kwargs = {0};
    dict_set(&kwargs, "frameskip", frameskip);
    dict_set(&kwargs, "width", 576); dict_set(&kwargs, "height", 330);
    dict_set(&kwargs, "paddle_width", 62); dict_set(&kwargs, "paddle_height", 8);
    dict_set(&kwargs, "ball_width", 32); dict_set(&kwargs, "ball_height", 32);
    dict_set(&kwargs, "brick_width", 32); dict_set(&kwargs, "brick_height", 12);
    dict_set(&kwargs, "brick_rows", 6); dict_set(&kwargs, "brick_cols", 18);
    dict_set(&kwargs, "initial_ball_speed", 256); dict_set(&kwargs, "max_ball_speed", 448);
    dict_set(&kwargs, "paddle_speed", 620); dict_set(&kwargs, "continuous", 0);
#ifdef TEST_PIXELS
    dict_set(&kwargs, "representation", representation);
    dict_set(&kwargs, "representation_mode", representation_mode);
    dict_set(&kwargs, "representation_seed", representation_seed);
#endif
    puf_init(&f->env, &kwargs);
    assert(f->env.rng == seed);
    dict_clear(&kwargs);
    puf_reset(&f->env);
}

static void check_observation(Fixture* f) {
    assert(f->buffer[0] == 12345 && f->buffer[OBS_SIZE + 1] == -12345);
    float* obs = f->env.agents[0].observations;
    for (int i = 0; i < OBS_SIZE; i++) {
        assert(isfinite(obs[i]));
#ifdef TEST_PIXELS
        assert(obs[i] == 0 || obs[i] == 0.25f || obs[i] == 0.5f || obs[i] == 0.75f || obs[i] == 1);
#endif
    }
    float saved[OBS_SIZE]; memcpy(saved, obs, sizeof(saved));
    unsigned int rng = f->env.rng;
    for (int i = 0; i < OBS_SIZE; i++) obs[i] = 99;
    compute_observations(&f->env);
    assert(memcmp(saved, obs, sizeof(saved)) == 0 && rng == f->env.rng);
}

static uint64_t state_hash(Fixture* f) {
    Env state = f->env;
    state.client = NULL; memset(state.agents, 0, sizeof(state.agents));
    state.brick_x = NULL; state.brick_y = NULL; state.brick_states = NULL;
    const unsigned char* bytes = (const unsigned char*)&state;
    uint64_t hash = UINT64_C(14695981039346656037);
    // Equal original fields, excluding pixel appearance and tail padding.
    for (size_t i = 0; i < offsetof(Env, rng) + sizeof(state.rng); i++)
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    float* arrays[] = {f->env.brick_x, f->env.brick_y, f->env.brick_states};
    for (int a = 0; a < 3; a++) {
        bytes = (const unsigned char*)arrays[a];
        for (size_t i = 0; i < f->env.num_bricks * sizeof(float); i++)
            hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

static void events(void) {
    Fixture f; setup(&f, 73, 1);
    assert(f.env.num_bricks == 108 && f.env.num_balls == 5);
    f.action = RIGHT;
    puf_step(&f.env); // The stock first frame launches and ignores discrete movement.
    assert(f.env.balls_fired == 1 && f.terminal == 0);
    check_observation(&f);
    printf("launch %016" PRIx64 " %a %a\n", state_hash(&f), f.reward, f.terminal);
    destroy_brick(&f.env, 0);
    assert(f.env.brick_states[0] == 1 && f.env.score == 7 && f.reward == 7);
    compute_observations(&f.env); check_observation(&f);
    printf("brick %016" PRIx64 " %a\n", state_hash(&f), f.reward);

    f.env.ball_x = 0; f.env.ball_y = f.env.height + 100;
    f.env.ball_vx = 0; f.env.ball_vy = 1;
    puf_step(&f.env);
    assert(f.env.num_balls == 4 && f.terminal == 0 && f.env.balls_fired == 0);
    printf("life %016" PRIx64 "\n", state_hash(&f));
    f.env.num_balls = 0; f.env.balls_fired = 1;
    f.env.ball_x = 0; f.env.ball_y = f.env.height + 100;
    f.env.ball_vx = 0; f.env.ball_vy = 1;
    puf_step(&f.env);
    assert(f.terminal == 1 && f.env.num_balls == 5 && f.env.score == 0 && f.env.log.n == 1);
    check_observation(&f);
    printf("terminal %016" PRIx64 "\n", state_hash(&f));
    puf_close(&f.env);
}

#ifdef TEST_PIXELS
static void expect(Fixture* f, int x, int y, float value) {
    int rep = f->env.representation;
    if (rep == 1) x = OBS_WIDTH - 1 - x;
    if (rep == 2) y = OBS_HEIGHT - 1 - y;
    if (rep == 3) value = 1 - value;
    if (rep == 4) {
        if (value == 0.5f) value = 0.75f;
        else if (value == 0.75f) value = 0.5f;
    }
    assert(f->env.agents[0].observations[y * OBS_WIDTH + x] == value);
}

static void pixels(void) {
    Fixture f; setup(&f, 73, 3);
    assert(OBS_CHANNELS == 1 && OBS_HEIGHT == 36 && OBS_WIDTH == 44);
    f.env.width = 44; f.env.height = 36;
    for (int i = 0; i < f.env.num_bricks; i++) f.env.brick_states[i] = 1;
    f.env.brick_states[0] = 0;
    f.env.brick_x[0] = -1.5f; f.env.brick_y[0] = 2.5f;
    f.env.brick_width = 4; f.env.brick_height = 2;
    f.env.paddle_x = 40; f.env.paddle_y = 34;
    f.env.paddle_width = 8; f.env.paddle_height = 4;
    f.env.ball_x = 1; f.env.ball_y = 3;
    f.env.ball_width = 2; f.env.ball_height = 2;
    compute_observations(&f.env); check_observation(&f);
    for (int y = 0; y < 36; y++) for (int x = 0; x < 44; x++) {
        float value = x < 3 && y >= 2 && y < 5 ? 0.5f : 0;
        if (x >= 40 && y >= 34) value = 0.75f;
        if (x >= 1 && x < 3 && y >= 3 && y < 5) value = 1;
        expect(&f, x, y, value);
    }
    float saved[OBS_SIZE]; memcpy(saved, f.buffer + 1, sizeof(saved));
    f.env.ball_vx = -100; f.env.ball_vy = 100;
    f.env.balls_fired = 999; f.env.num_balls = 999; f.env.score = 999; f.env.tick = 999;
    compute_observations(&f.env);
    assert(memcmp(saved, f.buffer + 1, sizeof(saved)) == 0);
    // Removing a brick clears its old pixels; no ghost contents from the prior frame.
    f.env.brick_states[0] = 1;
    compute_observations(&f.env); check_observation(&f);
    expect(&f, 0, 2, 0); expect(&f, 1, 3, 1);
    // Fully offscreen geometry contributes no pixels.
    f.env.paddle_x = 50; f.env.ball_x = -5;
    compute_observations(&f.env); check_observation(&f);
    for (int y = 0; y < 36; y++) for (int x = 0; x < 44; x++) expect(&f, x, y, 0);
    puf_close(&f.env);
}
#endif

int main(int argc, char** argv) {
    if (argc > 1) representation = strtod(argv[1], NULL);
    if (argc > 2) representation_mode = strtod(argv[2], NULL);
    if (argc > 3) representation_seed = strtod(argv[3], NULL);
    events();
#ifdef TEST_PIXELS
    pixels();
#endif
    unsigned int seeds[] = {0, 73, UINT32_MAX};
    int skips[] = {1, 3, 8};
    for (int s = 0; s < 3; s++) for (int k = 0; k < 3; k++) {
        Fixture f; setup(&f, seeds[s], skips[k]);
        unsigned int actions = 173;
        for (int t = 0; t < 2048; t++) {
            f.action = rand_r(&actions) % 3;
            puf_step(&f.env); check_observation(&f);
            printf("%u %d %d %016" PRIx64 " %a %a\n", seeds[s], skips[k], t,
                state_hash(&f), f.reward, f.terminal);
        }
        puf_close(&f.env);
    }
    return 0;
}
