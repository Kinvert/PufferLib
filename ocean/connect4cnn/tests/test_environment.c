// Compile against both headers and compare the complete seeded transition trace.
#include <assert.h>
#include <inttypes.h>
#include ENV_HEADER

typedef struct {
    Env env;
    float buffer[OBS_SIZE + 2];
    float action, reward, terminal;
} Fixture;

static void setup(Fixture* f, unsigned int seed) {
    memset(f, 0, sizeof(*f));
    f->buffer[0] = 12345.0f;
    f->buffer[OBS_SIZE + 1] = -12345.0f;
    f->env.agents[0].observations = f->buffer + 1;
    f->env.agents[0].actions = &f->action;
    f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal;
    Dict kwargs = {0};
    dict_set(&kwargs, "player_pieces", 0);
    dict_set(&kwargs, "env_pieces", 0);
    puf_init(&f->env, &kwargs);
    // These entries have numeric values only; dict_set allocates the item array.
    free(kwargs.items);
    f->env.rng = seed;
    puf_reset(&f->env);
}

static void check_observation(Fixture* f) {
    assert(f->buffer[0] == 12345.0f);
    assert(f->buffer[OBS_SIZE + 1] == -12345.0f);
    const float* obs = f->env.agents[0].observations;
    for (int column = 0; column < 7; column++) {
        for (int row = 0; row < 6; row++) {
            uint64_t bit = UINT64_C(1) << (column * 7 + row);
            int owner = (f->env.player_pieces & bit) ? 1
                      : (f->env.env_pieces & bit) ? -1 : 0;
#ifdef TEST_PIXELS
            float expected = owner == 1 ? 1.0f : owner == -1 ? 0.5f : 0.0f;
            for (int dy = 0; dy < 6; dy++) {
                for (int dx = 0; dx < 6; dx++) {
                    int y = (5 - row) * 6 + dy;
                    int x = 1 + column * 6 + dx;
                    assert(obs[y * OBS_WIDTH + x] == expected);
                }
            }
#else
            assert(obs[column * 6 + row] == (float)owner);
#endif
        }
    }
#ifdef TEST_PIXELS
    assert(OBS_HEIGHT == 36 && OBS_WIDTH == 44 && OBS_SIZE == 1584);
    for (int y = 0; y < OBS_HEIGHT; y++) {
        assert(obs[y * OBS_WIDTH] == 0);
        assert(obs[y * OBS_WIDTH + OBS_WIDTH - 1] == 0);
    }
#endif
}

static void check_fixtures(void) {
    Fixture f;
    setup(&f, 73);
    check_observation(&f);
    // Every location and owner, including all corners, plus dirty-buffer overwrite.
    for (int column = 0; column < 7; column++) {
        for (int row = 0; row < 6; row++) {
            for (int owner = 0; owner < 2; owner++) {
                puf_reset(&f.env);
                uint64_t bit = UINT64_C(1) << (column * 7 + row);
                if (owner) f.env.player_pieces = bit;
                else f.env.env_pieces = bit;
                for (int i = 1; i <= OBS_SIZE; i++) f.buffer[i] = 99;
                compute_observation(&f.env);
                check_observation(&f);
                puf_reset(&f.env);
                check_observation(&f);
            }
        }
    }
    // A player win must expose terminal/reward and the same-step reset image.
    f.env.player_pieces = 7; // three bottom pieces in column zero
    f.action = 0;
    puf_step(&f.env);
    assert(f.terminal == 1 && f.reward == 1 && f.env.log.n == 1);
    assert(f.env.player_pieces == 0 && f.env.env_pieces == 0);
    check_observation(&f);
    // A full column remains an invalid action and ends the episode.
    f.env.player_pieces = 21;
    f.env.env_pieces = 42;
    f.action = 0;
    puf_step(&f.env);
    assert(f.terminal == 1 && f.reward == -1 && f.env.log.invalids == 1);
    check_observation(&f);
    // Terminal image before a viewer-driven delayed reset: no window is opened.
    Client client = {0};
    f.env.client = &client;
    f.env.player_pieces = 7;
    f.action = 0;
    puf_step(&f.env);
    assert(f.terminal == 1 && f.env.pending_reset == 1);
    assert(f.env.player_pieces == 15);
    check_observation(&f);
    f.env.client = NULL;
    puf_reset(&f.env);
    assert(f.terminal == 0 && f.reward == 0 && f.env.pending_reset == 0);
    check_observation(&f);
    puf_close(&f.env);
}

int main(void) {
    check_fixtures();
    for (unsigned int seed = 1; seed <= 16; seed++) {
        Fixture f;
        setup(&f, seed);
        unsigned int action_rng = seed + 1000;
        for (int step = 0; step < 256; step++) {
            // Independent action RNG: policy does not consume the opponent RNG.
            action_rng = action_rng * 1664525u + 1013904223u;
            f.action = (float)(action_rng % 7);
            puf_step(&f.env);
            check_observation(&f);
            printf("%u %d %.0f %" PRIu64 " %" PRIu64 " %" PRIu64
                   " %u %d %d %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
                   seed, step, f.action, f.env.player_pieces, f.env.env_pieces,
                   f.env.last_env_bit, f.env.rng, f.env.tick, f.env.pending_reset,
                   f.reward, f.terminal, f.env.log.perf, f.env.log.score,
                   f.env.log.episode_return, f.env.log.episode_length,
                   f.env.log.n, f.env.log.invalids);
        }
        puf_close(&f.env);
    }
    return 0;
}
