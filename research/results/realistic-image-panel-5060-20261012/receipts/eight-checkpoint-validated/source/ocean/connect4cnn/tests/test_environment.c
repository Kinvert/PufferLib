// Compile against both headers and compare the complete seeded transition trace.
#include <assert.h>
#include <inttypes.h>
#include ENV_HEADER

static double representation = 0;
static double representation_mode = 0;
static double representation_seed = 0;

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
#ifdef TEST_PIXELS
    // Omitting zero also tests backward compatibility with old configs.
    if (representation != 0) dict_set(&kwargs, "representation", representation);
    dict_set(&kwargs, "representation_mode", representation_mode);
    dict_set(&kwargs, "representation_seed", representation_seed);
#endif
    f->env.rng = seed;
    puf_init(&f->env, &kwargs);
    assert(f->env.rng == seed);
    // These entries have numeric values only; dict_set allocates the item array.
    free(kwargs.items);
    f->env.rng = seed;
    puf_reset(&f->env);
}

static void check_observation(Fixture* f) {
    assert(f->buffer[0] == 12345.0f);
    assert(f->buffer[OBS_SIZE + 1] == -12345.0f);
    const float* obs = f->env.agents[0].observations;
#ifdef TEST_PIXELS
    if (f->env.representation != 0) {
        // Independent literal masks: bit 0 is the leftmost pixel in each row.
        const int widths[] = {6,6,6,6,6,6,4,2,1,2};
        const int heights[] = {6,6,6,6,6,6,4,2,1,4};
        const unsigned char masks[][6] = {
            {63,63,63,63,63,63}, {0,30,30,30,30,0}, {0,0,12,12,0,0},
            {30,63,63,63,63,30}, {0,12,30,30,12,0}, {33,18,12,12,18,33},
            {15,15,15,15}, {3,3}, {1}, {3,3,3,3},
        };
        const unsigned char ring[] = {30,51,33,33,51,30};
        int id = f->env.representation, cw = widths[id], ch = heights[id];
        int left = (44 - 7 * cw) / 2, top = (36 - 6 * ch) / 2;
        float expected[OBS_SIZE] = {0};
        for (int column = 0; column < 7; column++) {
            for (int row = 0; row < 6; row++) {
                uint64_t bit = UINT64_C(1) << (column * 7 + row);
                float value = f->env.player_pieces & bit ? 1 : f->env.env_pieces & bit ? 0.5f : 0;
                const unsigned char* mask = id == 5 && value == 0.5f ? ring : masks[id];
                for (int y = 0; y < ch; y++) {
                    for (int x = 0; x < cw; x++) {
                        if ((mask[y] >> x) & 1) expected[(top + (5-row)*ch + y)*44 + left + column*cw + x] = value;
                    }
                }
            }
        }
        assert(memcmp(obs, expected, sizeof(expected)) == 0);
        return;
    }
#endif
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

static void check_draw_rules(void) {
    // Build occupancy from board coordinates, independently of draw's bit mask.
    uint64_t full = 0;
    for (int column = 0; column < 7; column++) {
        for (int row = 0; row < 6; row++) {
            full |= UINT64_C(1) << (column * 7 + row);
        }
    }
    assert(draw(full));
    assert(!draw(0));
    for (int column = 0; column < 7; column++) {
        for (int row = 0; row < 6; row++) {
            assert(!draw(full ^ (UINT64_C(1) << (column * 7 + row))));
        }
        for (int other = column + 1; other < 7; other++) {
            assert(!draw((UINT64_C(1) << (column * 7))
                | (UINT64_C(1) << (other * 7))));
        }
    }
    // The former literal falsely ended some openings in the last two columns.
    for (unsigned int seed = 0; seed < 32; seed++) {
        Fixture f;
        setup(&f, seed);
        f.action = 6;
        puf_step(&f.env);
        assert(f.terminal == 0 && f.reward == 0 && f.env.tick == 1);
        assert(f.env.log.n == 0);
        assert(__builtin_popcountll(f.env.player_pieces | f.env.env_pieces) == 2);
        puf_close(&f.env);
    }
    // Bottom-to-top rows of a full board without any four-in-a-row.
    const char* rows[] = {"XXOOXXO", "OOXXOOX", "XXOOXXO",
                          "OOXXOOX", "XXOOXXO", "OOXXOOX"};
    Fixture f;
    setup(&f, 73);
    for (int row = 0; row < 6; row++) {
        for (int column = 0; column < 7; column++) {
            uint64_t bit = UINT64_C(1) << (column * 7 + row);
            if (rows[row][column] == 'O') f.env.player_pieces |= bit;
            else f.env.env_pieces |= bit;
        }
    }
    assert(!won(f.env.player_pieces) && !won(f.env.env_pieces));
    assert((f.env.player_pieces | f.env.env_pieces) == full);
    // Only column six is playable: the final player/opponent pair fills it.
    f.env.player_pieces ^= UINT64_C(1) << (6 * 7 + 4);
    f.env.env_pieces ^= UINT64_C(1) << (6 * 7 + 5);
    f.env.tick = 20;
    f.action = 6;
    puf_step(&f.env);
    assert(f.terminal == 1 && f.reward == 0);
    assert(f.env.log.n == 1 && f.env.log.episode_length == 21);
    assert(f.env.log.perf == 0 && f.env.log.score == 0 && f.env.log.invalids == 0);
    assert(f.env.player_pieces == 0 && f.env.env_pieces == 0);
    check_observation(&f);
    puf_close(&f.env);
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

int main(int argc, char** argv) {
    if (argc > 1) representation = atof(argv[1]);
    if (argc > 2) representation_mode = atof(argv[2]);
    if (argc > 3) representation_seed = atof(argv[3]);
    check_draw_rules();
    check_fixtures();
    for (unsigned int seed = 1; seed <= 16; seed++) {
        Fixture f;
        setup(&f, seed);
#ifdef TEST_PIXELS
        int assigned = f.env.representation;
#endif
        unsigned int action_rng = seed + 1000;
        int episode_steps = 0;
        int completed_games = 0;
        for (int step = 0; step < 256; step++) {
            // Independent action RNG: policy does not consume the opponent RNG.
            action_rng = action_rng * 1664525u + 1013904223u;
            f.action = (float)(action_rng % 7);
            int pieces_before = __builtin_popcountll(f.env.player_pieces | f.env.env_pieces);
            episode_steps++;
            puf_step(&f.env);
            assert(episode_steps <= 21);
            if (f.terminal) {
                completed_games++;
                assert(f.env.player_pieces == 0 && f.env.env_pieces == 0);
                assert(f.reward == -1 || f.reward == 0 || f.reward == 1);
                episode_steps = 0;
            } else {
                // Two legal moves per nonterminal call; a full board must end.
                int pieces_after = __builtin_popcountll(f.env.player_pieces | f.env.env_pieces);
                assert(pieces_after == pieces_before + 2 && pieces_after < 42);
                assert(episode_steps < 21);
            }
            assert(f.env.log.n == completed_games);
            check_observation(&f);
#ifdef TEST_PIXELS
            assert(f.env.representation == assigned);
#endif
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
