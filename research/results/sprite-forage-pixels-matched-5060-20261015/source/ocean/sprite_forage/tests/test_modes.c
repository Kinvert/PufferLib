// State/pixel equivalence and compact-buffer bounds. No neural computation.
#include <assert.h>
#include <math.h>
#include "../sprite_forage.h"

typedef struct {
    Env env;
    float* storage;
    float action, reward, terminal;
    int size;
} Fixture;

void setup(Fixture* f, int mode, int rep, unsigned int seed, unsigned int world_seed,
        int copies, int trees) {
    memset(f, 0, sizeof(*f));
    Ini ini = {0};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, "config/sprite_forage.ini");
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    dict_set(kwargs, "observation_mode", mode);
    dict_set(kwargs, "state_layout", rep % 3);
    dict_set(kwargs, "representation", rep);
    dict_set(kwargs, "seed", world_seed);
    dict_set(kwargs, "copies_per_type", copies);
    dict_set(kwargs, "trees", trees);
    f->size = puf_observation_size(kwargs);
    int state_size = rep % 3 == 1 ? FORAGE_AREA
        : 2 + 3 * (10 * copies + trees) + 25 * (rep % 3 == 2);
    assert(f->size == (mode ? FORAGE_PIXEL_OBS_SIZE : state_size));
    f->storage = calloc(f->size + 2, sizeof(float));
    assert(f->storage);
    f->storage[0] = 12345;
    f->storage[f->size + 1] = -12345;
    f->env.rng = seed;
    f->env.agents[0].observations = f->storage + 1;
    f->env.agents[0].actions = &f->action;
    f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal;
    puf_init(&f->env, kwargs);
    assert(f->env.rng == seed + world_seed);
    puf_reset(&f->env);
    puf_ini_free(&ini);
}

void check_observation(Fixture* f) {
    assert(f->storage[0] == 12345 && f->storage[f->size + 1] == -12345);
    float* saved = malloc(f->size * sizeof(float));
    assert(saved);
    memcpy(saved, f->storage + 1, f->size * sizeof(float));
    Env before = f->env;
    compute_observations(&f->env);
    assert(!memcmp(&before, &f->env, sizeof(Env)));
    assert(!memcmp(saved, f->storage + 1, f->size * sizeof(float)));
    for (int i = 0; i < f->size; i++) {
        assert(isfinite(saved[i]) && saved[i] >= (f->env.observation_mode ? 0 : -1)
            && saved[i] <= 1);
    }
    if (!f->env.observation_mode) {
        const float values[] = {0, .75f, .25f, .5f, .25f, .25f, .25f, .25f,
            -.25f, -.25f, -.5f, -1, 1};
        if (f->env.state_layout == 1) {
            for (int cell = 0; cell < FORAGE_AREA; cell++) {
                assert(saved[cell] == (cell == f->env.player_idx ? 1 : values[f->env.grid[cell]]));
            }
            free(saved);
            return;
        }
        int px = f->env.player_idx % FORAGE_WIDTH;
        int py = f->env.player_idx / FORAGE_WIDTH;
        assert(saved[0] == px / (float)FORAGE_WIDTH);
        assert(saved[1] == py / (float)FORAGE_HEIGHT);
        unsigned char represented[FORAGE_AREA] = {0};
        int previous_cell[FORAGE_TILE_TYPES] = {0};
        for (int slot = 0; slot < 10 * f->env.copies + f->env.trees; slot++) {
            float present = saved[2 + 3 * slot];
            assert(present == 0 || present == 1);
            if (!present) {
                assert(saved[3 + 3 * slot] == 0 && saved[4 + 3 * slot] == 0);
                continue;
            }
            float x = px + FORAGE_WIDTH * saved[3 + 3 * slot];
            float y = py + FORAGE_HEIGHT * saved[4 + 3 * slot];
            assert(fabsf(x - roundf(x)) < 1e-5f && fabsf(y - roundf(y)) < 1e-5f);
            assert(x >= 1 && x < FORAGE_WIDTH - 1 && y >= 1 && y < FORAGE_HEIGHT - 1);
            int cell = (int)lroundf(y) * FORAGE_WIDTH + (int)lroundf(x);
            int kind = slot < 10 * f->env.copies ? slot / f->env.copies + 1 : FORAGE_TREE;
            assert(f->env.grid[cell] == kind && !represented[cell]);
            represented[cell] = 1;
            assert(cell > previous_cell[kind]);
            previous_cell[kind] = cell;
        }
        if (f->env.state_layout == 2) {
            int offset = 2 + 3 * (10 * f->env.copies + f->env.trees);
            for (int y = -2; y <= 2; y++) {
                for (int x = -2; x <= 2; x++) {
                    int sx = px + x, sy = py + y;
                    float expected = sx < 0 || sx >= FORAGE_WIDTH || sy < 0 || sy >= FORAGE_HEIGHT
                        ? -1 : !x && !y ? 1 : values[f->env.grid[sy * FORAGE_WIDTH + sx]];
                    assert(saved[offset++] == expected);
                }
            }
        }
        for (int y = 1; y < FORAGE_HEIGHT - 1; y++) {
            for (int x = 1; x < FORAGE_WIDTH - 1; x++) {
                int cell = y * FORAGE_WIDTH + x;
                assert(represented[cell] == (f->env.grid[cell] != 0));
            }
        }
    }
    free(saved);
}

void same_world(Fixture* a, Fixture* b) {
    assert(a->env.rng == b->env.rng && a->env.player_idx == b->env.player_idx);
    assert(a->env.tick == b->env.tick && a->env.apples == b->env.apples);
    assert(a->env.negatives == b->env.negatives);
    assert(a->env.episode_return == b->env.episode_return);
    assert(!memcmp(a->env.grid, b->env.grid, FORAGE_AREA));
    assert(!memcmp(&a->env.log, &b->env.log, sizeof(Log)));
    assert(a->reward == b->reward && a->terminal == b->terminal);
}

int main(void) {
    const unsigned int seeds[] = {7, 42, 69, 420, 1337};
    int episodes = 0;
    for (int seed = 0; seed < 5; seed++) {
        Fixture cases[2 * FORAGE_REPRESENTATIONS];
        for (int i = 0; i < 2 * FORAGE_REPRESENTATIONS; i++) {
            setup(&cases[i], i / FORAGE_REPRESENTATIONS, i % FORAGE_REPRESENTATIONS,
                seeds[seed], 0, 3, 8);
            check_observation(&cases[i]);
            same_world(&cases[0], &cases[i]);
        }
        unsigned int sequence = 9001;
        for (int step = 0; step < 2048; step++) {
            sequence = sequence * 1664525u + 1013904223u;
            for (int i = 2 * FORAGE_REPRESENTATIONS - 1; i >= 0; i--) {
                cases[i].action = (sequence >> 16) % 5;
                puf_step(&cases[i].env);
                same_world(&cases[2 * FORAGE_REPRESENTATIONS - 1], &cases[i]);
                if (step % 97 == 0 || cases[i].terminal) {
                    check_observation(&cases[i]);
                }
            }
            episodes += cases[0].terminal != 0;
        }
        for (int i = 0; i < 2 * FORAGE_REPRESENTATIONS; i++) {
            check_observation(&cases[i]);
            puf_close(&cases[i].env);
            free(cases[i].storage);
        }
    }
    assert(episodes >= 20);
    Fixture shifted, direct;
    setup(&shifted, 0, 0, 7, 420, 3, 8);
    setup(&direct, 1, 4, 427, 0, 3, 8);
    same_world(&shifted, &direct);
    puf_close(&shifted.env);
    puf_close(&direct.env);
    free(shifted.storage);
    free(direct.storage);
    // The INI controls state-buffer length: exercise its smallest/largest layouts.
    for (int extreme = 0; extreme < 2; extreme++) {
        Fixture pair[2];
        for (int mode = 0; mode < 2; mode++) {
            setup(&pair[mode], mode, 0, 7, 0, extreme ? 8 : 1, extreme ? 48 : 0);
        }
        unsigned int sequence = 1337;
        for (int step = 0; step < 128; step++) {
            sequence = sequence * 1664525u + 1013904223u;
            for (int mode = 0; mode < 2; mode++) {
                pair[mode].action = (sequence >> 16) % 5;
                puf_step(&pair[mode].env);
                check_observation(&pair[mode]);
            }
            same_world(&pair[0], &pair[1]);
        }
        for (int mode = 0; mode < 2; mode++) {
            puf_close(&pair[mode].env);
            free(pair[mode].storage);
        }
    }
    // Different actions/episode lengths must not change the next assigned world.
    Fixture short_episode, long_episode;
    setup(&short_episode, 0, 0, 0, 2027, 3, 8);
    setup(&long_episode, 1, 4, 0, 2027, 3, 8);
    short_episode.env.max_steps = 1;
    long_episode.env.max_steps = 17;
    unsigned int actions = 9001;
    for (int episode = 0; episode < 100; episode++) {
        short_episode.action = 0;
        puf_step(&short_episode.env);
        assert(short_episode.terminal);
        do {
            actions = actions * 1664525u + 1013904223u;
            long_episode.action = (actions >> 16) % 5;
            puf_step(&long_episode.env);
        } while (!long_episode.terminal);
        assert(short_episode.env.tick == 0 && long_episode.env.tick == 0);
        assert(short_episode.env.rng == long_episode.env.rng);
        assert(short_episode.env.player_idx == long_episode.env.player_idx);
        assert(!memcmp(short_episode.env.grid, long_episode.env.grid, FORAGE_AREA));
    }
    puf_close(&short_episode.env);
    puf_close(&long_episode.env);
    free(short_episode.storage);
    free(long_episode.storage);
    printf("State/pixels: 5 seeds, all 5 skins, 10240 shared decisions/102400 env steps, "
        "%d paired endings; "
        "exact worlds/rewards/logs, scalar grids/row coordinates/local views, dirty/full observations "
        "and guarded compact buffers PASS\n", episodes);
    puts("State layouts: 32/386-float extremes, 256 additional paired decisions PASS");
    puts("Assigned worlds: 100 episode resets match across different actions/lengths/modes PASS");
}
