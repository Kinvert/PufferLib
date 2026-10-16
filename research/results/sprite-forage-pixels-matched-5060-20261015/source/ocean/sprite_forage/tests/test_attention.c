// World/raster checks only; no CPU neural computation.
#include <assert.h>
#define FORAGE_ATTENTION
#include "../sprite_forage.h"

typedef struct {
    Env env;
    float* storage;
    float action[NUM_ATNS], reward, terminal;
    int size;
} Fixture;

static void setup(Fixture* f, int partial, int pixels, int skin) {
    memset(f, 0, sizeof(*f));
    Ini ini = {0};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, "config/sprite_forage.ini");
    Dict* args = puf_ini_section(&ini, "env", 0);
    dict_set(args, "partial_observation", partial);
    dict_set(args, "observation_mode", pixels);
    dict_set(args, "representation", skin);
    f->size = puf_observation_size(args);
    assert(f->size == (partial ? (pixels ? 52228 : FORAGE_STATE_VALUES + 4) : (pixels ? 921600 : 141)));
    f->storage = calloc(f->size + 2, sizeof(float));
    assert(f->storage);
    f->storage[0] = 12345;
    f->storage[f->size + 1] = -12345;
    f->env.rng = 7;
    f->env.agents[0].observations = f->storage + 1;
    f->env.agents[0].actions = f->action;
    f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal;
    puf_init(&f->env, args);
    puf_reset(&f->env);
    puf_ini_free(&ini);
}

// Independent row-major view construction, with local center omitted.
static void cells(Env* e, int output[17]) {
    int n = 0;
    for (int view = 0; view < 2; view++) {
        int center = view ? e->gaze_idx : e->player_idx;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (!view && !dx && !dy) {
                    continue;
                }
                int x = center % FORAGE_WIDTH + dx;
                int y = center / FORAGE_WIDTH + dy;
                output[n++] = x < 0 || x >= FORAGE_WIDTH || y < 0 || y >= FORAGE_HEIGHT
                    ? -1 : y * FORAGE_WIDTH + x;
            }
        }
    }
    assert(n == 17);
}

static void views(Fixture* state, Fixture* pixels, float* full) {
    int indices[17];
    cells(&pixels->env, indices);
    Env shadow = pixels->env;
    shadow.partial_observation = 0;
    shadow.agents[0].observations = full;
    compute_observations(&shadow);
    for (int i = 0; i < 17; i++) {
        int cell = indices[i];
        assert(forage_attention_cell(&pixels->env, i) == cell);
        int kind = cell < 0 ? FORAGE_TREE
            : cell == pixels->env.player_idx ? FORAGE_PLAYER : pixels->env.grid[cell];
#if FORAGE_STATE_ONEHOT
        for (int code = 0; code < FORAGE_TILE_TYPES; code++) {
            assert(state->storage[1 + i * FORAGE_TILE_TYPES + code] == (code == kind));
        }
#else
        assert(state->storage[i + 1] == kind / (float)(FORAGE_TILE_TYPES - 1));
#endif
        float* tile = pixels->storage + 1 + i * FORAGE_TILE_PIXELS;
        if (cell < 0) {
            assert(!memcmp(tile, forage_tiles[pixels->env.representation][FORAGE_TREE],
                FORAGE_TILE_PIXELS * sizeof(float)));
            continue;
        }
        for (int y = 0; y < FORAGE_CELL; y++) {
            int start = ((cell / FORAGE_WIDTH * FORAGE_CELL + y) * OBS_WIDTH
                + cell % FORAGE_WIDTH * FORAGE_CELL) * 3;
            assert(!memcmp(tile + y * FORAGE_CELL * 3, full + start,
                FORAGE_CELL * 3 * sizeof(float)));
        }
    }
    float* geometry = state->storage + 1 + FORAGE_STATE_VALUES;
    assert(!memcmp(geometry, pixels->storage + 1 + FORAGE_VIEW_PIXELS, 4*sizeof(float)));
    int px = state->env.player_idx % FORAGE_WIDTH;
    int py = state->env.player_idx / FORAGE_WIDTH;
    int gx = state->env.gaze_idx % FORAGE_WIDTH;
    int gy = state->env.gaze_idx / FORAGE_WIDTH;
    assert(geometry[0] == px / 19.f && geometry[1] == py / 14.f);
    assert(geometry[2] == (gx - FORAGE_RELATIVE_GAZE * px) / 19.f);
    assert(geometry[3] == (gy - FORAGE_RELATIVE_GAZE * py) / 14.f);
    assert(state->storage[0] == 12345 && state->storage[state->size + 1] == -12345);
    assert(pixels->storage[0] == 12345 && pixels->storage[pixels->size + 1] == -12345);
}

int main(void) {
    int actions[] = ACT_SIZES;
    assert(actions[0] == 5);
#if FORAGE_GAZE_MODE == 2
    assert(NUM_ATNS == 3 && actions[1] == 20 && actions[2] == 15);
#else
    assert(NUM_ATNS == 2 && actions[1] == (FORAGE_GAZE_MODE ? 5 : 300));
#endif
    float* full = calloc(FORAGE_PIXEL_OBS_SIZE, sizeof(float));
    assert(full);
    for (int skin = 0; skin < FORAGE_REPRESENTATIONS; skin++) {
        Fixture f[4];
        for (int i = 0; i < 4; i++) {
            setup(&f[i], i >= 2, i % 2, skin);
        }
        unsigned int sequence = 69;
        for (int step = 0; step < 1200; step++) {
            sequence = sequence * 1664525u + 1013904223u;
            for (int i = 0; i < 4; i++) {
                f[i].action[0] = (sequence >> 16) % 5;
                int expected;
#if FORAGE_GAZE_MODE == 1
                f[i].action[1] = (step + (i < 2)) % 5;
                int x = f[i].env.gaze_idx % 20;
                int y = f[i].env.gaze_idx / 20;
                if (f[i].action[1] == 1 && y > 0) y--;
                if (f[i].action[1] == 2 && y < 14) y++;
                if (f[i].action[1] == 3 && x > 0) x--;
                if (f[i].action[1] == 4 && x < 19) x++;
                expected = y * 20 + x;
#elif FORAGE_GAZE_MODE == 2
                expected = i >= 2 ? step % 300 : 299 - step % 300;
                f[i].action[1] = expected % 20;
                f[i].action[2] = expected / 20;
#else
                f[i].action[1] = i >= 2 ? step % 300 : 299 - step % 300;
                expected = f[i].action[1];
#endif
                puf_step(&f[i].env);
                assert(f[i].env.rng == f[0].env.rng);
                assert(f[i].env.player_idx == f[0].env.player_idx && f[i].env.tick == f[0].env.tick);
                assert(!memcmp(f[i].env.grid, f[0].env.grid, FORAGE_AREA));
                assert(!memcmp(&f[i].env.log, &f[0].env.log, sizeof(Log)));
                assert(f[i].reward == f[0].reward && f[i].terminal == f[0].terminal);
                if (f[i].terminal) {
                    assert(f[i].env.gaze_idx == f[i].env.player_idx);
                } else {
                    assert(f[i].env.gaze_idx == expected);
                }
            }
            if (!(step % 20)) {
                views(&f[2], &f[3], full);
            }
        }
        // Every absolute gaze center, including all edges/corners and overlaps.
        for (int center = 0; center < 300; center++) {
            f[2].env.gaze_idx = f[3].env.gaze_idx = center;
            compute_observations(&f[2].env);
            compute_observations(&f[3].env);
            views(&f[2], &f[3], full);
        }
#if FORAGE_GAZE_MODE == 1
        // Every starting cell/direction, including clipping at each boundary.
        for (int center = 0; center < 300; center++) {
            for (int action = 0; action < 5; action++) {
                int x = center % 20, y = center / 20;
                if (action == 1 && y > 0) y--;
                if (action == 2 && y < 14) y++;
                if (action == 3 && x > 0) x--;
                if (action == 4 && x < 19) x++;
                f[2].env.gaze_idx = center;
                f[2].action[1] = action;
                unsigned int rng = f[2].env.rng;
                forage_attention_action(&f[2].env);
                assert(f[2].env.gaze_idx == y * 20 + x && f[2].env.rng == rng);
            }
        }
#elif FORAGE_GAZE_MODE == 2
        for (int y = 0; y < 15; y++) {
            for (int x = 0; x < 20; x++) {
                f[2].action[1] = x;
                f[2].action[2] = y;
                forage_attention_action(&f[2].env);
                assert(f[2].env.gaze_idx == y * 20 + x);
            }
        }
#endif
        int indices[17], visible[FORAGE_AREA] = {0};
        cells(&f[3].env, indices);
        for (int i = 0; i < 17; i++) {
            if (indices[i] >= 0) {
                visible[indices[i]] = 1;
            }
        }
        for (int mode = 2; mode < 4; mode++) {
            float* saved = malloc(f[mode].size * sizeof(float));
            memcpy(saved, f[mode].storage + 1, f[mode].size * sizeof(float));
            unsigned char grid[FORAGE_AREA];
            memcpy(grid, f[mode].env.grid, FORAGE_AREA);
            unsigned int rng = f[mode].env.rng;
            for (int cell = 0; cell < FORAGE_AREA; cell++) {
                if (!visible[cell]) {
                    f[mode].env.grid[cell] = (grid[cell] + 1) % FORAGE_TILE_TYPES;
                }
            }
            compute_observations(&f[mode].env);
            assert(f[mode].env.rng == rng);
            assert(!memcmp(saved, f[mode].storage + 1, f[mode].size * sizeof(float)));
            memcpy(f[mode].env.grid, grid, FORAGE_AREA);
            free(saved);
        }
        // Looking while waiting must refresh the view without moving the player.
        int player = f[3].env.player_idx;
        f[3].action[0] = 0;
        f[3].action[1] = 0;
        f[3].env.gaze_idx = 0;
#if FORAGE_GAZE_MODE == 2
        f[3].action[2] = 0;
#endif
        f[3].env.tick = 0;
        puf_step(&f[3].env);
        assert(!f[3].terminal && f[3].env.player_idx == player && f[3].env.gaze_idx == 0);
        for (int i = 0; i < 4; i++) {
            puf_close(&f[i].env);
            free(f[i].storage);
        }
    }
    free(full);
    puts("Attention PASS: 24000 world steps, all 1500 skin/gaze pairs, exact full-frame crops,");
    puts("state/pixel visibility/geometry, no hidden-cell leakage, gaze/RNG independence and wait/reset timing.");
}
