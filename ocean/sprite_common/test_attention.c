// Scalar game/raster checks only. No CPU neural computation or GPU context.
#include <assert.h>
#include <math.h>
#define SPRITE_ATTENTION
#include GAME_HEADER

typedef struct {
    Env env;
    float* buffer;
    float actions[NUM_ATNS];
    float reward, terminal;
    int size;
} Fixture;

void setup(Fixture* f, int pixels, int partial, int skin) {
    memset(f, 0, sizeof(*f));
    Ini ini = {0};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, GAME_CONFIG);
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    dict_set(kwargs, "observation_mode", pixels);
    dict_set(kwargs, "partial_observation", partial);
    dict_set(kwargs, "representation", skin);
    f->size = puf_observation_size(kwargs);
    assert(f->size == (partial ? (pixels ? SPRITE_VIEW_OBS : SPRITE_VIEW_STATE + 4)
        : pixels ? PIXEL_OBS_SIZE : STATE_OBS_SIZE));
    f->buffer = calloc(f->size + 2, sizeof(float));
    assert(f->buffer);
    f->buffer[0] = 12345;
    f->buffer[f->size + 1] = -12345;
    f->env.rng = 7;
    f->env.agents[0] = (Agent){.observations = f->buffer + 1, .actions = f->actions,
        .rewards = &f->reward, .terminals = &f->terminal};
    puf_init(&f->env, kwargs);
    puf_reset(&f->env);
    puf_ini_free(&ini);
}

void same_world(Fixture* a, Fixture* b) {
    assert(a->env.rng == b->env.rng && a->env.floor_seed == b->env.floor_seed);
    assert(a->env.player == b->env.player && a->env.tick == b->env.tick);
    assert(!memcmp(a->env.grid, b->env.grid, sizeof(a->env.grid)));
    assert(a->reward == b->reward && a->terminal == b->terminal);
    assert(!memcmp(&a->env.log, &b->env.log, sizeof(Log)));
}

void observations(Fixture* state, Fixture* pixels, float* full) {
    float* s = state->buffer + 1;
    float* p = pixels->buffer + 1;
    sprite_observe(full, pixels->env.grid, pixels->env.player,
        pixels->env.representation, pixels->env.floor_seed, NULL);
    for (int slot = 0; slot < SPRITE_VIEW_TILES; slot++) {
        int cell = sprite_attention_cell(&state->env, slot);
        int kind = cell < 0 ? SPRITE_OUTSIDE : cell == state->env.player
            ? SPRITE_PLAYER : state->env.grid[cell];
        for (int k = 0; k < SPRITE_TILE_TYPES; k++) {
            assert(s[slot * SPRITE_TILE_TYPES + k] == (k == kind));
        }
        for (int y = 0; y < SPRITE_CELL; y++) {
            const float* expected = cell < 0 ? sprite_tiles[pixels->env.representation][kind][0]
                + y * SPRITE_CELL * 3 : full + ((cell / GRID_WIDTH * SPRITE_CELL + y)
                    * OBS_WIDTH + cell % GRID_WIDTH * SPRITE_CELL) * 3;
            assert(!memcmp(p + slot * SPRITE_TILE_PIXELS + y * SPRITE_CELL * 3,
                expected, SPRITE_CELL * 3 * sizeof(float)));
        }
    }
    assert(!memcmp(s + SPRITE_VIEW_STATE, p + SPRITE_VIEW_PIXELS, 4 * sizeof(float)));
    for (int i = 0; i < 4; i++) {
        assert(isfinite(s[SPRITE_VIEW_STATE + i]) && fabsf(s[SPRITE_VIEW_STATE + i]) <= 1);
    }
    assert(state->buffer[0] == 12345 && state->buffer[state->size + 1] == -12345);
    assert(pixels->buffer[0] == 12345 && pixels->buffer[pixels->size + 1] == -12345);
}

int main(void) {
    const int sizes[] = ACT_SIZES;
    assert(NUM_ATNS == 3 && sizes[1] == GRID_WIDTH && sizes[2] == GRID_HEIGHT);
    for (int skin = 0; skin < 3; skin++) {
        Fixture state, pixels, control;
        setup(&state, 0, 1, skin);
        setup(&pixels, 1, 1, skin);
        setup(&control, 0, 0, skin);
        float* full = malloc(PIXEL_OBS_SIZE * sizeof(float));
        assert(full);
        unsigned int sequence = 1337;
        for (int step = 0; step < 1024; step++) {
            same_world(&state, &pixels);
            same_world(&state, &control);
            observations(&state, &pixels, full);
            sequence = sequence * 1664525u + 1013904223u;
            int movement = (sequence >> 16) % sizes[0];
            int gx = (sequence >> 8) % GRID_WIDTH;
            int gy = (sequence >> 24) % GRID_HEIGHT;
            Fixture* fixtures[] = {&state, &pixels, &control};
            for (int i = 0; i < 3; i++) {
                fixtures[i]->actions[0] = movement;
                fixtures[i]->actions[1] = gx;
                fixtures[i]->actions[2] = gy;
                puf_step(&fixtures[i]->env);
            }
            if (!state.terminal) {
                assert(state.env.gaze_idx == gy * GRID_WIDTH + gx);
                assert(pixels.env.gaze_idx == state.env.gaze_idx);
            } else {
                assert(state.env.gaze_idx == state.env.player);
                assert(pixels.env.gaze_idx == pixels.env.player);
            }
        }
        // Changing a hidden cell must not alter either partial observation.
        int hidden = -1;
        for (int cell = 0; cell < GRID_WIDTH * GRID_HEIGHT && hidden < 0; cell++) {
            int seen = 0;
            for (int slot = 0; slot < SPRITE_VIEW_TILES; slot++) {
                seen |= sprite_attention_cell(&state.env, slot) == cell;
            }
            if (!seen) {
                hidden = cell;
            }
        }
        assert(hidden >= 0);
        float* saved_state = malloc(state.size * sizeof(float));
        float* saved_pixels = malloc(pixels.size * sizeof(float));
        assert(saved_state && saved_pixels);
        memcpy(saved_state, state.buffer + 1, state.size * sizeof(float));
        memcpy(saved_pixels, pixels.buffer + 1, pixels.size * sizeof(float));
        state.env.grid[hidden] = (state.env.grid[hidden] + 1) % SPRITE_TILE_TYPES;
        pixels.env.grid[hidden] = state.env.grid[hidden];
        compute_observations(&state.env);
        compute_observations(&pixels.env);
        assert(!memcmp(saved_state, state.buffer + 1, state.size * sizeof(float)));
        assert(!memcmp(saved_pixels, pixels.buffer + 1, pixels.size * sizeof(float)));
        free(saved_state);
        free(saved_pixels);
        free(full);
        free(state.buffer);
        free(pixels.buffer);
        free(control.buffer);
    }
    puts("PASS: 3072 paired game steps, full-frame crop parity, all skins, one-hot cells, gaze/reset timing, hidden-cell isolation");
}
