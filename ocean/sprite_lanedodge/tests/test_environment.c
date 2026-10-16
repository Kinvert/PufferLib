// Game/observation tests only. No CPU CNN, GPU or policy.
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static float* tracked_obs;
static size_t pixel_bytes;
static void* track_memcpy(void* dst, const void* src, size_t size) {
    uintptr_t address = (uintptr_t)dst;
    uintptr_t first = (uintptr_t)tracked_obs;
    if (tracked_obs && address >= first && address < first + 640 * 480 * 3 * sizeof(float)) {
        pixel_bytes += size;
    }
    return memcpy(dst, src, size);
}

#define memcpy track_memcpy
#include "../sprite_lanedodge.h"
#undef memcpy
#define OBS_SIZE LD_PIXEL_OBS_SIZE

typedef struct { Env env; float* storage; float actions[2]; float reward, terminal; } Fixture;

static void setup(Fixture* f, int rep, unsigned int seed, int mode, int max_steps,
    int hazard_rate, int coin_rate) {
    memset(f, 0, sizeof(*f));
    int size = mode == 1 ? LD_PIXEL_OBS_SIZE
        : mode == 2 ? LD_ROI_OBS_SIZE : STATE_OBS_SIZE;
    f->storage = (float*)calloc(size + 2, sizeof(float)); assert(f->storage);
    f->storage[0] = 12345; f->storage[size + 1] = -12345;
    f->env.agents[0].observations = f->storage + 1;
    f->env.agents[0].actions = f->actions;
    f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal; f->env.rng = seed;
    Dict kwargs = {0};
    dict_set(&kwargs, "observation_mode", mode);
    dict_set(&kwargs, "representation", rep);
    dict_set(&kwargs, "max_steps", max_steps);
    dict_set(&kwargs, "hazard_rate", hazard_rate);
    dict_set(&kwargs, "coin_rate", coin_rate);
    dict_set(&kwargs, "step_cost", -.001);
    dict_set(&kwargs, "reward_coin", 1);
    dict_set(&kwargs, "reward_death", -.5);
    dict_set(&kwargs, "seed", 0);
    puf_init(&f->env, &kwargs); dict_clear(&kwargs);
    assert(f->env.rng == seed);
    if (mode == 1) {
        tracked_obs = f->env.agents[0].observations;
        pixel_bytes = 0;
    }
    puf_reset(&f->env);
    if (mode == 1) {
        assert(pixel_bytes == LD_PIXEL_OBS_SIZE * sizeof(float));
        tracked_obs = NULL;
    }
}

// Full independent pixel redraw: force every cached tile and walk all cells.
static void redraw_all(Env* env) {
    memset(env->pixel_grid, 255, sizeof(env->pixel_grid));
    for (int cell = 0; cell < LD_AREA; cell++) {
        int kind = cell == env->player_idx ? LD_PLAYER : env->grid[cell];
        if (env->pixel_grid[cell] != kind) {
            write_pixel_cell(env, cell);
        }
    }
}

static void observation(Fixture* f) {
    assert(f->storage[0] == 12345 && f->storage[OBS_SIZE + 1] == -12345);
    float* saved = (float*)malloc(OBS_SIZE * sizeof(float)); assert(saved);
    memcpy(saved, f->storage + 1, OBS_SIZE * sizeof(float));
    Env before = f->env; redraw_all(&f->env);
    assert(!memcmp(&before, &f->env, sizeof(Env)));
    // The dirty-cell fast path must reproduce a full independent redraw.
    assert(!memcmp(saved, f->storage + 1, OBS_SIZE * sizeof(float)));
    for (int i = 0; i < OBS_SIZE; i++) {
        assert(isfinite(saved[i]) && saved[i] >= 0 && saved[i] <= 1);
    }
    free(saved);
}

static void same_world(Fixture* a, Fixture* b) {
    assert(a->env.rng == b->env.rng && a->env.object_rng == b->env.object_rng
        && a->env.lane == b->env.lane && a->env.player_idx == b->env.player_idx
        && a->env.tick == b->env.tick && a->env.coins == b->env.coins
        && a->env.coins_spawned == b->env.coins_spawned
        && a->env.gaze_idx == b->env.gaze_idx);
    assert(!memcmp(a->env.grid, b->env.grid, LD_AREA));
    assert(a->reward == b->reward && a->terminal == b->terminal);
    assert(!memcmp(&a->env.log, &b->env.log, sizeof(Log)));
}

// The world is action-independent except the player's own row.
static void world_same_except_player_row(Env* a, Env* b) {
    assert(a->rng == b->rng);
    for (int cell = 0; cell < LD_AREA; cell++) {
        int y = cell / LD_WIDTH, x = cell % LD_WIDTH;
        // Rows 12-13 of the lane columns see player-dependent collisions.
        if (y >= LD_PLAYER_ROW - 1 && x >= 5 && x <= 14) continue;
        assert(a->grid[cell] == b->grid[cell]);
    }
}

int main(void) {
    // Art: fixed row width, palette glyphs, hazard/player shared silhouette.
    const char* palette = ".awRDYOBbsk";
    for (int kind = 0; kind < LD_TILE_TYPES; kind++) {
        for (int y = 0; y < 16; y++) {
            assert(strlen(LD_ART[kind][y]) == 16);
            for (int x = 0; x < 16; x++) {
                assert(strchr(palette, LD_ART[kind][y][x]));
            }
        }
    }
    for (int y = 0; y < 16; y++) {
        char row[17];
        for (int x = 0; x < 16; x++) {
            char glyph = LD_ART[LD_HAZARD][y][x];
            row[x] = glyph == 'R' ? 'b' : (glyph == 'D' ? 'B' : glyph);
        }
        row[16] = 0;
        assert(!strcmp(row, LD_ART[LD_PLAYER][y]));
    }
    assert(memcmp(LD_ART[LD_HAZARD][0], LD_ART[LD_PLAYER][0], 16));
    // Lane-edge art: dashed white stripe on the left edge, asphalt base.
    assert(!strcmp(LD_ART[LD_DIVIDER][0], "waaaaaaaaaaaaaaa"));
    assert(!strcmp(LD_ART[LD_DIVIDER][4], "aaaaaaaaaaaaaaaa"));
    assert(!strcmp(LD_ART[LD_DIVIDER][8], "waaaaaaaaaaaaaaa"));
    assert(!strcmp(LD_ART[LD_DIVIDER][12], "aaaaaaaaaaaaaaaa"));

    // Representations: rep 1 swaps RGB, rep 2 mirrors; caches stay put.
    for (int rep = 0; rep < LD_REPRESENTATIONS; rep++) ld_make_tiles(rep);
    // The stripe is bright and unshaded; the road base stays non-flat.
    assert(ld_tiles[0][LD_DIVIDER][0] == 235.f / 255.f);
    assert(ld_tiles[0][LD_DIVIDER][1] == 235.f / 255.f);
    assert(ld_tiles[0][LD_DIVIDER][2] == 240.f / 255.f);
    assert(ld_tiles[0][LD_DIVIDER][792] == ld_tiles[0][LD_ROAD][792]);
    for (int kind = 0; kind < LD_TILE_TYPES; kind++) {
        for (int i = 0; i < LD_CELL * LD_CELL * 3; i += 3) {
            assert(ld_tiles[1][kind][i] == ld_tiles[0][kind][i + 2]);
            assert(ld_tiles[1][kind][i + 1] == ld_tiles[0][kind][i + 1]);
            assert(ld_tiles[1][kind][i + 2] == ld_tiles[0][kind][i]);
        }
        for (int y = 0; y < LD_CELL; y++) {
            for (int x = 0; x < LD_CELL; x++) {
                int x2 = 2 * (15 - x / 2); // Tile-0 column with the same art column.
                assert(ld_tiles[2][kind][y * 96 + x * 3]
                    == ld_tiles[0][kind][y * 96 + x2 * 3]);
            }
        }
    }
    // Grass at art pixel (0,0) shades -4; theme 1 swaps RGB first: (70,134,92).
    assert(ld_tiles[1][LD_GRASS][0] == 70.f / 255.f);
    assert(ld_tiles[1][LD_GRASS][1] == 134.f / 255.f);
    assert(ld_tiles[1][LD_GRASS][2] == 92.f / 255.f);
    // Transparent sprite art composites over the shaded road, not the grass default.
    for (int kind = LD_HAZARD; kind < LD_TILE_TYPES; kind++) {
        for (int y = 0; y < LD_CELL; y++) {
            for (int x = 0; x < LD_CELL; x++) {
                int i = (y * LD_CELL + x) * 3;
                if (LD_ART[kind][y / 2][x / 2] != '.') {
                    continue;
                }
                for (int rep = 0; rep < LD_REPRESENTATIONS; rep++) {
                    assert(ld_tiles[rep][kind][i] == ld_tiles[rep][LD_ROAD][i]
                        && ld_tiles[rep][kind][i + 1] == ld_tiles[rep][LD_ROAD][i + 1]
                        && ld_tiles[rep][kind][i + 2] == ld_tiles[rep][LD_ROAD][i + 2]);
                }
            }
        }
    }
    // Ground shading is not a flat fill.
    assert(ld_tiles[0][LD_ROAD][0] != ld_tiles[0][LD_ROAD][6]);
    float saved = ld_tiles[0][LD_GRASS][0];
    ld_tiles[0][LD_GRASS][0] = -1;
    ld_make_tiles(0);
    assert(ld_tiles[0][LD_GRASS][0] == -1);
    ld_tiles[0][LD_GRASS][0] = saved;

    // Reset: static layout, player in lane 1, empty spawn row, one world draw.
    Fixture s; setup(&s, 0, 7, 0, 400, 45, 25);
    for (int cell = 0; cell < LD_AREA; cell++) {
        assert(s.env.grid[cell]
            == (unsigned char)ld_cell_bg(cell % LD_WIDTH));
    }
    assert(s.env.player_idx == LD_PLAYER_ROW * LD_WIDTH + LD_LANE_X(1));
    assert(s.env.lane == 1 && s.env.tick == 0 && s.env.coins == 0 && s.env.coins_spawned == 0);
    assert(s.env.rng == (7u * 1664525u + 1013904223u));

    // Observation size per mode.
    Dict kwargs = {0};
    dict_set(&kwargs, "observation_mode", 0);
    assert(puf_observation_size(&kwargs) == STATE_OBS_SIZE);
    dict_set(&kwargs, "observation_mode", 1);
    assert(puf_observation_size(&kwargs) == LD_PIXEL_OBS_SIZE);
    dict_set(&kwargs, "observation_mode", 2);
    assert(puf_observation_size(&kwargs) == LD_ROI_OBS_SIZE);
    dict_clear(&kwargs);

    // State semantics: lane value, nearest-wins distances, missing entities.
    s.env.grid[0 * LD_WIDTH + LD_LANE_X(0)] = LD_HAZARD;
    s.env.grid[3 * LD_WIDTH + LD_LANE_X(0)] = LD_HAZARD;
    s.env.grid[12 * LD_WIDTH + LD_LANE_X(1)] = LD_HAZARD;
    s.env.grid[5 * LD_WIDTH + LD_LANE_X(2)] = LD_COIN;
    compute_observations(&s.env);
    float* obs = s.env.agents[0].observations;
    float expected_lane[10] = {0, 1 / 9.f, 2 / 9.f, 3 / 9.f, 4 / 9.f,
        5 / 9.f, 6 / 9.f, 7 / 9.f, 8 / 9.f, 1};
    assert(obs[0] == expected_lane[s.env.lane]);
    assert(obs[1] == 11.f / 14.f); // Row 3 beats row 0: nearest hazard wins.
    assert(obs[2] == 2.f / 14.f);
    for (int i = 3; i < 11; i++) assert(obs[i] == 0.f);
    assert(obs[13] == 9.f / 14.f);
    for (int i = 11; i < STATE_OBS_SIZE; i++) {
        if (i != 13) assert(obs[i] == 0.f);
    }
    s.env.grid[0 * LD_WIDTH + LD_LANE_X(0)] = LD_ROAD;
    s.env.grid[3 * LD_WIDTH + LD_LANE_X(0)] = LD_ROAD;
    s.env.grid[12 * LD_WIDTH + LD_LANE_X(1)] = LD_DIVIDER;
    s.env.grid[5 * LD_WIDTH + LD_LANE_X(2)] = LD_DIVIDER;
    compute_observations(&s.env);
    for (int i = 1; i < STATE_OBS_SIZE; i++) assert(obs[i] == 0.f);

    // Lane clamp: boundary actions hold the lane.
    Fixture cl; setup(&cl, 0, 7, 0, 50, 0, 0);
    cl.env.lane = 0; cl.actions[0] = 1;
    puf_step(&cl.env);
    assert(cl.env.lane == 0 && cl.env.player_idx == LD_PLAYER_ROW * LD_WIDTH + LD_LANE_X(0));
    cl.env.lane = 9; cl.actions[0] = 2;
    puf_step(&cl.env);
    assert(cl.env.lane == 9 && cl.env.player_idx == LD_PLAYER_ROW * LD_WIDTH + LD_LANE_X(9));

    // Hazard lands on the player row: death, reward, perf, auto reset.
    Fixture d; setup(&d, 0, 7, 0, 50, 0, 0);
    d.env.grid[12 * LD_WIDTH + LD_LANE_X(1)] = LD_HAZARD;
    d.actions[0] = 0;
    puf_step(&d.env);
    assert(d.terminal == 1);
    assert(d.env.log.n == 1 && d.env.log.deaths == 1 && d.env.log.successes == 0);
    assert(fabsf(d.reward - (-.001f + -.5f)) < 1e-6f);
    assert(fabsf(d.env.log.perf - 0.5f * (1.f / 50.f)) < 1e-6f);
    assert(d.env.tick == 0 && d.env.coins == 0);

    // Coin lands on the player row: collected, cell cleared.
    Fixture c; setup(&c, 0, 7, 0, 50, 0, 0);
    c.env.grid[12 * LD_WIDTH + LD_LANE_X(1)] = LD_COIN;
    c.actions[0] = 0;
    puf_step(&c.env);
    assert(!c.terminal && c.env.coins == 1);
    assert(fabsf(c.reward - (-.001f + 1.f)) < 1e-6f);
    assert(c.env.grid[13 * LD_WIDTH + LD_LANE_X(1)] == LD_DIVIDER);
    // The arriving coin is consumed on both rows; a second step collects nothing.
    c.actions[0] = 0;
    puf_step(&c.env);
    assert(!c.terminal && c.env.coins == 1);
    assert(c.env.grid[12 * LD_WIDTH + LD_LANE_X(1)] == LD_DIVIDER);
    assert(c.env.grid[13 * LD_WIDTH + LD_LANE_X(1)] == LD_DIVIDER);
    assert(fabsf(c.reward - (-.001f)) < 1e-6f);

    // Same forced case in pixel mode: the dirty path matches a full redraw.
    Fixture cp; setup(&cp, 0, 7, 1, 50, 0, 0);
    cp.env.grid[12 * LD_WIDTH + LD_LANE_X(1)] = LD_COIN;
    cp.actions[0] = 0;
    puf_step(&cp.env);
    assert(!cp.terminal && cp.env.coins == 1);
    assert(cp.env.grid[12 * LD_WIDTH + LD_LANE_X(1)] == LD_DIVIDER);
    observation(&cp);
    cp.actions[0] = 0;
    puf_step(&cp.env);
    assert(cp.env.coins == 1);
    assert(fabsf(cp.reward - (-.001f)) < 1e-6f);

    // Player moves into a row-13 hazard: death.
    Fixture d2; setup(&d2, 0, 7, 0, 50, 0, 0);
    d2.env.grid[LD_PLAYER_ROW * LD_WIDTH + LD_LANE_X(0)] = LD_HAZARD;
    d2.actions[0] = 1; // Lane 1 -> 0.
    puf_step(&d2.env);
    assert(d2.terminal == 1 && d2.env.log.deaths == 1);

    // Player moves onto a row-13 coin: collected.
    Fixture c2; setup(&c2, 0, 7, 0, 50, 0, 0);
    c2.env.grid[LD_PLAYER_ROW * LD_WIDTH + LD_LANE_X(1)] = LD_COIN;
    c2.actions[0] = 0; // Stay in lane 1.
    puf_step(&c2.env);
    assert(!c2.terminal && c2.env.coins == 1);
    assert(c2.env.grid[LD_PLAYER_ROW * LD_WIDTH + LD_LANE_X(1)] == LD_DIVIDER);

    // Timeout: full episode, success, survival-only perf.
    Fixture t; setup(&t, 0, 7, 0, 50, 0, 0);
    for (int i = 0; i < 50; i++) {
        t.actions[0] = 0;
        puf_step(&t.env);
        assert(i < 49 || t.terminal);
    }
    assert(t.terminal == 1 && t.env.log.n == 1);
    assert(t.env.log.successes == 1 && t.env.log.deaths == 0);
    assert(fabsf(t.env.log.perf - 0.5f) < 1e-6f);
    assert(fabsf(t.env.log.episode_length - 50.f) < 1e-6f);

    // World schedule is action-independent: two action histories, one world.
    // Hazards off so neither fixed-lane player dies and both time out in lockstep.
    Fixture a, b;
    setup(&a, 0, 7, 0, 400, 0, 25);
    setup(&b, 0, 7, 0, 400, 0, 25);
    while (a.env.log.n == 0 && b.env.log.n == 0) {
        world_same_except_player_row(&a.env, &b.env);
        assert(a.env.object_rng == b.env.object_rng);
        assert(a.env.coins_spawned == b.env.coins_spawned);
        a.actions[0] = 1; b.actions[0] = 0; (void)rand();
        puf_step(&a.env); puf_step(&b.env);
    }
    // The episode-6 object stream is a function of (seed, episode), not history.
    unsigned int a_obj = 0, b_obj = 0;
    for (int step = 0; step < 20000 && (a.env.log.n < 5 || b.env.log.n < 5); step++) {
        int a_n = a.env.log.n, b_n = b.env.log.n;
        a.actions[0] = 1; b.actions[0] = 0; (void)rand();
        puf_step(&a.env); puf_step(&b.env);
        if (a_n < 5 && a.env.log.n == 5) a_obj = a.env.object_rng;
        if (b_n < 5 && b.env.log.n == 5) b_obj = b.env.object_rng;
    }
    assert(a.env.log.n == 5 && b.env.log.n == 5);
    // Independent world chain: initial reset plus five episode resets.
    unsigned int w = 7;
    for (int k = 0; k < 6; k++) w = w * 1664525u + 1013904223u;
    assert(a.env.rng == w && b.env.rng == w);
    unsigned int fresh = w ^ 0x9e3779b9u;
    fresh ^= fresh >> 16;
    fresh *= 0x85ebca6bu;
    assert(a_obj == fresh && b_obj == fresh);

    // State, pixel, and ROI modes run the identical game under one action stream.
    Fixture st, px, ro;
    setup(&st, 0, 7, 0, 400, 45, 25);
    setup(&px, 0, 7, 1, 400, 45, 25);
    setup(&ro, 0, 7, 2, 400, 45, 25);
    unsigned int actions = 1337;
    for (int step = 0; step < 1000; step++) {
        actions = actions * 1664525u + 1013904223u;
        int act = (actions >> 16) % 3;
        int gaze = (actions >> 8) % LD_AREA;
        st.actions[0] = act; st.actions[1] = gaze;
        px.actions[0] = act; px.actions[1] = gaze;
        ro.actions[0] = act; ro.actions[1] = gaze;
        (void)rand();
        puf_step(&st.env);
        puf_step(&px.env);
        puf_step(&ro.env);
        same_world(&st, &px);
        same_world(&st, &ro);
        float* sobs = st.env.agents[0].observations;
        float lane_value[10] = {0, 1 / 9.f, 2 / 9.f, 3 / 9.f, 4 / 9.f,
            5 / 9.f, 6 / 9.f, 7 / 9.f, 8 / 9.f, 1};
        assert(sobs[0] == lane_value[st.env.lane]);
        for (int i = 0; i < STATE_OBS_SIZE; i++) {
            assert(isfinite(sobs[i]) && sobs[i] >= 0 && sobs[i] <= 1);
        }
        for (int i = 0; i < LD_ROI_OBS_SIZE; i++) {
            float value = ro.env.agents[0].observations[i];
            assert(isfinite(value) && value >= 0 && value <= 1);
        }
        if (step % 100 == 0) observation(&px);
    }
    for (int i = 0; i < LD_PIXEL_OBS_SIZE; i++) {
        float value = px.env.agents[0].observations[i];
        assert(isfinite(value) && value >= 0 && value <= 1);
    }

    // Mode 2: the ROI window follows the gaze; out-of-range cells are grass.
    Fixture ro2; setup(&ro2, 0, 7, 2, 50, 0, 0);
    float* o = ro2.env.agents[0].observations;
    // Reset: the gaze sits on the player cell; the center slot is the player.
    assert(ro2.env.gaze_idx == ro2.env.player_idx);
    assert(o[9 * LD_CELL * LD_CELL * 3] == 1.f / 9.f);
    assert(o[9 * LD_CELL * LD_CELL * 3 + 1] == 6.f / 19.f);
    assert(o[9 * LD_CELL * LD_CELL * 3 + 2] == 13.f / 14.f);
    assert(!memcmp(o + 4 * LD_CELL * LD_CELL * 3, ld_tiles[0][LD_PLAYER],
        LD_CELL * LD_CELL * 3 * sizeof(float)));
    // A hazard above the player fills the top-center slot; lanes show edges.
    ro2.env.grid[12 * LD_WIDTH + LD_LANE_X(1)] = LD_HAZARD;
    write_roi(&ro2.env);
    assert(!memcmp(o + LD_CELL * LD_CELL * 3, ld_tiles[0][LD_HAZARD],
        LD_CELL * LD_CELL * 3 * sizeof(float)));
    assert(!memcmp(o, ld_tiles[0][LD_ROAD], LD_CELL * LD_CELL * 3 * sizeof(float)));
    assert(!memcmp(o + 2 * LD_CELL * LD_CELL * 3, ld_tiles[0][LD_DIVIDER],
        LD_CELL * LD_CELL * 3 * sizeof(float)));
    ro2.env.grid[12 * LD_WIDTH + LD_LANE_X(1)] = LD_DIVIDER;
    // Gaze action: valid IDs retarget the window, invalid ones are ignored.
    ro2.actions[0] = 0;
    ro2.actions[1] = LD_AREA + 5;
    puf_step(&ro2.env);
    assert(ro2.env.gaze_idx == ro2.env.player_idx);
    ro2.actions[1] = 0;
    puf_step(&ro2.env);
    assert(ro2.env.gaze_idx == 0);
    assert(o[9 * LD_CELL * LD_CELL * 3 + 1] == 0.f);
    assert(o[9 * LD_CELL * LD_CELL * 3 + 2] == 0.f);
    assert(!memcmp(o, ld_tiles[0][LD_GRASS], LD_CELL * LD_CELL * 3 * sizeof(float)));

    // Pixel reset cache: first reset writes the full image; a repeated world writes nothing.
    Fixture r; setup(&r, 0, 7, 1, 400, 45, 25);
    tracked_obs = r.env.agents[0].observations;
    pixel_bytes = 0;
    r.env.rng = 7;
    puf_reset(&r.env);
    assert(pixel_bytes == 0);
    tracked_obs = NULL;

    // Representations change the skin, not the world.
    Fixture r1; setup(&r1, 1, 7, 1, 400, 45, 25);
    Fixture r2; setup(&r2, 2, 7, 1, 400, 45, 25);
    same_world(&r, &r1);
    same_world(&r, &r2);
    assert(memcmp(r.storage + 1, r1.storage + 1, OBS_SIZE * sizeof(float)));
    assert(memcmp(r.storage + 1, r2.storage + 1, OBS_SIZE * sizeof(float)));

    free(s.env.agents[0].observations - 1);
    free(cl.env.agents[0].observations - 1);
    free(d.env.agents[0].observations - 1);
    free(c.env.agents[0].observations - 1);
    free(cp.env.agents[0].observations - 1);
    free(d2.env.agents[0].observations - 1);
    free(c2.env.agents[0].observations - 1);
    free(t.env.agents[0].observations - 1);
    free(a.env.agents[0].observations - 1);
    free(b.env.agents[0].observations - 1);
    free(st.env.agents[0].observations - 1);
    free(px.env.agents[0].observations - 1);
    free(ro.env.agents[0].observations - 1);
    free(ro2.env.agents[0].observations - 1);
    free(r.env.agents[0].observations - 1);
    free(r1.env.agents[0].observations - 1);
    free(r2.env.agents[0].observations - 1);
    return 0;
}
