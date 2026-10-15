// Game/observation tests only. No CPU CNN, GPU or policy.
#include <assert.h>
#include <math.h>
#include "../sprite_forage.h"
#define OBS_SIZE FORAGE_PIXEL_OBS_SIZE

static void forage_sprite(int kind, unsigned char pixels[256]) {
    for (int y = 0; y < 16; y++) {
        assert(strlen(FORAGE_ART[kind][y]) == 16);
        memcpy(pixels + y * 16, FORAGE_ART[kind][y], 16);
    }
}

typedef struct { Env env; float* storage; float action, reward, terminal; } Fixture;
static void setup(Fixture* f, int rep, unsigned int seed) {
    memset(f, 0, sizeof(*f));
    f->storage = (float*)calloc(OBS_SIZE + 2, sizeof(float)); assert(f->storage);
    f->storage[0] = 12345; f->storage[OBS_SIZE + 1] = -12345;
    f->env.agents[0].observations = f->storage + 1;
    f->env.agents[0].actions = &f->action; f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal; f->env.rng = seed;
    Dict kwargs = {0};
    dict_set(&kwargs, "copies_per_type", 3); dict_set(&kwargs, "trees", 8);
    dict_set(&kwargs, "max_steps", 512); dict_set(&kwargs, "step_cost", -.001);
    dict_set(&kwargs, "reward_good", 1);
    dict_set(&kwargs, "reward_other", .5);
    dict_set(&kwargs, "reward_grape", 1);
    dict_set(&kwargs, "state_layout", 0);
    dict_set(&kwargs, "reward_bad", -.5);
    dict_set(&kwargs, "reward_death", -1);
    dict_set(&kwargs, "representation", rep);
    dict_set(&kwargs, "observation_mode", 1);
    dict_set(&kwargs, "seed", 0);
    puf_init(&f->env, &kwargs); dict_clear(&kwargs);
    assert(f->env.rng == seed); puf_reset(&f->env);
}
static void close_fixture(Fixture* f) { puf_close(&f->env); free(f->storage); }
static void observation(Fixture* f) {
    assert(f->storage[0] == 12345 && f->storage[OBS_SIZE + 1] == -12345);
    float* saved = (float*)malloc(OBS_SIZE * sizeof(float)); assert(saved);
    memcpy(saved, f->storage + 1, OBS_SIZE * sizeof(float));
    Env before = f->env; compute_observations(&f->env);
    assert(!memcmp(&before, &f->env, sizeof(Env)));
    // Compare the dirty two-cell fast path against a full independent redraw.
    assert(!memcmp(saved, f->storage + 1, OBS_SIZE * sizeof(float)));
    for (int i = 0; i < OBS_SIZE; i++) assert(isfinite(saved[i]) && saved[i] >= 0 && saved[i] <= 1);
    free(saved);
}
static void same_world(Fixture* a, Fixture* b) {
    assert(a->env.rng == b->env.rng && a->env.player_idx == b->env.player_idx && a->env.tick == b->env.tick);
    assert(!memcmp(a->env.grid, b->env.grid, FORAGE_AREA));
    assert(a->reward == b->reward && a->terminal == b->terminal);
    assert(!memcmp(&a->env.log, &b->env.log, sizeof(Log)));
}
static void histogram(int kind, int counts[256]) {
    unsigned char image[256]; forage_sprite(kind, image); memset(counts, 0, 256 * sizeof(int));
    for (int i = 0; i < 256; i++) counts[image[i]]++;
}
static void controlled(Fixture* f, int kind) {
    Env* e = &f->env; memset(e->grid, 0, FORAGE_AREA);
    for (int i = 0; i < FORAGE_AREA; i++) if (i < FORAGE_WIDTH || i >= FORAGE_AREA - FORAGE_WIDTH
        || i % FORAGE_WIDTH == 0 || i % FORAGE_WIDTH == FORAGE_WIDTH - 1) e->grid[i] = FORAGE_TREE;
    e->player_idx = FORAGE_WIDTH * 7 + 9; e->grid[e->player_idx + 1] = kind;
    e->apples = e->negatives = e->tick = 0; e->episode_return = 0;
    memset(&e->log, 0, sizeof(Log));
    e->max_steps = 512; compute_observations(e); f->action = 4;
}
int main(void) {
    int first[256], second[256];
    histogram(FORAGE_APPLE, first); histogram(FORAGE_SPOTTED_APPLE, second);
    assert(!memcmp(first, second, sizeof(first)));
    histogram(FORAGE_MUSHROOM, first); histogram(FORAGE_TOADSTOOL, second);
    assert(!memcmp(first, second, sizeof(first)));
    unsigned char a[256], b[256];
    forage_sprite(FORAGE_APPLE, a); forage_sprite(FORAGE_SPOTTED_APPLE, b); assert(memcmp(a, b, 256));
    forage_sprite(FORAGE_MUSHROOM, a); forage_sprite(FORAGE_TOADSTOOL, b); assert(memcmp(a, b, 256));
    Fixture f[FORAGE_REPRESENTATIONS];
    const unsigned char empty_tiles[sizeof(forage_tiles[0])] = {0};
    for (int rep = 0; rep < FORAGE_REPRESENTATIONS; rep++) {
        setup(&f[rep], rep, 7);
        for (int unused = rep + 1; unused < FORAGE_REPRESENTATIONS; unused++) {
            assert(!memcmp(forage_tiles[unused], empty_tiles, sizeof(empty_tiles)));
        }
        // A repeated request must leave the already-built cache untouched.
        float saved = forage_tiles[rep][FORAGE_EMPTY][0];
        forage_tiles[rep][FORAGE_EMPTY][0] = -1;
        forage_make_tiles(rep);
        assert(forage_tiles[rep][FORAGE_EMPTY][0] == -1);
        forage_tiles[rep][FORAGE_EMPTY][0] = saved;
    }
    for (int rep = 0; rep < FORAGE_REPRESENTATIONS; rep++) {
        int counts[13] = {0}; for (int i = 0; i < FORAGE_AREA; i++) counts[f[rep].env.grid[i]]++;
        for (int kind = 1; kind <= 10; kind++) {
            assert(counts[kind] == (kind == FORAGE_GRAPE ? 1 : 3));
        }
        assert(forage_connected(&f[rep].env)); observation(&f[rep]);
        if (rep) { same_world(&f[0], &f[rep]); assert(memcmp(f[0].storage, f[rep].storage, (OBS_SIZE + 2) * sizeof(float))); }
    }
    // Opposite stepping order and unrelated libc RNG draws cannot affect worlds.
    unsigned int actions = 1337;
    for (int step = 0; step < 1000; step++) {
        actions = actions * 1664525u + 1013904223u;
        for (int rep = FORAGE_REPRESENTATIONS - 1; rep >= 0; rep--) { f[rep].action = (actions >> 16) % 5; (void)rand(); puf_step(&f[rep].env); }
        for (int rep = 1; rep < FORAGE_REPRESENTATIONS; rep++) same_world(&f[0], &f[rep]);
        if (step % 17 == 0) for (int rep = 0; rep < FORAGE_REPRESENTATIONS; rep++) observation(&f[rep]);
    }
    for (unsigned int seed = 0; seed < 32; seed++) {
        f[0].env.rng = seed; puf_reset(&f[0].env); assert(forage_connected(&f[0].env));
        int counts[13] = {0}; for (int i = 0; i < FORAGE_AREA; i++) counts[f[0].env.grid[i]]++;
        for (int kind = 1; kind <= 10; kind++) {
            assert(counts[kind] == (kind == FORAGE_GRAPE ? 1 : 3));
        }
        assert(!f[0].env.grid[f[0].env.player_idx]);
    }
    for (int kind = 1; kind <= 10; kind++) {
        controlled(&f[0], kind); puf_step(&f[0].env);
        const float reward[] = {0, 1, .5, 1, .5, .5, .5, .5, -.5, -.5, -1};
        assert(fabsf(f[0].reward - (reward[kind] - .001f)) < 1e-6f);
        assert((f[0].terminal != 0) == (kind == FORAGE_TOADSTOOL || kind == FORAGE_GRAPE));
        if (f[0].terminal) {
            assert(f[0].env.tick == 0);
            assert(f[0].env.log.deaths == (kind == FORAGE_TOADSTOOL));
            assert(f[0].env.log.successes == (kind == FORAGE_GRAPE));
            assert(f[0].env.log.perf == (kind == FORAGE_GRAPE ? .25f : 0));
        } else {
            assert(f[0].env.grid[f[0].env.player_idx] == FORAGE_EMPTY);
            observation(&f[0]);
        }
    }
    // Collecting the last apple does not terminate; the single grape does.
    controlled(&f[0], FORAGE_APPLE);
    int start = f[0].env.player_idx;
    f[0].env.grid[start + 2] = FORAGE_APPLE;
    f[0].env.grid[start + 3] = FORAGE_APPLE;
    f[0].env.grid[start + 4] = FORAGE_GRAPE;
    compute_observations(&f[0].env);
    for (int i = 0; i < 3; i++) {
        puf_step(&f[0].env);
        assert(!f[0].terminal && f[0].env.apples == i + 1 && f[0].env.log.n == 0);
    }
    puf_step(&f[0].env);
    assert(f[0].terminal && f[0].env.tick == 0 && f[0].env.log.successes == 1);
    assert(f[0].env.log.perf == 1);
    assert(f[0].reward > 0);
    assert(fabsf(f[0].env.log.episode_return - 3.996f) < 1e-6f);
    controlled(&f[0], FORAGE_EMPTY); f[0].env.max_steps = 1; puf_step(&f[0].env);
    assert(f[0].terminal && f[0].env.tick == 0);
    controlled(&f[0], FORAGE_TREE); int player = f[0].env.player_idx; puf_step(&f[0].env);
    assert(f[0].env.player_idx == player && f[0].reward == f[0].env.step_cost && !f[0].terminal); observation(&f[0]);
    // Partial apple credit, grape credit, penalty subtraction, clamp and copies.
    for (int copies = 1; copies <= 8; copies++) {
        for (int apples = 0; apples <= copies; apples++) {
            for (int bad = 0; bad <= copies + 1; bad++) {
                for (int grape = 0; grape < 2; grape++) {
                    controlled(&f[0], grape ? FORAGE_GRAPE : FORAGE_EMPTY);
                    f[0].env.copies = copies;
                    f[0].env.apples = apples;
                    f[0].env.negatives = bad;
                    f[0].env.max_steps = 1;
                    float expected = fminf(1, fmaxf(0, .25f * (apples + grape - bad)));
                    puf_step(&f[0].env);
                    assert(f[0].terminal && f[0].env.log.n == 1);
                    assert(fabsf(f[0].env.log.perf - expected) < 1e-7f);
                }
            }
        }
    }
    f[0].env.copies = 3;
    f[0].env.reward_other = 0;
    controlled(&f[0], FORAGE_PEAR);
    puf_step(&f[0].env);
    assert(f[0].reward == f[0].env.step_cost && !f[0].terminal);
    assert(f[0].env.grid[f[0].env.player_idx] == FORAGE_EMPTY);
    assert(f[0].env.apples == 0 && f[0].env.negatives == 0);
    observation(&f[0]);
    f[0].env.reward_good = 3;
    f[0].env.reward_other = 1.5f;
    f[0].env.reward_grape = 3;
    f[0].env.reward_bad = -2;
    f[0].env.reward_death = -4;
    f[0].env.step_cost = -.007f;
    const int kinds[] = {FORAGE_APPLE, FORAGE_PEAR, FORAGE_GRAPE, FORAGE_WASP, FORAGE_TOADSTOOL};
    const float rewards[] = {2.993f, 1.493f, 2.993f, -2.007f, -4.007f};
    const float scores[] = {.25, 0, .25, 0, 0};
    for (int i = 0; i < 5; i++) {
        controlled(&f[0], kinds[i]);
        memset(&f[0].env.log, 0, sizeof(Log));
        // Existing apple credit makes a missing penalty visible in perf.
        if (i >= 3) {
            f[0].env.apples = 1;
        }
        f[0].env.max_steps = 1;
        puf_step(&f[0].env);
        assert(f[0].terminal && fabsf(f[0].reward - rewards[i]) < 1e-6f);
        assert(fabsf(f[0].env.log.perf - scores[i]) < 1e-7f);
        assert(f[0].env.log.deaths == (i == 4));
    }
    for (int rep = 0; rep < FORAGE_REPRESENTATIONS; rep++) close_fixture(&f[rep]);
    puts("Sprite Forage: spawn coverage, connectivity, texture histograms, rewards, death/success/horizon, dirty pixels and appearance/RNG determinism PASS");
    return 0;
}
