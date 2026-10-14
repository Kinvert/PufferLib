// Game/observation tests only. No CPU CNN, GPU or policy.
#include <assert.h>
#include "../foragecnn.h"

typedef struct { Env env; float* storage; float action, reward, terminal; } Fixture;
static void setup(Fixture* f, int rep, unsigned int seed) {
    memset(f, 0, sizeof(*f));
    f->storage = (float*)calloc(OBS_SIZE + 2, sizeof(float)); assert(f->storage);
    f->storage[0] = 12345; f->storage[OBS_SIZE + 1] = -12345;
    f->env.agents[0].observations = f->storage + 1;
    f->env.agents[0].actions = &f->action; f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal; f->env.rng = seed;
    Dict kwargs = {0};
    dict_set(&kwargs, "rule_version", 1); dict_set(&kwargs, "num_agents", 1);
    dict_set(&kwargs, "copies_per_type", 3); dict_set(&kwargs, "trees", 8);
    dict_set(&kwargs, "max_steps", 512); dict_set(&kwargs, "step_cost", -.001);
    dict_set(&kwargs, "representation", rep);
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
    assert(a->env.rng == b->env.rng && a->env.player == b->env.player && a->env.tick == b->env.tick);
    assert(!memcmp(a->env.grid, b->env.grid, FORAGE_AREA));
    assert(a->reward == b->reward && a->terminal == b->terminal);
    assert(!memcmp(&a->env.log, &b->env.log, sizeof(Log)));
}
static void histogram(int kind, int counts[256]) {
    unsigned char image[256]; forage_sprite(kind, image); memset(counts, 0, 256 * sizeof(int));
    for (int i = 0; i < 256; i++) counts[image[i]]++;
}
typedef struct { Fixture* fixture; int representation; } ThreadInit;
static void* threaded_init(void* raw) {
    ThreadInit* task = (ThreadInit*)raw;
    setup(task->fixture, task->representation, 7); return NULL;
}
static void controlled(Fixture* f, int kind) {
    Env* e = &f->env; memset(e->grid, 0, FORAGE_AREA);
    for (int i = 0; i < FORAGE_AREA; i++) if (i < FORAGE_WIDTH || i >= FORAGE_AREA - FORAGE_WIDTH
        || i % FORAGE_WIDTH == 0 || i % FORAGE_WIDTH == FORAGE_WIDTH - 1) e->grid[i] = FORAGE_TREE;
    e->player = FORAGE_WIDTH * 7 + 9; e->grid[e->player + 1] = kind;
    e->remaining = 21; e->positives = e->negatives = e->tick = 0; e->positive_value = e->episode_return = 0;
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
    pthread_t threads[FORAGE_REPRESENTATIONS]; ThreadInit tasks[FORAGE_REPRESENTATIONS];
    for (int rep = 0; rep < FORAGE_REPRESENTATIONS; rep++) {
        tasks[rep].fixture = &f[rep]; tasks[rep].representation = rep;
        assert(!pthread_create(&threads[rep], NULL, threaded_init, &tasks[rep]));
    }
    for (int rep = 0; rep < FORAGE_REPRESENTATIONS; rep++) assert(!pthread_join(threads[rep], NULL));
    for (int rep = 0; rep < FORAGE_REPRESENTATIONS; rep++) {
        int counts[13] = {0}; for (int i = 0; i < FORAGE_AREA; i++) counts[f[rep].env.grid[i]]++;
        for (int kind = 1; kind <= 10; kind++) assert(counts[kind] == 3);
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
        for (int kind = 1; kind <= 10; kind++) assert(counts[kind] == 3);
        assert(!f[0].env.grid[f[0].env.player]);
    }
    for (int kind = 1; kind <= 10; kind++) {
        controlled(&f[0], kind); puf_step(&f[0].env);
        assert(fabsf(f[0].reward - (FORAGE_REWARDS[kind] - .001f)) < 1e-6f);
        assert((f[0].terminal != 0) == (kind == FORAGE_TOADSTOOL));
        if (kind == FORAGE_TOADSTOOL) assert(f[0].env.last_end_reason == 1 && f[0].env.tick == 0);
        else { assert(f[0].env.grid[f[0].env.player] == FORAGE_EMPTY); observation(&f[0]); }
    }
    controlled(&f[0], FORAGE_APPLE); f[0].env.remaining = 1; puf_step(&f[0].env);
    assert(f[0].terminal && f[0].env.last_end_reason == 2 && f[0].env.last_decisions == 1);
    controlled(&f[0], FORAGE_EMPTY); f[0].env.max_steps = 1; puf_step(&f[0].env);
    assert(f[0].terminal && f[0].env.last_end_reason == 3 && f[0].env.last_decisions == 1);
    controlled(&f[0], FORAGE_TREE); int player = f[0].env.player; puf_step(&f[0].env);
    assert(f[0].env.player == player && f[0].reward == f[0].env.step_cost && !f[0].terminal); observation(&f[0]);
    for (int rep = 0; rep < FORAGE_REPRESENTATIONS; rep++) close_fixture(&f[rep]);
    puts("ForageCNN: spawn coverage, connectivity, texture histograms, rewards, death/success/horizon, dirty pixels and appearance/RNG determinism PASS");
    return 0;
}
