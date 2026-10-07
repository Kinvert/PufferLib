// Native world/raster only. Independent array/deque reference, never a policy.
#include <assert.h>
#include <inttypes.h>
#include ENV_HEADER

static double representation, representation_mode, representation_seed;
static const char* invalid_key;
static double invalid_value;
typedef struct {
    Env env;
    float obs[OBS_SIZE + 2], action, reward, terminal;
} Fixture;
typedef struct {
    Env rules;
    unsigned char grid[128 * 128];
    int positions[128 * 128], length, tick, food;
    unsigned int rng;
    float total, reward, terminal;
    Log log;
    int last_score, last_food, last_decisions, last_end_reason;
    float last_return;
} Reference;

static void setup(Fixture* f, unsigned int seed, int width, int food, int ring, int horizon) {
    memset(f, 0, sizeof(*f));
    f->obs[0] = 12345; f->obs[OBS_SIZE + 1] = -12345;
    f->env.agents[0].observations = f->obs + 1;
    f->env.agents[0].actions = &f->action;
    f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal;
    f->env.rng = seed;
    Dict kwargs = {0};
    dict_set(&kwargs, "rule_version", 1); dict_set(&kwargs, "num_agents", 1);
    dict_set(&kwargs, "width", width); dict_set(&kwargs, "height", width);
    dict_set(&kwargs, "num_food", food); dict_set(&kwargs, "vision", 5);
    dict_set(&kwargs, "leave_corpse_on_death", 0); dict_set(&kwargs, "max_snake_length", ring);
    dict_set(&kwargs, "max_steps", horizon); dict_set(&kwargs, "cell_size", 12);
    dict_set(&kwargs, "reward_food", .1); dict_set(&kwargs, "reward_death", -1);
    dict_set(&kwargs, "representation", representation);
    dict_set(&kwargs, "representation_mode", representation_mode);
    dict_set(&kwargs, "representation_seed", representation_seed);
    if (invalid_key) dict_set(&kwargs, invalid_key, invalid_value);
    puf_init(&f->env, &kwargs); dict_clear(&kwargs);
    assert(f->env.rng == seed);
    puf_reset(&f->env);
}

// Enumerate candidate cells, unlike the production count/rank scanning path.
static int ref_empty(Reference* r) {
    int free_cells[128 * 128], count = 0;
    for (int y = 5; y < r->rules.height - 5; y++) {
        for (int x = 5; x < r->rules.width - 5; x++) {
            int cell = y * r->rules.width + x;
            if (r->grid[cell] == 0) free_cells[count++] = cell;
        }
    }
    assert(count > 0);
    uint32_t threshold = (uint32_t)(UINT64_C(4294967296) % count);
    do { r->rng = (uint32_t)(UINT64_C(1664525)*r->rng + UINT64_C(1013904223)); } while (r->rng < threshold);
    return free_cells[r->rng % (unsigned int)count];
}
static void ref_reset(Reference* r) {
    for (int y = 0; y < r->rules.height; y++) {
        for (int x = 0; x < r->rules.width; x++) {
            r->grid[y * r->rules.width + x] =
                y >= 5 && y < r->rules.height-5 && x >= 5 && x < r->rules.width-5 ? 0 : 3;
        }
    }
    r->length = 1; r->tick = r->food = 0; r->total = r->reward = r->terminal = 0;
    r->last_score = r->last_food = r->last_decisions = r->last_end_reason = 0; r->last_return = 0;
    r->positions[0] = ref_empty(r); r->grid[r->positions[0]] = 7;
    for (int i = 0; i < r->rules.food; i++) r->grid[ref_empty(r)] = 1;
}
static void ref_end(Reference* r, int reason) {
    int score = r->length, food = r->food, tick = r->tick;
    float total = r->total, reward = r->reward;
    r->log.score += score; r->log.perf += fminf(score/120.0f, 1);
    r->log.food_collected += food; r->log.episode_return += total;
    r->log.episode_length += tick; r->log.deaths += reason == 1;
    r->log.horizon_ends += reason == 2; r->log.n++;
    ref_reset(r);
    r->last_score = score; r->last_food = food; r->last_decisions = tick;
    r->last_return = total; r->last_end_reason = reason; r->reward = reward; r->terminal = 1;
}
static void ref_step(Reference* r, int action) {
    r->tick++; r->reward = r->terminal = 0; r->last_end_reason = 0;
    int directions[4] = {-r->rules.width, r->rules.width, -1, 1};
    int next = r->positions[0] + directions[action];
    if (r->length > 1 && next == r->positions[1]) next = r->positions[0] - directions[action];
    int tile = r->grid[next];
    if (tile == 3 || tile == 7) { r->reward = r->rules.reward_death; r->total += r->reward; ref_end(r, 1); return; }
    bool grow = tile == 1 && r->length < r->rules.max_snake_length - 1;
    if (!grow) r->grid[r->positions[r->length-1]] = 0;
    int length = r->length + grow;
    for (int i = length-1; i > 0; i--) r->positions[i] = r->positions[i-1];
    r->positions[0] = next; r->length = length; r->grid[next] = 7;
    if (tile == 1) { r->food++; r->reward = r->rules.reward_food; r->total += r->reward; r->grid[ref_empty(r)] = 1; }
    if (r->tick == r->rules.max_steps) ref_end(r, 2);
}
static void compare(Fixture* f, Reference* r) {
    Env* e = &f->env;
    assert(e->rng == r->rng && e->length == r->length && e->tick == r->tick && e->food_collected == r->food);
    assert(e->episode_return == r->total && f->reward == r->reward && f->terminal == r->terminal);
    assert(e->last_score == r->last_score && e->last_food == r->last_food && e->last_decisions == r->last_decisions);
    assert(e->last_end_reason == r->last_end_reason && e->last_return == r->last_return);
    assert(memcmp(&e->log, &r->log, sizeof(Log)) == 0);
    assert(memcmp(e->grid, r->grid, e->width * e->height) == 0);
    for (int i = 0; i < e->length; i++) {
        int ptr = (e->head_ptr + e->max_snake_length - i) % e->max_snake_length;
        assert(e->snake[ptr] == r->positions[i]);
    }
    int bodies = 0, foods = 0;
    for (int i = 0; i < e->width * e->height; i++) {
        bodies += e->grid[i] == 7; foods += e->grid[i] == 1;
        assert(e->grid[i] == 0 || e->grid[i] == 1 || e->grid[i] == 3 || e->grid[i] == 7);
    }
    assert(bodies == e->length && foods == e->food);
}
static Reference reference(Fixture* f, unsigned int seed) {
    Reference r = {0}; r.rules = f->env; r.rng = seed; ref_reset(&r); return r;
}
static void from_fixture(Reference* r, Fixture* f) {
    *r = (Reference){0}; r->rules = f->env; r->rng = f->env.rng;
    memcpy(r->grid, f->env.grid, f->env.width * f->env.height);
    r->length = f->env.length; r->tick = f->env.tick; r->total = f->env.episode_return;
    r->food = f->env.food_collected; r->log = f->env.log;
    for (int i = 0; i < r->length; i++) r->positions[i] = f->env.snake[(f->env.head_ptr + f->env.max_snake_length - i) % f->env.max_snake_length];
}
static uint64_t world_hash(Fixture* f) {
    Env* e = &f->env; uint64_t hash = UINT64_C(14695981039346656037);
    const unsigned char* bytes = e->grid;
    for (int i = 0; i < e->width*e->height; i++) hash = (hash ^ bytes[i])*UINT64_C(1099511628211);
    int integers[] = {e->length, e->head_ptr, e->tick, e->food_collected, e->last_score, e->last_food,
        e->last_decisions, e->last_end_reason};
    for (unsigned int k = 0; k < sizeof(integers)/sizeof(int); k++) {
        uint32_t value = (uint32_t)integers[k];
        for (int b = 0; b < 4; b++) hash = (hash ^ ((value >> (8*b)) & 255))*UINT64_C(1099511628211);
    }
    for (int i = 0; i < e->max_snake_length; i++) {
        uint32_t value = (uint32_t)e->snake[i];
        for (int b = 0; b < 4; b++) hash = (hash ^ ((value >> (8*b)) & 255))*UINT64_C(1099511628211);
    }
    for (int b = 0; b < 4; b++) hash = (hash ^ ((e->rng >> (8*b)) & 255))*UINT64_C(1099511628211);
    float scalars[] = {e->episode_return, e->last_return, f->reward, f->terminal,
        e->log.perf, e->log.score, e->log.food_collected, e->log.episode_return,
        e->log.episode_length, e->log.deaths, e->log.horizon_ends, e->log.n};
    for (unsigned int k = 0; k < sizeof(scalars)/sizeof(float); k++) {
        uint32_t value; memcpy(&value, &scalars[k], 4);
        for (int b = 0; b < 4; b++) hash = (hash ^ ((value >> (8*b)) & 255))*UINT64_C(1099511628211);
    }
    return hash;
}
static void observation(Fixture* f) {
    assert(f->obs[0] == 12345 && f->obs[OBS_SIZE+1] == -12345);
    float saved[OBS_SIZE]; memcpy(saved, f->obs+1, sizeof(saved));
    uint32_t rng = f->env.rng;
    for (int i = 0; i < OBS_SIZE; i++) f->obs[i+1] = 99;
    compute_observations(&f->env);
    assert(memcmp(saved, f->obs+1, sizeof(saved)) == 0 && rng == f->env.rng);
    int head = f->env.snake[f->env.head_ptr];
    for (int i = 0; i < OBS_SIZE; i++) {
        float expected = 0;
#if SNAKEBENCH_PIXELS
        int py = i / 44, px = i % 44;
        if (f->env.representation == 5) px = 43-px;
        if (px >= 5 && px < 38 && py >= 1 && py < 34) {
            int x = (px-5)/3, y = (py-1)/3, dx = (px-5)%3, dy = (py-1)%3;
            int cell = (head/f->env.width-5+y)*f->env.width + head%f->env.width-5+x;
            switch (f->env.grid[cell]) { case 1: expected=.25f; break; case 3: expected=.5f; break; case 7: expected=.75f; break; }
            if (f->env.representation == 4 && expected != .5f && expected != 0) expected = 1-expected;
            if (x == 5 && y == 5) expected = 1;
            if (f->env.representation == 1 && (dx-1)*(dx-1)+(dy-1)*(dy-1) > 1) expected = 0;
            if (f->env.representation == 2 && (dx == 2 || dy == 2)) expected = 0;
        }
        if (f->env.representation == 3) expected = 1-expected;
#else
        int x = (i/8)%11, y = (i/8)/11;
        int tile = f->env.grid[(head/f->env.width-5+y)*f->env.width + head%f->env.width-5+x];
        expected = i%8 == tile;
#endif
        assert(f->obs[i+1] == expected);
    }
}

static void place(Fixture* f, const int* cells, int count, int head_ptr) {
    for (int i = 0; i < f->env.width*f->env.height; i++) if (f->env.grid[i] == 7) f->env.grid[i] = 0;
    for (int i = 0; i < f->env.max_snake_length; i++) f->env.snake[i] = -1;
    f->env.length = count; f->env.head_ptr = head_ptr;
    for (int i = 0; i < count; i++) {
        f->env.snake[(head_ptr+f->env.max_snake_length-i)%f->env.max_snake_length] = cells[i];
        f->env.grid[cells[i]] = 7;
    }
}
static void events(void) {
    Fixture f; setup(&f, 0, 26, 1, 3, 100);
    int cells[] = {13*26+13, 13*26+12}; place(&f, cells, 2, 2);
    for (int i = 0; i < 26*26; i++) if (f.env.grid[i] == 1) f.env.grid[i] = 0;
    f.env.grid[cells[0]+1] = 1;
    Reference r; from_fixture(&r, &f);
    f.action = 2; puf_step(&f.env); ref_step(&r, 2); compare(&f, &r);
    assert(f.env.snake[f.env.head_ptr] == cells[0]+1 && f.env.length == 2 && f.env.food_collected == 1);
    assert(f.env.head_ptr == 0); // Circular head write wrapped at its capacity.
    assert(f.reward == f.env.reward_food && !f.terminal); observation(&f);
    printf("neck-growth-limit %016" PRIx64 "\n", world_hash(&f)); puf_close(&f.env);

    setup(&f, 1, 26, 1, 8, 1);
    int body[] = {13*26+13, 13*26+12, 14*26+12, 14*26+13}; place(&f, body, 4, 0);
    from_fixture(&r, &f); f.action = 1; puf_step(&f.env); ref_step(&r, 1); compare(&f, &r);
    assert(f.terminal == 1 && f.reward == -1 && f.env.last_score == 4 && f.env.last_end_reason == 1);
    assert(f.env.last_decisions == 1 && f.env.last_return == -1 && f.env.length == 1);
    printf("occupied-tail-death-at-horizon %016" PRIx64 "\n", world_hash(&f)); puf_close(&f.env);

    setup(&f, UINT32_MAX, 26, 1, 8, 1); place(&f, cells, 1, 0);
    for (int i = 0; i < 26*26; i++) if (f.env.grid[i] == 1) f.env.grid[i] = 0;
    f.env.grid[cells[0]+1] = 1; from_fixture(&r, &f);
    f.action = 3; puf_step(&f.env); ref_step(&r, 3); compare(&f, &r); observation(&f);
    assert(f.terminal == 1 && f.reward == f.env.reward_food && f.env.last_score == 2);
    assert(f.env.last_food == 1 && f.env.last_end_reason == 2 && f.env.last_return == f.env.reward_food);
    printf("food-horizon %016" PRIx64 "\n", world_hash(&f)); puf_close(&f.env);

    setup(&f, 17, 26, 1, 8, 1);
    int edge = 5*26+5; place(&f, &edge, 1, 0); from_fixture(&r, &f);
    f.action = 0; puf_step(&f.env); ref_step(&r, 0); compare(&f, &r);
    assert(f.terminal == 1 && f.reward == -1 && f.env.last_end_reason == 1);
    printf("wall-death-at-horizon %016" PRIx64 "\n", world_hash(&f)); puf_close(&f.env);

    setup(&f, 73, 26, 4, 128, 64);
    unsigned char* grid = f.env.grid; int* ring = f.env.snake;
    memset(grid, 255, 26*26); for (int i = 0; i < 128; i++) ring[i] = 99999;
    f.env.tick = 999; f.env.length = 77; f.env.food_collected = 100; f.env.episode_return = 42;
    f.env.rng = 73; puf_reset(&f.env); r = reference(&f, 73); compare(&f, &r);
    assert(f.env.grid == grid && f.env.snake == ring); observation(&f);
    printf("dirty-reset %016" PRIx64 "\n", world_hash(&f));
    // Guarantee the moved food is outside the crop; never conditionally skip this assertion.
    int center = 13*26+13; place(&f, &center, 1, 0);
    for (int i = 0; i < 26*26; i++) if (f.env.grid[i] == 1) f.env.grid[i] = 0;
    int outside[] = {6*26+6, 6*26+20, 20*26+6, 20*26+20};
    for (int i = 0; i < 4; i++) f.env.grid[outside[i]] = 1;
    compute_observations(&f.env); observation(&f);
    float before[OBS_SIZE]; memcpy(before, f.obs+1, sizeof(before));
    f.env.grid[outside[0]] = 0; f.env.grid[outside[0]+1] = 1;
    compute_observations(&f.env); assert(memcmp(before, f.obs+1, sizeof(before)) == 0);
    f.env.tick = 1000; f.env.food_collected = 123; f.env.last_return = 99; f.env.rng = 5;
    compute_observations(&f.env); assert(memcmp(before, f.obs+1, sizeof(before)) == 0);
    puf_close(&f.env);
}
static void trajectories(void) {
    for (int board = 0; board < 3; board++) {
        int width = board == 0 ? 13 : board == 1 ? 26 : 38;
        int food = board == 0 ? 7 : 4, ring = board == 0 ? 3 : board == 1 ? 8 : 128;
        for (unsigned int s = 0; s < 32; s++) {
            uint32_t seed = s == 31 ? UINT32_MAX : s * UINT32_C(0x9e3779b9);
            Fixture f; setup(&f, seed, width, food, ring, 37);
            Reference r = reference(&f, seed); compare(&f, &r); observation(&f);
            for (int t = 0; t < 512; t++) {
                int action = ((s*7 + t*13) ^ ((t+3)/5)) % 4;
                f.action = action; puf_step(&f.env); ref_step(&r, action); compare(&f, &r);
                observation(&f);
            }
            printf("trajectory %d %u %016" PRIx64 "\n", width, seed, world_hash(&f));
            puf_close(&f.env);
        }
    }
}
static void order_independence(void) {
    Fixture a, b, aa, bb;
    setup(&a, 11, 26, 4, 128, 37); setup(&b, 17, 26, 4, 128, 37);
    setup(&aa, 11, 26, 4, 128, 37); setup(&bb, 17, 26, 4, 128, 37);
    srand(999);
    for (int t = 0; t < 256; t++) {
        a.action = aa.action = t%4; b.action = bb.action = (t/3)%4;
        puf_step(&a.env); (void)rand(); puf_step(&b.env);
        puf_step(&bb.env); (void)rand(); puf_step(&aa.env);
        assert(world_hash(&a) == world_hash(&aa) && world_hash(&b) == world_hash(&bb));
    }
    printf("order-independent %016" PRIx64 " %016" PRIx64 "\n", world_hash(&a), world_hash(&b));
    puf_close(&a.env); puf_close(&b.env); puf_close(&aa.env); puf_close(&bb.env);
}
int main(int argc, char** argv) {
    if (argc == 4 && !strcmp(argv[1], "--invalid")) {
        invalid_key = argv[2]; invalid_value = strtod(argv[3], NULL);
        Fixture f; setup(&f, 0, 26, 4, 128, 37); puf_close(&f.env); return 0;
    }
    if (argc == 3 && !strcmp(argv[1], "--action")) {
        Fixture f; setup(&f, 0, 26, 4, 128, 37); f.action = strtof(argv[2], NULL);
        puf_step(&f.env); puf_close(&f.env); return 0;
    }
    representation = argc > 1 ? strtod(argv[1], NULL) : 0;
    representation_mode = argc > 2 ? strtod(argv[2], NULL) : 0;
    representation_seed = argc > 3 ? strtod(argv[3], NULL) : 0;
    events(); trajectories(); order_independence();
    return 0;
}
