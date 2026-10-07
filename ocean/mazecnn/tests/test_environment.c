// Native Maze game/raster checks. No neural network executes.
#include <inttypes.h>
#include ENV_HEADER

#ifdef TEST_PIXELS
static double representation, mode, appearance_seed;
#endif

typedef struct {
    Env env;
    obs_t observations[OBS_SIZE + 2];
    float action, reward, terminal;
} Fixture;

static uint64_t hash_bytes(uint64_t h, const void* data, size_t n) {
    const unsigned char* p = data;
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}

static Dict settings(int maps, int size) {
    Dict d = {0};
    dict_set(&d, "num_maps", maps); dict_set(&d, "map_size", size);
#ifdef TEST_PIXELS
    dict_set(&d, "representation", representation);
    dict_set(&d, "representation_mode", mode);
    dict_set(&d, "representation_seed", appearance_seed);
#endif
    return d;
}

static void bind(Fixture* f) {
    f->observations[0] = 73; f->observations[OBS_SIZE + 1] = 91;
    f->env.agents[0].observations = f->observations + 1;
    f->env.agents[0].actions = &f->action;
    f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal;
}

static void setup(Fixture* f, unsigned int seed, int maps, int size) {
    memset(f, 0, sizeof(*f)); bind(f); f->env.rng = seed;
    Dict d = settings(maps, size);
    puf_init(&f->env, &d); dict_clear(&d);
    assert(f->env.rng == seed);
    puf_reset(&f->env);
}

static int crop(State* s, int row, int col) {
    int y = s->y + row - 5, x = s->x + col - 5;
    return x < 0 || x >= 47 || y < 0 || y >= 47 ? 0 : s->maze[y * 47 + x];
}

static void pixels(Fixture* f) {
    assert(f->observations[0] == 73 && f->observations[OBS_SIZE + 1] == 91);
    obs_t* obs = f->env.agents[0].observations;
#ifdef TEST_PIXELS
    assert(OBS_CHANNELS == 1 && OBS_HEIGHT == 36 && OBS_WIDTH == 44);
    int id = f->env.representation;
    for (int y = 0; y < 36; y++) for (int x = 0; x < 44; x++) {
        float expected = 0;
        if (x >= 5 && x < 38 && y >= 1 && y < 34) {
            int row = (y - 1) / 3, col = (x - 5) / 3;
            int px = (x - 5) % 3, py = (y - 1) % 3;
            int tile = crop(&f->env.state, row, id == 5 ? 10 - col : col);
            expected = tile == 1 ? .5f : tile == 2 ? 1 : tile == 4 ? .75f : 0;
            if (id == 4) expected = tile == 1 ? .75f : tile == 2 ? 1 : tile == 4 ? .5f : 0;
            if (id == 1 && px != 1 && py != 1) expected = 0;
            if (id == 2 && (px == 2 || py == 2)) expected = 0;
        }
        if (id == 3) expected = 1 - expected;
        assert(obs[y * 44 + x] == expected);
    }
#else
    for (int y = 0; y < 11; y++) for (int x = 0; x < 11; x++) assert(obs[y * 11 + x] == crop(&f->env.state, y, x));
#endif
    obs_t saved[OBS_SIZE]; memcpy(saved, obs, sizeof(saved));
    unsigned int rng = f->env.rng; State state = f->env.state;
    memset(obs, 99, sizeof(saved)); compute_observations(&f->env);
    assert(memcmp(saved, obs, sizeof(saved)) == 0 && rng == f->env.rng);
    assert(memcmp(&state, &f->env.state, sizeof(state)) == 0);
}

// Independent BFS validates generated levels and supplies test actions only.
static int next_action(State* s) {
    int queue[47 * 47], first[47 * 47] = {0};
    unsigned char seen[47 * 47] = {0};
    int start = s->y * 47 + s->x, head = 0, tail = 1;
    queue[0] = start; seen[start] = 1;
    const int delta[5] = {0, 1, -47, -1, 47};
    while (head < tail) {
        int here = queue[head++];
        if (s->maze[here] == 4) return first[here];
        for (int action = 1; action < 5; action++) {
            int dest = here + delta[action], x = dest % 47, y = dest / 47;
            if (dest < 0 || dest >= 47 * 47 || x < 0 || x >= s->width || y >= s->height
                    || s->maze[dest] == 1 || seen[dest]) continue;
            seen[dest] = 1; first[dest] = here == start ? action : first[here];
            assert(tail < 47 * 47); queue[tail++] = dest;
        }
    }
    assert(!"Generated level has no reachable goal"); return 0;
}

typedef struct { State state; Log log; int tick; unsigned int rng; } Reference;

static void ref_log(Reference* r, float reward) {
    r->log.perf += reward; r->log.score += reward;
    r->log.episode_return += reward; r->log.episode_length += r->tick; r->log.n++;
}

static void reference_step(Reference* r, State* levels, int count, int action,
        float* reward, float* terminal) {
    *reward = *terminal = 0; r->tick++;
    int direction = action ? action : r->state.direction;
    int x = r->state.x + (direction == 1) - (direction == 3);
    int y = r->state.y + (direction == 4) - (direction == 2);
    // Preserve the original inclusive bound; generated borders prevent escape.
    if (x >= 0 && x <= r->state.width && y >= 0 && y <= r->state.height) {
        int dest = y * 47 + x;
        assert(dest >= 0 && dest < 47 * 47);
        if (r->state.maze[dest] != 1) {
            if (r->state.maze[dest] == 4) { *reward = *terminal = 1; ref_log(r, 1); }
            r->state.maze[r->state.y * 47 + r->state.x] = 0;
            r->state.maze[dest] = 2; r->state.x = x; r->state.y = y;
        }
    }
    if (r->tick >= 2 * r->state.width * r->state.height) { *terminal = 1; ref_log(r, *reward); }
    if (*terminal) {
        r->tick = 0;
        // Original terminal path selects two levels. Retain both RNG draws.
        rand_r(&r->rng);
        r->state = levels[rand_r(&r->rng) % count];
    }
}

static void trajectories(int maps, int size, unsigned int seed) {
    Fixture f; setup(&f, seed, maps, size);
    uint64_t h = hash_bytes(UINT64_C(14695981039346656037), f.env.levels, maps * sizeof(State));
    for (int i = 0; i < maps; i++) {
        State* s = &f.env.levels[i];
        assert(s->width >= 5 && s->width <= 47 && (s->width & 1) && s->width == s->height);
        assert(s->maze[48] == 2 && s->maze[(s->height - 2) * 47 + s->width - 2] == 4);
        assert(next_action(s) >= 1);
    }
    Reference r = {f.env.state, f.env.log, f.env.tick, f.env.rng};
    unsigned int actions = seed ^ 913;
    int successes = 0, caps = 0;
    for (int i = 0; i < 12288; i++) {
        int action = i % 128 < 32 ? 0 : i % 8 ? rand_r(&actions) % 5 : next_action(&r.state);
        float reward, terminal;
        reference_step(&r, f.env.levels, maps, action, &reward, &terminal);
        f.action = action; puf_step(&f.env);
        assert(f.reward == reward && f.terminal == terminal);
        assert(f.env.tick == r.tick && f.env.rng == r.rng);
        assert(memcmp(&f.env.state, &r.state, sizeof(State)) == 0);
        assert(memcmp(&f.env.log, &r.log, sizeof(Log)) == 0);
        successes += reward == 1; caps += terminal && reward == 0;
        if (!(i % 128) || terminal) pixels(&f);
        h = hash_bytes(h, &r.state, sizeof(State));
        h = hash_bytes(h, &r.log, sizeof(Log)); h = hash_bytes(h, &r.rng, sizeof(r.rng));
    }
    printf("maps=%d size=%d seed=%u hash=%016" PRIx64 " goals=%d caps=%d\n", maps, size, seed, h, successes, caps);
    puf_close(&f.env);
}

static void fixtures(void) {
    Fixture f; setup(&f, 19, 8, 5);
    // Corners, local crop and offscreen information isolation.
    for (int edge = 0; edge < 4; edge++) {
        f.env.state.x = edge & 1 ? 46 : 0; f.env.state.y = edge & 2 ? 46 : 0;
        compute_observations(&f.env);
        pixels(&f);
    }
    puf_reset(&f.env);
    obs_t saved[OBS_SIZE]; memcpy(saved, f.env.agents[0].observations, sizeof(saved));
    f.env.state.maze[46 * 47 + 46] = GOAL;
    compute_observations(&f.env);
    assert(memcmp(saved, f.env.agents[0].observations, sizeof(saved)) == 0);
    // Goal on timeout decision produces two original log entries and two draws.
    puf_reset(&f.env); f.env.state.maze[49] = GOAL;
    f.env.tick = 49; f.action = ATN_EAST;
    float old_n = f.env.log.n; unsigned int rng = f.env.rng;
    rand_r(&rng); rand_r(&rng); puf_step(&f.env);
    assert(f.terminal == 1 && f.reward == 1 && f.env.tick == 0);
    assert(f.env.log.n == old_n + 2 && f.env.rng == rng); pixels(&f);
    // Dirty state is overwritten by original level reset; same seed repeats.
    unsigned int reset_rng = 53; f.env.rng = reset_rng; puf_reset(&f.env);
    State expected = f.env.state;
    memset(&f.env.state, 99, sizeof(State)); f.env.tick = 999;
    f.env.rng = reset_rng; puf_reset(&f.env);
    assert(f.env.tick == 0 && memcmp(&expected, &f.env.state, sizeof(State)) == 0);
    pixels(&f); puf_close(&f.env);
}

static void vector_order(void) {
    Dict env = settings(16, 11), vec = {0};
    dict_set(&vec, "total_agents", 16); dict_set(&vec, "num_buffers", 4);
    int count, starts[4], counts[4];
    Env* group = my_vec_init(&count, starts, counts, &vec, &env);
    assert(count == 16);
    Fixture f[16] = {0};
    for (int i = 0; i < 16; i++) {
        assert(group[i].levels == group[0].levels && !group[i].owns_levels);
        f[i].env = group[i]; bind(&f[i]); puf_reset(&f[i].env); pixels(&f[i]);
#ifdef TEST_PIXELS
        assert(group[i].representation == cnn_appearance_init(&env, i, 6));
#endif
    }
    for (int b = 0; b < 4; b++) assert(starts[b] == b * 4 && counts[b] == 4);
    unsigned int rng[16]; State states[16];
    for (int i = 0; i < 16; i++) {
        f[i].env.rng = group[i].rng; puf_reset(&f[i].env);
        rng[i] = f[i].env.rng; states[i] = f[i].env.state;
    }
    srand(99);
    for (int i = 15; i >= 0; i--) {
        rand(); f[i].env.rng = group[i].rng; puf_reset(&f[i].env);
        assert(f[i].env.rng == rng[i] && memcmp(&states[i], &f[i].env.state, sizeof(State)) == 0);
    }
    uint64_t h = hash_bytes(UINT64_C(14695981039346656037), states, sizeof(states));
    printf("vector-order hash=%016" PRIx64 "\n", h);
    my_vec_close(group); free(group); dict_clear(&env); dict_clear(&vec);
}

int main(int argc, char** argv) {
#ifdef TEST_PIXELS
    if (argc == 4 && strcmp(argv[1], "--invalid") == 0) {
        Dict d = settings(8, 11); dict_set(&d, argv[2], strtod(argv[3], NULL));
        Fixture f = {0}; bind(&f); puf_init(&f.env, &d); return 0;
    }
    if (argc == 4 && strcmp(argv[1], "--invalid-vec") == 0) {
        Dict d = settings(8, 11), v = {0};
        dict_set(&v, "total_agents", 16); dict_set(&v, "num_buffers", 4);
        if (!strcmp(argv[2], "num_maps") || !strcmp(argv[2], "map_size")) dict_set(&d, argv[2], strtod(argv[3], NULL));
        else dict_set(&v, argv[2], strtod(argv[3], NULL));
        int n, starts[16], counts[16]; my_vec_init(&n, starts, counts, &v, &d); return 0;
    }
    representation = argc > 1 ? strtod(argv[1], NULL) : 0;
    mode = argc > 2 ? strtod(argv[2], NULL) : 0;
    appearance_seed = argc > 3 ? strtod(argv[3], NULL) : 0;
#endif
    fixtures(); vector_order();
    trajectories(32, 5, 73); trajectories(64, 11, 173);
    trajectories(64, 47, 53111); trajectories(128, -1, 53112);
    puts("PASS: native Maze game/reference/local-raster checks"); return 0;
}
