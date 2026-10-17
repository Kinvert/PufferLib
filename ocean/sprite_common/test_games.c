// Game/observation tests only. No neural inference or GPU context.
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static float* tracked_pixels;
static size_t copied_bytes;
static void* track_copy(void* dst, const void* src, size_t size) {
    uintptr_t at = (uintptr_t)dst;
    uintptr_t first = (uintptr_t)tracked_pixels;
    if (tracked_pixels && at >= first && at < first + 640 * 480 * 3 * sizeof(float)) {
        copied_bytes += size;
    }
    return memcpy(dst, src, size);
}
#define memcpy track_copy
#include GAME_HEADER
#undef memcpy

typedef struct {
    Env env;
    float* buffer;
    float action, reward, terminal;
    int size;
} Fixture;

void setup(Fixture* f, int mode, int rep, unsigned int seed) {
    memset(f, 0, sizeof(*f));
    Ini ini = {0};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, GAME_CONFIG);
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    dict_set(kwargs, "observation_mode", mode);
    dict_set(kwargs, "representation", rep);
    dict_set(kwargs, "seed", 2027);
    f->size = puf_observation_size(kwargs);
    assert(f->size == (mode ? PIXEL_OBS_SIZE : STATE_OBS_SIZE));
    f->buffer = calloc(f->size + 2, sizeof(float));
    assert(f->buffer);
    f->buffer[0] = 12345;
    f->buffer[f->size + 1] = -12345;
    f->env.rng = seed;
    f->env.agents[0].observations = f->buffer + 1;
    f->env.agents[0].actions = &f->action;
    f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal;
    puf_init(&f->env, kwargs);
    assert(f->env.rng == seed + 2027);
    tracked_pixels = mode ? f->buffer + 1 : NULL;
    copied_bytes = 0;
    puf_reset(&f->env);
    if (mode) {
        assert(copied_bytes == PIXEL_OBS_SIZE * sizeof(float));
        copied_bytes = 0;
        compute_observations(&f->env);
        assert(copied_bytes == 0 && "Unchanged tiles must not be copied again");
    }
    tracked_pixels = NULL;
    puf_ini_free(&ini);
}

void same_world(Fixture* a, Fixture* b) {
    assert(a->env.player == b->env.player && a->env.tick == b->env.tick);
    assert(a->env.rng == b->env.rng && a->env.episode_return == b->env.episode_return);
    assert(a->env.floor_seed == b->env.floor_seed);
    assert(!memcmp(a->env.grid, b->env.grid, sizeof(a->env.grid)));
    assert(!memcmp(&a->env.log, &b->env.log, sizeof(Log)));
    assert(a->reward == b->reward && a->terminal == b->terminal);
#ifdef SUITE_CATCH
    assert(a->env.object_rng == b->env.object_rng);
    assert(a->env.object == b->env.object && a->env.good == b->env.good);
    assert(a->env.count == b->env.count && a->env.correct == b->env.correct);
#endif
#ifdef SUITE_RUNNER
    assert(a->env.object_rng == b->env.object_rng);
    assert(a->env.object == b->env.object && a->env.kind == b->env.kind);
    assert(a->env.count == b->env.count && a->env.correct == b->env.correct);
    assert(a->env.vy == b->env.vy);
#endif
#ifdef SUITE_PLATFORM
    assert(a->env.vy == b->env.vy);
    assert(a->env.goal == b->env.goal);
#endif
#ifdef SUITE_MAZE
    assert(a->env.goal == b->env.goal);
    assert(!memcmp(a->env.distance, b->env.distance, sizeof(a->env.distance)));
#endif
#ifdef SUITE_KEY
    assert(a->env.key == b->env.key && a->env.door == b->env.door);
    assert(a->env.goal == b->env.goal && a->env.has_key == b->env.has_key);
    assert(a->env.mimic == b->env.mimic && a->env.floor_seed == b->env.floor_seed);
    assert(!memcmp(a->env.key_distance, b->env.key_distance, sizeof(a->env.key_distance)));
    assert(!memcmp(a->env.goal_distance, b->env.goal_distance, sizeof(a->env.goal_distance)));
#endif
#ifdef SUITE_PUSH
    assert(a->env.crate == b->env.crate && a->env.goal == b->env.goal);
    assert(a->env.poison == b->env.poison && a->env.enemy == b->env.enemy);
    assert(a->env.floor_seed == b->env.floor_seed);
#endif
}

void check_observation(Fixture* f) {
    assert(f->buffer[0] == 12345 && f->buffer[f->size + 1] == -12345);
    float* saved = malloc(f->size * sizeof(float));
    assert(saved);
    memcpy(saved, f->buffer + 1, f->size * sizeof(float));
    unsigned int rng = f->env.rng;
#if defined(SUITE_CATCH) || defined(SUITE_RUNNER)
    unsigned int object_rng = f->env.object_rng;
#endif
    compute_observations(&f->env);
    assert(f->env.rng == rng);
#if defined(SUITE_CATCH) || defined(SUITE_RUNNER)
    assert(f->env.object_rng == object_rng);
#endif
    assert(!memcmp(saved, f->buffer + 1, f->size * sizeof(float)));
    if (f->env.observation_mode) {
        float* full = malloc(PIXEL_OBS_SIZE * sizeof(float));
        assert(full);
        sprite_observe(full, f->env.grid, f->env.player, f->env.representation,
            f->env.floor_seed, NULL);
        assert(!memcmp(saved, full, PIXEL_OBS_SIZE * sizeof(float)));
        free(full);
    }
    for (int i = 0; i < f->size; i++) {
        assert(isfinite(saved[i]) && saved[i] >= -1 && saved[i] <= 1);
    }
    if (!f->env.observation_mode) {
        assert(saved[0] == (f->env.player % GRID_WIDTH) / (float)GRID_WIDTH);
#ifdef SUITE_CATCH
        assert(saved[1] == (f->env.object % GRID_WIDTH) / (float)GRID_WIDTH);
        assert(saved[3] == f->env.good);
#else
        assert(saved[1] == (f->env.player / GRID_WIDTH) / (float)GRID_HEIGHT);
#endif
    }
    free(saved);
}

int oracle(Fixture* f) {
#ifdef SUITE_CATCH
    int dx = f->env.object % GRID_WIDTH - f->env.player % GRID_WIDTH;
    if (f->env.good) {
        return dx < 0 ? 1 : dx > 0 ? 2 : 0;
    }
    return dx == 0 ? (f->env.player % GRID_WIDTH ? 1 : 2) : 0;
#elif defined(SUITE_RUNNER)
    int y = f->env.object / GRID_WIDTH;
    int jump = f->env.kind > RUNNER_BERRY || y < 11;
    return jump && f->env.object % GRID_WIDTH == (y == 9 ? 4 : 5);
#elif defined(SUITE_PUSH)
    int cx = f->env.crate % GRID_WIDTH, cy = f->env.crate / GRID_WIDTH;
    int gx = f->env.goal % GRID_WIDTH, gy = f->env.goal / GRID_WIDTH;
    int dx = gx > cx ? 1 : gx < cx ? -1 : 0;
    int dy = gy > cy ? 1 : gy < cy ? -1 : 0;
    int behind = f->env.crate - dx - dy * GRID_WIDTH;
    int delta[] = {0, -GRID_WIDTH, GRID_WIDTH, -1, 1};
    if (f->env.player == behind) {
        return dy < 0 ? 1 : dy > 0 ? 2 : dx < 0 ? 3 : 4;
    }
    int queue[GRID_WIDTH * GRID_HEIGHT], first[GRID_WIDTH * GRID_HEIGHT];
    unsigned char visited[GRID_WIDTH * GRID_HEIGHT] = {0};
    int head = 0, tail = 1;
    queue[0] = f->env.player;
    first[0] = 0;
    visited[f->env.player] = 1;
    while (head < tail) {
        int slot = head++, cell = queue[slot];
        for (int a = 1; a < 5; a++) {
            int next = cell + delta[a];
            int kind = f->env.grid[next];
            if (visited[next] || kind == PUSH_WALL || kind == PUSH_POISON
                || kind == PUSH_ENEMY || next == f->env.crate) {
                continue;
            }
            int action = slot == 0 ? a : first[slot];
            if (next == behind) {
                return action;
            }
            visited[next] = 1;
            queue[tail] = next;
            first[tail++] = action;
        }
    }
    assert(0);
    return 0;
#elif defined(SUITE_KEY)
    int delta[] = {0, -GRID_WIDTH, GRID_WIDTH, -1, 1};
    short* distance = f->env.has_key ? f->env.goal_distance : f->env.key_distance;
    for (int a = 1; a < 5; a++) {
        int cell = f->env.player + delta[a];
        int kind = f->env.grid[cell];
        if (kind != KEY_WALL && kind != KEY_DOOR && distance[cell] >= 0
            && distance[cell] < distance[f->env.player]) {
            return a;
        }
    }
    assert(0);
    return 0;
#elif defined(SUITE_MAZE)
    int delta[] = {0, -GRID_WIDTH, GRID_WIDTH, -1, 1};
    for (int a = 1; a < 5; a++) {
        int cell = f->env.player + delta[a];
        if (f->env.grid[cell] != MAZE_HEDGE && f->env.distance[cell] >= 0
            && f->env.distance[cell] < f->env.distance[f->env.player]) {
            return a;
        }
    }
    assert(0);
    return 0;
#else
    // Search the actual transition function; validates gap solvability.
    struct Node {int player, vy, first;};
    struct Node queue[GRID_WIDTH * GRID_HEIGHT * 6];
    unsigned char visited[GRID_WIDTH * GRID_HEIGHT * 6] = {0};
    int head = 0, tail = 1;
    queue[0] = (struct Node){f->env.player, f->env.vy, 0};
    visited[6 * f->env.player + f->env.vy + 2] = 1;
    while (head < tail) {
        struct Node node = queue[head++];
        for (int a = 0; a < 5; a++) {
            Env trial = f->env;
            float action = a, reward = 0, terminal = 0, obs[STATE_OBS_SIZE];
            trial.player = node.player;
            trial.vy = node.vy;
            trial.tick = 0;
            trial.log = (Log){0};
            trial.agents[0].actions = &action;
            trial.agents[0].rewards = &reward;
            trial.agents[0].terminals = &terminal;
            trial.agents[0].observations = obs;
            puf_step(&trial);
            int first = head == 1 ? a : node.first;
            if (trial.log.successes) {
                return first;
            }
            int key = 6 * trial.player + trial.vy + 2;
            assert(key >= 0 && key < GRID_WIDTH * GRID_HEIGHT * 6);
            if (!terminal && !visited[key]) {
                visited[key] = 1;
                assert(tail < GRID_WIDTH * GRID_HEIGHT * 6);
                queue[tail++] = (struct Node){trial.player, trial.vy, first};
            }
        }
    }
    assert(0 && "Unsolvable generated platform");
    return 0;
#endif
}

int main(void) {
    for (int kind = 0; kind < SPRITE_TILE_TYPES; kind++) {
        for (int y = 0; y < 8; y++) {
            assert(strlen(SPRITE_ART[kind][y]) == 8);
            for (int x = 0; x < 8; x++) {
                assert(SPRITE_ART[kind][y][x] >= '0' && SPRITE_ART[kind][y][x] <= '7');
            }
        }
    }
    for (int rep = 0; rep < 3; rep++) {
        sprite_make_tiles(rep);
        for (int texture = 0; texture < SPRITE_TEXTURES; texture++) {
            float* tile = sprite_tiles[rep][0][texture];
            int varied = 0;
            for (int i = 1; i < SPRITE_CELL * SPRITE_CELL; i++) {
                varied |= memcmp(tile, tile + 3 * i, 3 * sizeof(float)) != 0;
            }
            assert(varied && "Background must not be a solid color");
            if (texture) {
                assert(memcmp(tile, sprite_tiles[rep][0][0],
                    SPRITE_CELL * SPRITE_CELL * 3 * sizeof(float)));
            }
#if defined(SUITE_KEY) || defined(SUITE_PUSH)
#ifdef SUITE_KEY
            int good = KEY_CHEST, bad = KEY_MIMIC;
#else
            int good = PUSH_CRATE, bad = PUSH_POISON;
#endif
            int changed = 0;
            for (int i = 0; i < SPRITE_CELL * SPRITE_CELL; i++) {
                changed += memcmp(sprite_tiles[rep][good][texture] + 3 * i,
                    sprite_tiles[rep][bad][texture] + 3 * i, 3 * sizeof(float)) != 0;
            }
            assert(changed == 32 && "Exactly two source pixels differ, enlarged4x");
#endif
        }
    }
    unsigned char empty[GRID_WIDTH * GRID_HEIGHT] = {0};
    float* background_a = calloc(PIXEL_OBS_SIZE, sizeof(float));
    float* background_b = calloc(PIXEL_OBS_SIZE, sizeof(float));
    sprite_observe(background_a, empty, -1, 0, 7, NULL);
    sprite_observe(background_b, empty, -1, 0, 42, NULL);
    assert(memcmp(background_a, background_b, PIXEL_OBS_SIZE * sizeof(float)));
    sprite_observe(background_b, empty, -1, 0, 7, NULL);
    assert(!memcmp(background_a, background_b, PIXEL_OBS_SIZE * sizeof(float)));
    free(background_a);
    free(background_b);
    const int actions[] = ACT_SIZES;
    unsigned int seeds[] = {7, 42, 69};
    for (int seed = 0; seed < 3; seed++) {
        Fixture cases[6];
        for (int i = 0; i < 6; i++) {
            setup(&cases[i], i / 3, i % 3, seeds[seed]);
            same_world(&cases[0], &cases[i]);
            check_observation(&cases[i]);
        }
        unsigned int sequence = 420;
        for (int step = 0; step < 1024; step++) {
            sequence = sequence * 1664525u + 1013904223u;
            for (int i = 0; i < 6; i++) {
                cases[i].action = (sequence >> 16) % actions[0];
                cases[i].reward = cases[i].terminal = 0;
                puf_step(&cases[i].env);
                same_world(&cases[0], &cases[i]);
                if (step % 64 == 0 || cases[i].terminal) {
                    check_observation(&cases[i]);
                }
            }
        }
        for (int i = 0; i < 6; i++) {
            puf_close(&cases[i].env);
            free(cases[i].buffer);
        }
    }
    Fixture a, b;
    setup(&a, 0, 0, 7);
    setup(&b, 0, 2, 7);
#ifdef SUITE_PLATFORM
    unsigned int ground_heights = 0, step_heights = 0;
#endif
#ifdef SUITE_MAZE
    unsigned int trap_columns = 0;
#endif
    // Different actions and episode lengths must not alter assigned next worlds.
    for (int episode = 0; episode < 100; episode++) {
        same_world(&a, &b);
#ifdef SUITE_MAZE
        int delta[] = {-GRID_WIDTH, GRID_WIDTH, -1, 1}, traps = 0;
        for (int cell = GRID_WIDTH + 1; cell < (GRID_HEIGHT - 1) * GRID_WIDTH; cell++) {
            if (a.env.grid[cell] != MAZE_TRAP) {
                continue;
            }
            traps++;
            trap_columns |= 1u << (cell % GRID_WIDTH);
            int exits = 0;
            for (int i = 0; i < 4; i++) {
                exits += a.env.grid[cell + delta[i]] != MAZE_HEDGE;
            }
            assert(exits == 1 && a.env.distance[cell] == -1);
            assert(cell != a.env.player && cell != a.env.goal);
        }
        assert(traps == 3 && a.env.distance[a.env.player] >= 0);
#endif
#ifdef SUITE_PLATFORM
        int start_y = a.env.player / GRID_WIDTH, end_y = a.env.goal / GRID_WIDTH;
        assert(start_y >= 9 && start_y <= 12);
        assert(start_y - end_y >= 1 && start_y - end_y <= 2);
        ground_heights |= 1u << start_y;
        step_heights |= 1u << (start_y - end_y);
#endif
#if defined(SUITE_KEY) || defined(SUITE_PUSH)
        int floors[3] = {0};
        for (int cell = 0; cell < GRID_WIDTH * GRID_HEIGHT; cell++) {
            if (a.env.grid[cell] < 3) {
                floors[a.env.grid[cell]]++;
            }
        }
        assert(floors[0] && floors[1] && floors[2]);
#endif
        do {
            a.action = oracle(&a);
            a.terminal = a.reward = 0;
            puf_step(&a.env);
        } while (!a.terminal);
        assert(a.env.log.perf == episode + 1);
        int steps = 0;
        do {
            b.action = 0;
            b.terminal = b.reward = 0;
            puf_step(&b.env);
            steps++;
            assert(steps <= 512);
        } while (!b.terminal);
        // Logs/rewards differ, but each next episode gets the same initial world.
        b.env.log = a.env.log;
        b.reward = a.reward;
        b.terminal = a.terminal;
    }
#ifdef SUITE_PLATFORM
    assert(ground_heights == (15u << 9) && step_heights == 6);
#endif
#ifdef SUITE_MAZE
    assert((trap_columns & 0x3ffu) && (trap_columns >> 10));
    assert(trap_columns & ~(1u << 18));
#endif
    free(a.buffer);
    free(b.buffer);
    // Exact terminal/autoreset and obstacle cases, independent of the planner.
    Fixture f;
    setup(&f, 0, 0, 7);
#if defined(SUITE_CATCH)
    for (int good = 0; good < 2; good++) {
        puf_reset(&f.env);
        f.env.grid[f.env.object] = CATCH_EMPTY;
        f.env.object = GRID_WIDTH + 10;
        f.env.grid[f.env.object] = good ? CATCH_GEM : CATCH_SKULL;
        f.env.good = good;
        f.action = 2;
        puf_step(&f.env);
        float first = f.reward;
        assert(first == (good ? -.02f : .02f));
        f.action = 1;
        puf_step(&f.env);
        assert(first + f.reward == 0);
    }
    for (int good = 0; good < 2; good++) {
        for (int caught = 0; caught < 2; caught++) {
            puf_reset(&f.env);
            f.env.log = (Log){0};
            f.env.drops = 1;
            f.env.grid[f.env.object] = CATCH_EMPTY;
            f.env.object = 12 * GRID_WIDTH + (caught ? 10 : 12);
            f.env.grid[f.env.object] = good ? CATCH_GEM : CATCH_SKULL;
            f.env.good = good;
            f.env.tick = 1;
            f.action = 0;
            puf_step(&f.env);
            assert(f.terminal == 1 && f.env.log.n == 1);
            assert(f.env.log.perf == (caught == good));
            assert(f.reward == (caught == good ? 1 : -1));
            assert(f.env.log.deaths == (caught && !good));
        }
    }
    puf_reset(&f.env);
    f.env.log = (Log){0};
    f.env.drops = 10;
    f.env.count = f.env.correct = 2;
    f.env.grid[f.env.object] = CATCH_EMPTY;
    f.env.object = 12 * GRID_WIDTH + 10;
    f.env.grid[f.env.object] = CATCH_SKULL;
    f.env.good = 0;
    f.env.tick = 1;
    f.action = 0;
    puf_step(&f.env);
    assert(f.terminal == 1 && f.env.log.deaths == 1 && f.env.log.successes == 0);
    assert(f.env.log.perf == 2.f / 10 && f.env.log.episode_length == 2);
#elif defined(SUITE_RUNNER)
    for (int kind = RUNNER_STAR; kind <= RUNNER_ROCK; kind++) {
        for (int touched = 0; touched < 2; touched++) {
            puf_reset(&f.env);
            f.env.log = (Log){0};
            f.env.items = 1;
            f.env.grid[f.env.object] = RUNNER_EMPTY;
            f.env.object = 11 * GRID_WIDTH + 4;
            f.env.kind = kind;
            f.env.grid[f.env.object] = kind;
            f.env.player = (touched ? 11 : 8) * GRID_WIDTH + 3;
            f.action = 0;
            puf_step(&f.env);
            int correct = touched == (kind <= RUNNER_BERRY);
            assert(f.terminal == 1 && f.env.log.n == 1 && f.env.log.perf == correct);
            assert(f.reward == (correct ? 1 : -1));
            assert(f.env.log.deaths == (touched && kind > RUNNER_BERRY));
        }
    }
    puf_reset(&f.env);
    f.env.log = (Log){0};
    f.env.items = 12;
    f.env.count = f.env.correct = 2;
    f.env.grid[f.env.object] = RUNNER_EMPTY;
    f.env.object = 11 * GRID_WIDTH + 4;
    f.env.kind = RUNNER_CACTUS;
    f.env.grid[f.env.object] = RUNNER_CACTUS;
    f.action = 0;
    puf_step(&f.env);
    assert(f.terminal == 1 && f.env.log.deaths == 1 && f.env.log.successes == 0);
    assert(f.env.log.perf == 2.f / 12 && f.env.log.episode_length == 1);
    // A ground-level policy misses an airborne reward; jumping collects it.
    for (int jump = 0; jump < 2; jump++) {
        puf_reset(&f.env);
        f.env.log = (Log){0};
        f.env.items = 1;
        f.env.grid[f.env.object] = RUNNER_EMPTY;
        f.env.object = 9 * GRID_WIDTH + 4;
        f.env.kind = RUNNER_STAR;
        f.env.grid[f.env.object] = RUNNER_STAR;
        f.action = jump;
        puf_step(&f.env);
        assert(f.terminal == 1 && f.env.log.perf == jump);
    }
#elif defined(SUITE_KEY)
    f.env.player = f.env.door - 1;
    f.action = 4;
    int blocked = f.env.player;
    puf_step(&f.env);
    assert(f.env.player == blocked && !f.env.has_key);
    f.env.player = f.env.key - 1;
    puf_step(&f.env);
    assert(f.env.has_key && f.env.grid[f.env.key] == key_floor(&f.env, f.env.key));
    assert(f.env.grid[f.env.door] == key_floor(&f.env, f.env.door));
    f.env.player = f.env.goal - 1;
    puf_step(&f.env);
    assert(f.terminal == 1 && f.env.log.perf == 1 && f.reward > .9f);
    f.env.log = (Log){0};
    f.env.player = f.env.mimic - 1;
    f.action = 4;
    puf_step(&f.env);
    assert(f.terminal == 1 && f.env.log.perf == 0 && f.reward == -1);
    assert(f.env.log.deaths == 1);
    int differences = 0;
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            differences += SPRITE_ART[KEY_CHEST][y][x] != SPRITE_ART[KEY_MIMIC][y][x];
        }
    }
    assert(differences == 2);
#elif defined(SUITE_PUSH)
    f.env.grid[f.env.crate] = PUSH_EMPTY;
    f.env.crate = GRID_WIDTH + 1;
    f.env.grid[f.env.crate] = PUSH_CRATE;
    f.env.player = GRID_WIDTH + 2;
    f.action = 3;
    puf_step(&f.env);
    assert(f.env.player == GRID_WIDTH + 2 && f.env.crate == GRID_WIDTH + 1);
    puf_reset(&f.env);
    f.env.log = (Log){0};
    f.env.player = f.env.poison - 1;
    f.action = 4;
    puf_step(&f.env);
    assert(f.terminal == 1 && f.env.log.perf == 0 && f.reward < -.9f);
    assert(f.env.log.deaths == 1);
    // Verify the generated trap corner really strands a sideways-pushed crate.
    int forward = (f.env.goal - f.env.crate) / 3;
    int side = abs(forward) == 1 ? forward * GRID_WIDTH : forward / GRID_WIDTH;
    int crate = f.env.crate;
    f.env.player = crate - side;
    f.action = side == 1 ? 4 : side == -1 ? 3 : side == GRID_WIDTH ? 2 : 1;
    puf_step(&f.env);
    assert(f.env.crate == crate + side);
    assert(f.env.grid[f.env.crate + side] == PUSH_WALL);
    assert(f.env.grid[f.env.crate - forward] == PUSH_WALL);
    int delta[] = {-GRID_WIDTH, GRID_WIDTH, -1, 1};
    for (int i = 0; i < 4; i++) {
        assert(f.env.grid[f.env.crate + delta[i]] == PUSH_WALL
            || f.env.grid[f.env.crate - delta[i]] == PUSH_WALL);
    }
    // Clear room: pursue on turns2/4/6/8, never on an odd turn.
    puf_reset(&f.env);
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            f.env.grid[y * GRID_WIDTH + x] = !x || !y
                || x == GRID_WIDTH - 1 || y == GRID_HEIGHT - 1 ? PUSH_WALL : PUSH_EMPTY;
        }
    }
    f.env.player = 7 * GRID_WIDTH + 7;
    f.env.enemy = f.env.player + 4;
    f.env.crate = 10 * GRID_WIDTH + 9;
    f.env.goal = 10 * GRID_WIDTH + 12;
    f.env.poison = 12 * GRID_WIDTH + 12;
    f.env.grid[f.env.enemy] = PUSH_ENEMY;
    f.env.grid[f.env.crate] = PUSH_CRATE;
    f.env.grid[f.env.goal] = PUSH_GOAL;
    f.env.grid[f.env.poison] = PUSH_POISON;
    f.env.log = (Log){0};
    Env room = f.env;
    f.action = 0;
    int enemy = f.env.enemy;
    for (int turn = 1; turn < 8; turn++) {
        puf_step(&f.env);
        assert(!f.terminal && f.env.enemy == enemy - turn / 2);
    }
    puf_step(&f.env);
    assert(f.terminal && f.env.log.perf == 0 && f.reward < -.9f);
    f.env = room;
    f.env.grid[f.env.enemy] = PUSH_EMPTY;
    f.env.enemy = f.env.player + 1;
    f.env.grid[f.env.enemy] = PUSH_ENEMY;
    f.action = 4;
    puf_step(&f.env);
    assert(f.terminal && f.env.log.perf == 0 && f.reward < -.9f);
    // Block the preferred horizontal axis: the enemy takes the open vertical one.
    f.env = room;
    f.env.grid[f.env.enemy] = PUSH_EMPTY;
    f.env.enemy = 9 * GRID_WIDTH + 11;
    f.env.grid[f.env.enemy] = PUSH_ENEMY;
    f.env.grid[f.env.enemy - 1] = PUSH_WALL;
    f.env.tick = 1;
    f.action = 0;
    enemy = f.env.enemy;
    puf_step(&f.env);
    assert(!f.terminal && f.env.enemy == enemy - GRID_WIDTH);
    f.env = room;
    f.env.grid[f.env.enemy] = PUSH_EMPTY;
    f.env.enemy = f.env.goal;
    f.env.grid[f.env.enemy] = PUSH_ENEMY;
    f.env.tick = 1;
    puf_step(&f.env);
    assert(!f.terminal && f.env.grid[f.env.goal] == PUSH_GOAL);
    int differences = 0;
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            differences += SPRITE_ART[PUSH_CRATE][y][x] != SPRITE_ART[PUSH_POISON][y][x];
        }
    }
    assert(differences == 2);
#elif defined(SUITE_MAZE)
    int delta[] = {0, -GRID_WIDTH, GRID_WIDTH, -1, 1};
    for (int a = 1; a < 5; a++) {
        int cell = f.env.goal - delta[a];
        if (f.env.distance[cell] == 1) {
            f.env.player = cell;
            f.action = a;
            break;
        }
    }
    puf_step(&f.env);
    assert(f.terminal == 1 && f.env.log.perf == 1 && f.reward > .9f);
    int traps = 0, dead_ends = 0;
    for (int cell = GRID_WIDTH + 1; cell < (GRID_HEIGHT - 1) * GRID_WIDTH; cell++) {
        if (f.env.grid[cell] == MAZE_HEDGE) {
            continue;
        }
        traps += f.env.grid[cell] == MAZE_TRAP;
        int exits = 0;
        for (int a = 1; a < 5; a++) {
            exits += f.env.grid[cell + delta[a]] != MAZE_HEDGE;
        }
        dead_ends += exits == 1;
    }
    assert(traps == 3 && dead_ends >= 7);
    for (int cell = 0; cell < GRID_WIDTH * GRID_HEIGHT; cell++) {
        if (f.env.grid[cell] == MAZE_TRAP) {
            for (int a = 1; a < 5; a++) {
                int previous = cell - delta[a];
                if (f.env.grid[previous] != MAZE_HEDGE) {
                    f.env.player = previous;
                    f.action = a;
                    break;
                }
            }
            break;
        }
    }
    f.env.log = (Log){0};
    puf_step(&f.env);
    assert(f.terminal && f.env.log.perf == 0 && f.reward == -1);
    assert(f.env.log.deaths == 1);
#else
    int standing = f.env.player;
    f.env.vy = 3;
    f.action = 0;
    puf_step(&f.env);
    assert(f.env.player == standing);
    int gap = 0;
    while (f.env.grid[(GRID_HEIGHT - 1) * GRID_WIDTH + gap] != PLATFORM_EMPTY) {
        gap++;
    }
    f.env.player = 13 * GRID_WIDTH + gap;
    f.env.vy = 3;
    puf_step(&f.env);
    assert(f.terminal == 1 && f.env.log.perf == 0 && f.reward < -.9f);
    assert(f.env.log.deaths == 1);
#endif
    free(f.buffer);
    puts("PASS: 18432 paired mode/skin steps; dirty/full pixels; guards; "
        "100 solvable assigned episodes across different policies; original artwork");
}
