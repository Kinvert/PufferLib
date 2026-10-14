// Native sprite-recognition foraging. Original art, direct-memory RGB pixels.
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
typedef float obs_t;
#include "pufferenv.h"
#include "../connect4cnn/appearance.h"

#ifndef CNN_IMAGE_V1
#define CNN_IMAGE_V1
#define CNN_IMAGE_HEIGHT 480
#define CNN_IMAGE_WIDTH 640
#define CNN_IMAGE_CHANNELS 3
#endif
#include "../connect4cnn/image.h"
#if OBS_HEIGHT != 480 || OBS_WIDTH != 640 || OBS_CHANNELS != 3
#error "forage-sprites-v1 requires 640x480 RGB; version a different input explicitly"
#endif
#include "sprites.h"
#include "imported_sprites.h"
#define ACT_SIZES {5}
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 8
#define FORAGE_WIDTH 20
#define FORAGE_HEIGHT 15
#define FORAGE_AREA (FORAGE_WIDTH * FORAGE_HEIGHT)
#define FORAGE_CELL 32
#if FORAGE_IMPORTED_CELL_PIXELS != FORAGE_CELL
#error "Rebuild the imported atlas for the native cell size"
#endif
#define FORAGE_REPRESENTATIONS 5
#define FORAGE_RULES "forage-sprites-v1"

struct Log {
    float perf, score, episode_return, episode_length, positive_collected;
    float negative_collected, deaths, successes, n;
};
struct Env {
    Log log;
    Agent agents[1];
    int num_agents, tag, boundary_reached;
    unsigned int rng;
    unsigned char grid[FORAGE_AREA];
    int player, tick, representation, copies, trees, max_steps, remaining;
    int positives, negatives, last_end_reason, last_decisions;
    float episode_return, positive_value, last_return, step_cost;
    Texture2D texture;
    Color* display;
    int viewer_open;
};

// Shared immutable precomposed tiles, initialized once even under vector threads.
static obs_t forage_tiles[FORAGE_REPRESENTATIONS][FORAGE_TILE_TYPES][FORAGE_CELL * FORAGE_CELL * 3];
static pthread_once_t forage_tiles_once = PTHREAD_ONCE_INIT;

static void forage_require(bool condition, const char* message) {
    if (!condition) { fprintf(stderr, "ForageCNN: %s\n", message); exit(1); }
}
static int forage_int(Dict* kwargs, const char* key, int low, int high) {
    DictItem* value = dict_find(kwargs, key);
    forage_require(value && isfinite(value->value) && value->value == floor(value->value)
        && value->value >= low && value->value <= high, key);
    return (int)value->value;
}
static unsigned int forage_draw(Env* env, unsigned int count) {
    unsigned int value, threshold = (0u - count) % count;
    do { env->rng = env->rng * 1664525u + 1013904223u; value = env->rng; } while (value < threshold);
    return value % count;
}
static int forage_empty(Env* env) {
    int free_cells[FORAGE_AREA], count = 0;
    for (int i = 0; i < FORAGE_AREA; i++) if (!env->grid[i] && i != env->player) free_cells[count++] = i;
    forage_require(count > 0, "no free spawn cell");
    return free_cells[forage_draw(env, (unsigned int)count)];
}
static Color forage_color(unsigned char glyph, int theme) {
    Color color;
    switch (glyph) {
        case 'k': color = (Color){16, 20, 31, 255}; break;
        case 'r': color = (Color){203, 53, 65, 255}; break;
        case 'o': color = (Color){246, 148, 77, 255}; break;
        case 'y': color = (Color){242, 207, 78, 255}; break;
        case 'g': color = (Color){91, 165, 79, 255}; break;
        case 'G': color = (Color){43, 103, 65, 255}; break;
        case 'p': color = (Color){155, 94, 178, 255}; break;
        case 'b': color = (Color){77, 137, 192, 255}; break;
        case 'w': color = (Color){240, 226, 191, 255}; break;
        case 's': color = (Color){125, 83, 58, 255}; break;
        case 'n': color = (Color){226, 171, 128, 255}; break;
        default: color = (Color){30, 49, 38, 255}; break;
    }
    if (theme) { unsigned char red = color.r; color.r = color.b; color.b = red; }
    return color;
}
static void forage_make_tiles(void) {
    for (int rep = 0; rep < FORAGE_REPRESENTATIONS; rep++) for (int kind = 0; kind < FORAGE_TILE_TYPES; kind++) {
        unsigned char sprite[256]; forage_sprite(kind, sprite);
        for (int y = 0; y < FORAGE_CELL; y++) for (int x = 0; x < FORAGE_CELL; x++) {
            int sx = rep == 2 ? 15 - x / 2 : x / 2;
            unsigned char glyph = sprite[(y / 2) * 16 + sx];
            Color color = forage_color(glyph, rep == 1);
            if (rep == 4 && kind != FORAGE_EMPTY) {
                const unsigned char* source = FORAGE_IMPORTED[kind] + (y * FORAGE_CELL + x) * 4;
                glyph = source[3] ? '!' : '.';
                color = (Color){source[0], source[1], source[2], source[3]};
            }
            if (glyph == '.') {
                color = forage_color('.', rep == 1);
                int blade = rep == 3 ? ((x / 2 + 3 * (y / 2)) % 7 == 0 || (x + y) % 11 == 0)
                    : ((x == 4 || x == 24) && y % 16 >= 9 && y % 16 <= 12);
                if (blade) color = forage_color('G', rep == 1);
            }
            int at = (y * FORAGE_CELL + x) * 3;
            forage_tiles[rep][kind][at] = color.r / 255.f;
            forage_tiles[rep][kind][at + 1] = color.g / 255.f;
            forage_tiles[rep][kind][at + 2] = color.b / 255.f;
        }
    }
}
static void forage_cell(Env* env, int cell) {
    int kind = cell == env->player ? FORAGE_PLAYER : env->grid[cell];
    int left = (cell % FORAGE_WIDTH) * FORAGE_CELL, top = (cell / FORAGE_WIDTH) * FORAGE_CELL;
    const obs_t* tile = forage_tiles[env->representation][kind];
    for (int y = 0; y < FORAGE_CELL; y++) memcpy(env->agents[0].observations + ((top + y) * OBS_WIDTH + left) * 3,
        tile + y * FORAGE_CELL * 3, FORAGE_CELL * 3 * sizeof(obs_t));
}
void compute_observations(Env* env) {
    for (int cell = 0; cell < FORAGE_AREA; cell++) forage_cell(env, cell);
}
static bool forage_connected(Env* env) {
    unsigned char visited[FORAGE_AREA] = {0};
    int queue[FORAGE_AREA], head = 0, tail = 1, free_cells = 0;
    int delta[] = {-FORAGE_WIDTH, FORAGE_WIDTH, -1, 1};
    queue[0] = env->player; visited[env->player] = 1;
    for (int i = 0; i < FORAGE_AREA; i++) free_cells += env->grid[i] < FORAGE_TOADSTOOL;
    while (head < tail) for (int a = 0, cell = queue[head++]; a < 4; a++) {
        int next = cell + delta[a];
        if (next < 0 || next >= FORAGE_AREA || visited[next] || env->grid[next] >= FORAGE_TOADSTOOL) continue;
        visited[next] = 1; queue[tail++] = next;
    }
    return tail == free_cells;
}
void puf_init(Env* env, Dict* kwargs) {
    forage_int(kwargs, "rule_version", 1, 1);
    forage_int(kwargs, "num_agents", 1, 1);
    env->copies = forage_int(kwargs, "copies_per_type", 1, 8);
    env->trees = forage_int(kwargs, "trees", 0, 48);
    env->max_steps = forage_int(kwargs, "max_steps", 1, 65536);
    DictItem* step_cost = dict_find(kwargs, "step_cost");
    forage_require(step_cost && isfinite(step_cost->value) && step_cost->value >= -.1 && step_cost->value <= 0, "step_cost");
    env->step_cost = (float)step_cost->value;
    env->representation = cnn_appearance_init(kwargs, env->rng, FORAGE_REPRESENTATIONS);
    env->num_agents = 1; env->agents[0].policy = 0; env->agents[0].action_mask = NULL;
    pthread_once(&forage_tiles_once, forage_make_tiles);
}
void puf_reset(Env* env) {
    memset(env->grid, 0, sizeof(env->grid)); env->player = -1;
    for (int y = 0; y < FORAGE_HEIGHT; y++) for (int x = 0; x < FORAGE_WIDTH; x++)
        if (!x || !y || x == FORAGE_WIDTH - 1 || y == FORAGE_HEIGHT - 1) env->grid[y * FORAGE_WIDTH + x] = FORAGE_TREE;
    env->player = forage_empty(env);
    for (int n = 0; n < env->trees; n++) {
        bool accepted = false;
        for (int attempt = 0; attempt < 1000; attempt++) {
            int cell = forage_empty(env); env->grid[cell] = FORAGE_TREE;
            if (forage_connected(env)) { accepted = true; break; }
            env->grid[cell] = FORAGE_EMPTY;
        }
        forage_require(accepted, "could not place connected trees");
    }
    for (int kind = 1; kind <= 10; kind++) for (int n = 0; n < env->copies; n++) {
        bool accepted = false;
        for (int attempt = 0; attempt < 1000; attempt++) {
            int cell = forage_empty(env); env->grid[cell] = (unsigned char)kind;
            if (kind != FORAGE_TOADSTOOL || forage_connected(env)) { accepted = true; break; }
            env->grid[cell] = FORAGE_EMPTY;
        }
        forage_require(accepted, "could not place reachable objects");
    }
    env->tick = env->positives = env->negatives = 0; env->remaining = 7 * env->copies;
    env->episode_return = env->positive_value = 0;
    env->last_end_reason = env->last_decisions = 0; env->last_return = 0;
    env->agents[0].rewards[0] = env->agents[0].terminals[0] = 0;
    compute_observations(env);
}
static void forage_end(Env* env, int reason) {
    float reward = env->agents[0].rewards[0], total = env->episode_return;
    int decisions = env->tick;
    env->log.perf += env->positive_value / (1.45f * env->copies);
    env->log.score += total; env->log.episode_return += total; env->log.episode_length += decisions;
    env->log.positive_collected += env->positives; env->log.negative_collected += env->negatives;
    env->log.deaths += reason == 1; env->log.successes += reason == 2; env->log.n++;
    puf_reset(env);
    env->last_end_reason = reason; env->last_decisions = decisions; env->last_return = total;
    env->agents[0].rewards[0] = reward; env->agents[0].terminals[0] = 1;
}
void puf_step(Env* env) {
    float action = env->agents[0].actions[0];
    forage_require(isfinite(action) && action == floorf(action) && action >= 0 && action <= 4, "action must be integer0..4");
    env->agents[0].rewards[0] = env->step_cost; env->agents[0].terminals[0] = 0;
    env->last_end_reason = 0; env->tick++;
    int delta[] = {0, -FORAGE_WIDTH, FORAGE_WIDTH, -1, 1};
    int previous = env->player, next = previous + delta[(int)action], kind = FORAGE_EMPTY;
    if (env->grid[next] != FORAGE_TREE) {
        env->player = next; kind = env->grid[next];
        if (kind) {
            env->agents[0].rewards[0] += FORAGE_REWARDS[kind]; env->grid[next] = FORAGE_EMPTY;
            if (kind <= FORAGE_MUSHROOM) { env->remaining--; env->positives++; env->positive_value += FORAGE_REWARDS[kind]; }
            else env->negatives++;
        }
    }
    env->episode_return += env->agents[0].rewards[0];
    int reason = kind == FORAGE_TOADSTOOL ? 1 : !env->remaining ? 2 : env->tick >= env->max_steps ? 3 : 0;
    if (reason) { forage_end(env, reason); return; }
    if (env->player != previous) { forage_cell(env, previous); forage_cell(env, env->player); }
}
void puf_render(Env* env) {
    if (!env->viewer_open) {
        InitWindow(OBS_WIDTH, OBS_HEIGHT, "PufferLib ForageCNN - policy pixels"); SetTargetFPS(PUF_STEPS_PER_SEC);
        env->display = (Color*)calloc(OBS_WIDTH * OBS_HEIGHT, sizeof(Color));
        forage_require(env->display != NULL, "viewer allocation failed");
        Image image = {env->display, OBS_WIDTH, OBS_HEIGHT, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
        env->texture = LoadTextureFromImage(image); env->viewer_open = 1;
    }
    for (int i = 0; i < OBS_WIDTH * OBS_HEIGHT; i++) env->display[i] = (Color){
        (unsigned char)lroundf(env->agents[0].observations[i * 3] * 255),
        (unsigned char)lroundf(env->agents[0].observations[i * 3 + 1] * 255),
        (unsigned char)lroundf(env->agents[0].observations[i * 3 + 2] * 255), 255};
    UpdateTexture(env->texture, env->display);
    BeginDrawing(); ClearBackground(BLACK); DrawTexture(env->texture, 0, 0, WHITE); EndDrawing();
}
void puf_close(Env* env) {
    if (env->viewer_open) { UnloadTexture(env->texture); CloseWindow(); free(env->display); env->viewer_open = 0; }
}
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf); dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return); dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "positive_collected", log->positive_collected); dict_set(out, "negative_collected", log->negative_collected);
    dict_set(out, "deaths", log->deaths); dict_set(out, "successes", log->successes); dict_set(out, "n", log->n);
}
