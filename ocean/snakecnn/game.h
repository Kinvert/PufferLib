// snake-local-episodic-v1: one agent, portable local RNG, clean finite episodes.
// Movement derives from ocean/snake/snake.h; stock Snake remains untouched.
#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include "../connect4cnn/appearance.h"

#define ACT_SIZES {4}
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 10
#define SNAKEBENCH_VISION 5
#define SNAKEBENCH_WINDOW 11
#define SNAKECNN_NUM_REPRESENTATIONS 6
#define SNAKEBENCH_RULES "snake-local-episodic-v1"
#if SNAKEBENCH_PIXELS
#define OBS_CHANNELS 1
#define OBS_HEIGHT 36
#define OBS_WIDTH 44
#define OBS_SIZE (OBS_HEIGHT * OBS_WIDTH)
#else
#define OBS_SIZE (SNAKEBENCH_WINDOW * SNAKEBENCH_WINDOW * 8)
#endif

#define SNAKE_EMPTY 0
#define SNAKE_FOOD 1
#define SNAKE_WALL 3
#define SNAKE_BODY 7

struct Log {
    float perf;
    float score;
    float food_collected;
    float episode_return;
    float episode_length;
    float deaths;
    float horizon_ends;
    float n;
};
struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
    unsigned char* grid;
    int* snake;
    int width, height, food, max_snake_length, max_steps, cell_size;
    int length, head_ptr, tick, food_collected;
    float reward_food, reward_death, episode_return;
    int last_score, last_food, last_decisions, last_end_reason;
    float last_return;
    int representation;
    int viewer_open;
};

static void snakebench_require(bool condition, const char* message) {
    if (!condition) { fprintf(stderr, "Snake benchmark: %s\n", message); exit(1); }
}
static int snakebench_int(Dict* kwargs, const char* key, int low, int high) {
    DictItem* item = dict_find(kwargs, key);
    snakebench_require(item && isfinite(item->value) && item->value >= low
        && item->value <= high && item->value == floor(item->value), key);
    return (int)item->value;
}
static float snakebench_reward(Dict* kwargs, const char* key, float low, float high) {
    DictItem* item = dict_find(kwargs, key);
    snakebench_require(item && isfinite(item->value) && item->value >= low && item->value <= high, key);
    return (float)item->value;
}
static unsigned int snakebench_draw(Env* env, unsigned int bound) {
    assert(bound > 0);
    unsigned int threshold = (0u - bound) % bound, draw;
    do { env->rng = env->rng * 1664525u + 1013904223u; draw = env->rng; } while (draw < threshold);
    return draw % bound;
}
static int snakebench_empty(Env* env) {
    int count = 0, area = env->width * env->height;
    for (int i = 0; i < area; i++) count += env->grid[i] == SNAKE_EMPTY;
    snakebench_require(count > 0, "no empty spawn cell");
    unsigned int rank = snakebench_draw(env, (unsigned int)count);
    for (int i = 0; i < area; i++) {
        if (env->grid[i] == SNAKE_EMPTY && rank-- == 0) return i;
    }
    snakebench_require(false, "spawn rank failed");
    return -1;
}

void puf_init(Env* env, Dict* kwargs) {
    snakebench_int(kwargs, "rule_version", 1, 1);
    snakebench_int(kwargs, "num_agents", 1, 1);
    snakebench_int(kwargs, "vision", 5, 5);
    snakebench_int(kwargs, "leave_corpse_on_death", 0, 0);
    env->width = snakebench_int(kwargs, "width", 13, 128);
    env->height = snakebench_int(kwargs, "height", 13, 128);
    int capacity = (env->width - 10) * (env->height - 10);
    env->food = snakebench_int(kwargs, "num_food", 1, capacity - 1);
    env->max_snake_length = snakebench_int(kwargs, "max_snake_length", 2, capacity - env->food + 1);
    env->max_steps = snakebench_int(kwargs, "max_steps", 1, 16777216);
    env->cell_size = snakebench_int(kwargs, "cell_size", 1, 64);
    env->reward_food = snakebench_reward(kwargs, "reward_food", 0, 16);
    env->reward_death = snakebench_reward(kwargs, "reward_death", -16, 0);
    env->representation = cnn_appearance_init(kwargs, env->rng, SNAKECNN_NUM_REPRESENTATIONS);
    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
    env->grid = (unsigned char*)calloc(env->width * env->height, sizeof(unsigned char));
    env->snake = (int*)malloc(env->max_snake_length * sizeof(int));
    snakebench_require(env->grid && env->snake, "allocation failed");
}

void compute_observations(Env* env) {
    obs_t* obs = env->agents[0].observations;
    memset(obs, 0, OBS_SIZE * sizeof(obs_t));
    int head = env->snake[env->head_ptr];
    int row = head / env->width - 5, col = head % env->width - 5;
    for (int y = 0; y < 11; y++) {
        for (int x = 0; x < 11; x++) {
            int tile = env->grid[(row + y) * env->width + col + x];
#if SNAKEBENCH_PIXELS
            float value = tile == SNAKE_FOOD ? .25f : tile == SNAKE_WALL ? .5f : tile == SNAKE_BODY ? .75f : 0;
            if (x == 5 && y == 5) value = 1;
            if (env->representation == 4 && tile == SNAKE_FOOD) value = .75f;
            if (env->representation == 4 && tile == SNAKE_BODY && (x != 5 || y != 5)) value = .25f;
            for (int dy = 0; dy < 3; dy++) {
                for (int dx = 0; dx < 3; dx++) {
                    if (env->representation == 1 && dx != 1 && dy != 1) continue;
                    if (env->representation == 2 && (dx == 2 || dy == 2)) continue;
                    obs[(1 + 3*y + dy) * OBS_WIDTH + 5 + 3*x + dx] = value;
                }
            }
#else
            obs[(y * 11 + x) * 8 + tile] = 1;
#endif
        }
    }
#if SNAKEBENCH_PIXELS
    if (env->representation == 3) {
        for (int i = 0; i < OBS_SIZE; i++) obs[i] = 1 - obs[i];
    } else if (env->representation == 5) {
        for (int y = 0; y < OBS_HEIGHT; y++) {
            for (int x = 0; x < OBS_WIDTH / 2; x++) {
                float value = obs[y * OBS_WIDTH + x];
                obs[y * OBS_WIDTH + x] = obs[y * OBS_WIDTH + OBS_WIDTH - 1 - x];
                obs[y * OBS_WIDTH + OBS_WIDTH - 1 - x] = value;
            }
        }
    }
#endif
}

void puf_reset(Env* env) {
    memset(env->grid, 0, env->width * env->height * sizeof(unsigned char));
    for (int i = 0; i < env->max_snake_length; i++) env->snake[i] = -1;
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            if (y < 5 || y >= env->height - 5 || x < 5 || x >= env->width - 5)
                env->grid[y * env->width + x] = SNAKE_WALL;
        }
    }
    env->head_ptr = 0;
    env->length = 1;
    env->tick = env->food_collected = 0;
    env->episode_return = 0;
    env->last_score = env->last_food = env->last_decisions = env->last_end_reason = 0;
    env->last_return = 0;
    env->snake[0] = snakebench_empty(env);
    env->grid[env->snake[0]] = SNAKE_BODY;
    for (int i = 0; i < env->food; i++) env->grid[snakebench_empty(env)] = SNAKE_FOOD;
    env->agents[0].actions[0] = env->agents[0].rewards[0] = env->agents[0].terminals[0] = 0;
    compute_observations(env);
}

static void snakebench_end(Env* env, int reason) {
    int score = env->length, food = env->food_collected, decisions = env->tick;
    float total = env->episode_return, reward = env->agents[0].rewards[0];
    env->log.score += score;
    env->log.perf += fminf(score / 120.0f, 1);
    env->log.food_collected += food;
    env->log.episode_return += total;
    env->log.episode_length += decisions;
    env->log.deaths += reason == 1;
    env->log.horizon_ends += reason == 2;
    env->log.n += 1;
    puf_reset(env);
    env->last_score = score;
    env->last_food = food;
    env->last_decisions = decisions;
    env->last_return = total;
    env->last_end_reason = reason;
    env->agents[0].rewards[0] = reward;
    env->agents[0].terminals[0] = 1;
}

void puf_step(Env* env) {
    float action = env->agents[0].actions[0];
    snakebench_require(isfinite(action) && action >= 0 && action <= 3 && action == floorf(action), "invalid action");
    env->agents[0].rewards[0] = env->agents[0].terminals[0] = 0;
    env->last_end_reason = 0;
    env->tick++;
    int atn = (int)action;
    int delta = atn == 0 ? -env->width : atn == 1 ? env->width : atn == 2 ? -1 : 1;
    int head = env->snake[env->head_ptr], next = head + delta;
    int neck_ptr = (env->head_ptr + env->max_snake_length - 1) % env->max_snake_length;
    if (env->length > 1 && next == env->snake[neck_ptr]) next = head - delta;
    int tile = env->grid[next];
    if (tile >= SNAKE_WALL) {
        env->agents[0].rewards[0] = env->reward_death;
        env->episode_return += env->reward_death;
        snakebench_end(env, 1);
        return;
    }
    int ptr = (env->head_ptr + 1) % env->max_snake_length;
    if (tile == SNAKE_FOOD && env->length < env->max_snake_length - 1) {
        env->length++;
    } else {
        int tail_ptr = (ptr + env->max_snake_length - env->length) % env->max_snake_length;
        env->grid[env->snake[tail_ptr]] = SNAKE_EMPTY;
        env->snake[tail_ptr] = -1;
    }
    env->head_ptr = ptr;
    env->snake[ptr] = next;
    env->grid[next] = SNAKE_BODY;
    if (tile == SNAKE_FOOD) {
        env->food_collected++;
        env->agents[0].rewards[0] = env->reward_food;
        env->episode_return += env->reward_food;
        env->grid[snakebench_empty(env)] = SNAKE_FOOD;
    }
    if (env->tick == env->max_steps) { snakebench_end(env, 2); return; }
    compute_observations(env);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "food_collected", log->food_collected);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "deaths", log->deaths);
    dict_set(out, "horizon_ends", log->horizon_ends);
    dict_set(out, "n", log->n);
}
void puf_render(Env* env) {
    if (!env->viewer_open) {
        InitWindow(env->width * env->cell_size, env->height * env->cell_size, "PufferLib episodic Snake");
        SetTargetFPS(10); env->viewer_open = 1;
    }
    BeginDrawing(); ClearBackground(BLACK);
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int tile = env->grid[y * env->width + x];
            Color color = tile == SNAKE_WALL ? GRAY : tile == SNAKE_FOOD ? RED : tile == SNAKE_BODY ? GREEN : BLACK;
            DrawRectangle(x * env->cell_size, y * env->cell_size, env->cell_size, env->cell_size, color);
        }
    }
    EndDrawing(); puf_web_vsync();
}
void puf_close(Env* env) {
    if (env->viewer_open) CloseWindow();
    free(env->grid); free(env->snake);
}
