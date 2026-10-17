#include <stdlib.h>
#include <string.h>
typedef float obs_t;
#include "pufferenv.h"
#ifdef SPRITE_ATTENTION
#define ACT_SIZES {5, 20, 15}
#define NUM_ATNS 3
#else
#define ACT_SIZES {5}
#define NUM_ATNS 1
#endif
#define PUF_STEPS_PER_SEC 12
#define GRID_WIDTH 20
#define GRID_HEIGHT 15
#define STATE_OBS_SIZE 36
#define SPRITE_RULES "sprite-push-v2"
#include "sprites.h"
#include "../sprite_common/sprite.h"

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float successes;
    float deaths;
    float n;
};
struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
    unsigned int floor_seed;
    unsigned char grid[GRID_WIDTH * GRID_HEIGHT];
    unsigned char pixel_grid[GRID_WIDTH * GRID_HEIGHT];
    SpriteClient* client;
    int observation_mode;
    int representation;
    int max_steps;
    int player;
    int crate;
    int poison;
    int enemy;
    int goal;
    int tick;
#ifdef SPRITE_ATTENTION
    int partial_observation;
    int gaze_idx;
#endif
    float episode_return;
};

#ifdef SPRITE_ATTENTION
#define SPRITE_OUTSIDE PUSH_WALL
#include "../sprite_common/attention.h"
#endif

int puf_observation_size(Dict* kwargs) {
#ifdef SPRITE_ATTENTION
    if (dict_get(kwargs, "partial_observation")) {
        return (dict_get(kwargs, "observation_mode") ? SPRITE_VIEW_PIXELS : SPRITE_VIEW_STATE) + 4;
    }
#endif
    return dict_get(kwargs, "observation_mode") ? PIXEL_OBS_SIZE : STATE_OBS_SIZE;
}

int push_distance(int a, int b) {
    return abs(a % GRID_WIDTH - b % GRID_WIDTH) + abs(a / GRID_WIDTH - b / GRID_WIDTH);
}

int push_floor(Env* env, int cell) {
    unsigned int value = env->floor_seed ^ (cell * 0x9e3779b9u);
    value ^= value >> 16;
    value *= 0x85ebca6bu;
    return PUSH_EMPTY + (value >> 24) % 3;
}

void compute_observations(Env* env) {
#ifdef SPRITE_ATTENTION
    if (env->partial_observation) {
        sprite_attention_observations(env);
        return;
    }
#endif
    float* obs = env->agents[0].observations;
    if (env->observation_mode) {
        sprite_observe(obs, env->grid, env->player, env->representation, env->floor_seed,
            env->pixel_grid);
        return;
    }
    int px = env->player % GRID_WIDTH, py = env->player / GRID_WIDTH;
    obs[0] = px / (float)GRID_WIDTH;
    obs[1] = py / (float)GRID_HEIGHT;
    obs[2] = (env->crate % GRID_WIDTH - px) / (float)GRID_WIDTH;
    obs[3] = (env->crate / GRID_WIDTH - py) / (float)GRID_HEIGHT;
    obs[4] = (env->goal % GRID_WIDTH - px) / (float)GRID_WIDTH;
    obs[5] = (env->goal / GRID_WIDTH - py) / (float)GRID_HEIGHT;
    obs[6] = (env->poison % GRID_WIDTH - px) / (float)GRID_WIDTH;
    obs[7] = (env->poison / GRID_WIDTH - py) / (float)GRID_HEIGHT;
    obs[8] = (env->enemy % GRID_WIDTH - px) / (float)GRID_WIDTH;
    obs[9] = (env->enemy / GRID_WIDTH - py) / (float)GRID_HEIGHT;
    obs[10] = env->tick % 2;
    int index = 11;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            int sx = px + x, sy = py + y;
            int kind = sx < 0 || sx >= GRID_WIDTH || sy < 0 || sy >= GRID_HEIGHT
                ? PUSH_WALL : env->grid[sy * GRID_WIDTH + sx];
            obs[index++] = kind == PUSH_WALL ? -1 : kind == PUSH_POISON ? -.75f
                : kind == PUSH_ENEMY ? -.5f : kind == PUSH_CRATE ? .5f
                : kind == PUSH_GOAL ? 1 : 0;
        }
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->rng += (unsigned int)dict_get(kwargs, "seed");
    env->observation_mode = dict_get(kwargs, "observation_mode");
#ifdef SPRITE_ATTENTION
    env->partial_observation = dict_get(kwargs, "partial_observation");
#endif
    env->representation = dict_get(kwargs, "representation");
    env->max_steps = dict_get(kwargs, "max_steps");
    env->num_agents = 1;
    env->agents[0].policy = 0;
    if (env->observation_mode) {
        sprite_make_tiles(env->representation);
        memset(env->pixel_grid, 255, sizeof(env->pixel_grid));
    }
}

void puf_reset(Env* env) {
    env->floor_seed = env->rng ^ 0x68bc21ebu;
    env->rng = env->rng * 1664525u + 1013904223u;
    int x = 6 + (env->rng >> 16) % 8, y = 4 + (env->rng >> 24) % 7;
    env->crate = y * GRID_WIDTH + x;
    env->rng = env->rng * 1664525u + 1013904223u;
    int direction = (env->rng >> 16) % 4;
    int dx[] = {0, 1, 0, -1}, dy[] = {-1, 0, 1, 0};
    env->goal = (y + 3 * dy[direction]) * GRID_WIDTH + x + 3 * dx[direction];
    env->player = (y - 2 * dy[direction] + dx[direction]) * GRID_WIDTH
        + x - 2 * dx[direction] + dy[direction];
    int forward = dy[direction] * GRID_WIDTH + dx[direction];
    int side = dx[direction] * GRID_WIDTH + dy[direction];
    for (int cy = 0; cy < GRID_HEIGHT; cy++) {
        for (int cx = 0; cx < GRID_WIDTH; cx++) {
            int cell = cy * GRID_WIDTH + cx;
            int along = (cx - x) * dx[direction] + (cy - y) * dy[direction];
            int across = (cx - x) * dy[direction] + (cy - y) * dx[direction];
            unsigned int value = env->rng ^ (cell * 0x27d4eb2du);
            int clear_route = along >= -2 && along <= 3 && abs(across) <= 1;
            int wall = !cx || !cy || cx == GRID_WIDTH - 1 || cy == GRID_HEIGHT - 1;
            env->grid[cell] = wall || (!clear_route && (value >> 16) % 9 == 0)
                ? PUSH_WALL : push_floor(env, cell);
        }
    }
    // A sideways push enters a real corner; the forward solution stays open.
    env->grid[env->crate + 2 * side] = PUSH_WALL;
    env->grid[env->crate + side - forward] = PUSH_WALL;
    env->poison = env->crate + 2 * forward + 2 * side;
    env->enemy = (y < 7 ? 13 : 1) * GRID_WIDTH + (x < 10 ? 18 : 1);
    env->grid[env->poison] = PUSH_POISON;
    env->grid[env->enemy] = PUSH_ENEMY;
    env->grid[env->goal] = PUSH_GOAL;
    env->grid[env->crate] = PUSH_CRATE;
    env->tick = 0;
    env->episode_return = 0;
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 0;
#ifdef SPRITE_ATTENTION
    env->gaze_idx = env->player;
#endif
    compute_observations(env);
}

void puf_step(Env* env) {
#ifdef SPRITE_ATTENTION
    sprite_attention_action(env);
#endif
    int delta[] = {0, -GRID_WIDTH, GRID_WIDTH, -1, 1};
    int previous = env->player, old_crate = env->crate, old_enemy = env->enemy;
    int move = delta[(int)env->agents[0].actions[0]];
    int next = previous + move;
    int target = next == env->crate ? env->grid[next + move] : PUSH_WALL;
    if (move && next == env->crate && target != PUSH_WALL
        && target != PUSH_POISON && target != PUSH_ENEMY) {
        env->grid[env->crate] = env->crate == env->goal ? PUSH_GOAL
            : push_floor(env, env->crate);
        env->crate += move;
        env->grid[env->crate] = PUSH_CRATE;
    }
    if (env->grid[next] != PUSH_WALL && next != env->crate) {
        env->player = next;
    }
    int died = env->player == env->poison || env->player == env->enemy;
    env->tick++;
    if (!died && env->tick % 2 == 0) {
        int ex = env->enemy % GRID_WIDTH, ey = env->enemy / GRID_WIDTH;
        int dx = env->player % GRID_WIDTH - ex, dy = env->player / GRID_WIDTH - ey;
        int horizontal = dx > 0 ? 1 : dx < 0 ? -1 : 0;
        int vertical = dy > 0 ? GRID_WIDTH : dy < 0 ? -GRID_WIDTH : 0;
        int moves[] = {horizontal, vertical};
        if (abs(dy) > abs(dx)) {
            moves[0] = vertical;
            moves[1] = horizontal;
        }
        for (int i = 0; i < 2; i++) {
            int cell = env->enemy + moves[i], kind = env->grid[cell];
            if (moves[i] && kind != PUSH_WALL && kind != PUSH_CRATE && kind != PUSH_POISON) {
                env->grid[env->enemy] = env->enemy == env->goal ? PUSH_GOAL
                    : push_floor(env, env->enemy);
                env->enemy = cell;
                env->grid[cell] = PUSH_ENEMY;
                break;
            }
        }
        died = env->enemy == env->player;
    }
    int success = !died && env->crate == env->goal;
    float reward = .02f * (push_distance(previous, old_crate)
        - push_distance(env->player, env->crate)) + .1f * (push_distance(old_crate, env->goal)
        - push_distance(env->crate, env->goal)) - .001f + success - died;
    env->episode_return += reward;
    env->agents[0].terminals[0] = 0;
    if (success || died || env->tick >= env->max_steps) {
        env->log.perf += success;
        env->log.successes += success;
        env->log.deaths += died;
        env->log.score += env->episode_return;
        env->log.episode_return += env->episode_return;
        env->log.episode_length += env->tick;
        env->log.n++;
        puf_reset(env);
        env->agents[0].terminals[0] = 1;
#ifdef SPRITE_ATTENTION
    } else if (env->partial_observation) {
        compute_observations(env);
#endif
    } else if (env->observation_mode) {
        int dirty[] = {previous, env->player, old_crate, env->crate, old_enemy, env->enemy};
        for (int i = 0; i < 6; i++) {
            sprite_write_cell(env->agents[0].observations, env->grid, env->player,
                dirty[i], env->representation, env->floor_seed, env->pixel_grid);
        }
    } else {
        compute_observations(env);
    }
    env->agents[0].rewards[0] = reward;
}

void puf_render(Env* env) {
    sprite_render(&env->client, env->grid, env->player, env->representation, env->floor_seed,
        "PufferLib Sprite Push", PUF_STEPS_PER_SEC);
}

void puf_close(Env* env) {
    sprite_close(env->client);
    env->client = NULL;
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "successes", log->successes);
    dict_set(out, "deaths", log->deaths);
    dict_set(out, "n", log->n);
}
