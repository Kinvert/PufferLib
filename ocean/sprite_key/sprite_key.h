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
#define STATE_OBS_SIZE 37
#define SPRITE_RULES "sprite-key-v2"
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
    short key_distance[GRID_WIDTH * GRID_HEIGHT], goal_distance[GRID_WIDTH * GRID_HEIGHT];
    SpriteClient* client;
    int observation_mode;
    int representation;
    int max_steps;
    int player;
    int key;
    int mimic;
    int door;
    int goal;
    int has_key;
    int tick;
#ifdef SPRITE_ATTENTION
    int partial_observation;
    int gaze_idx;
#endif
    float episode_return;
};

#ifdef SPRITE_ATTENTION
#define SPRITE_OUTSIDE KEY_WALL
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

int key_floor(Env* env, int cell) {
    unsigned int value = env->floor_seed ^ (cell * 0x9e3779b9u);
    value ^= value >> 16;
    value *= 0x85ebca6bu;
    return KEY_EMPTY + (value >> 24) % 3;
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
    obs[2] = !env->has_key;
    obs[3] = env->has_key ? 0 : (env->key % GRID_WIDTH - px) / (float)GRID_WIDTH;
    obs[4] = env->has_key ? 0 : (env->key / GRID_WIDTH - py) / (float)GRID_HEIGHT;
    obs[5] = (env->door % GRID_WIDTH - px) / (float)GRID_WIDTH;
    obs[6] = (env->door / GRID_WIDTH - py) / (float)GRID_HEIGHT;
    obs[7] = (env->goal % GRID_WIDTH - px) / (float)GRID_WIDTH;
    obs[8] = (env->goal / GRID_WIDTH - py) / (float)GRID_HEIGHT;
    obs[9] = env->has_key;
    obs[10] = (env->mimic % GRID_WIDTH - px) / (float)GRID_WIDTH;
    obs[11] = (env->mimic / GRID_WIDTH - py) / (float)GRID_HEIGHT;
    int index = 12;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            int sx = px + x, sy = py + y;
            int kind = sx < 0 || sx >= GRID_WIDTH || sy < 0 || sy >= GRID_HEIGHT
                ? KEY_WALL : env->grid[sy * GRID_WIDTH + sx];
            obs[index++] = kind == KEY_WALL ? -1 : kind == KEY_MIMIC ? -.75f
                : kind == KEY_DOOR ? -.5f : kind == KEY_CHEST ? .5f
                : kind == KEY_EXIT ? 1 : 0;
        }
    }
}

void key_distances(Env* env, int goal, short* distance, int opened) {
    for (int i = 0; i < GRID_WIDTH * GRID_HEIGHT; i++) {
        distance[i] = -1;
    }
    int queue[GRID_WIDTH * GRID_HEIGHT], head = 0, tail = 1;
    int delta[] = {-GRID_WIDTH, GRID_WIDTH, -1, 1};
    queue[0] = goal;
    distance[goal] = 0;
    while (head < tail) {
        int cell = queue[head++];
        for (int a = 0; a < 4; a++) {
            int next = cell + delta[a];
            int kind = env->grid[next];
            if (kind != KEY_WALL && kind != KEY_MIMIC
                && (opened || kind != KEY_DOOR) && distance[next] < 0) {
                distance[next] = distance[cell] + 1;
                queue[tail++] = next;
            }
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
    env->floor_seed = env->rng ^ 0x51ed270bu;
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            int cell = y * GRID_WIDTH + x;
            env->grid[cell] = !x || !y || x == GRID_WIDTH - 1
                || y == GRID_HEIGHT - 1 || x == 10 ? KEY_WALL : key_floor(env, cell);
        }
    }
    env->rng = env->rng * 1664525u + 1013904223u;
    env->door = (1 + (env->rng >> 16) % 13) * GRID_WIDTH + 10;
    env->grid[env->door] = KEY_DOOR;
    env->rng = env->rng * 1664525u + 1013904223u;
    int chest = (env->rng >> 16) % 91;
    env->key = (1 + chest / 7) * GRID_WIDTH + 2 + chest % 7;
    env->grid[env->key] = KEY_CHEST;
    env->rng = env->rng * 1664525u + 1013904223u;
    chest = (chest + 1 + (env->rng >> 16) % 90) % 91;
    env->mimic = (1 + chest / 7) * GRID_WIDTH + 2 + chest % 7;
    env->grid[env->mimic] = KEY_MIMIC;
    env->rng = env->rng * 1664525u + 1013904223u;
    env->goal = (1 + (env->rng >> 16) % 13) * GRID_WIDTH + 18;
    env->grid[env->goal] = KEY_EXIT;
    env->player = 7 * GRID_WIDTH + 1;
    key_distances(env, env->key, env->key_distance, 0);
    key_distances(env, env->goal, env->goal_distance, 1);
    env->tick = env->has_key = 0;
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
    int previous = env->player;
    int next = previous + delta[(int)env->agents[0].actions[0]];
    int kind = env->grid[next];
    short* distance = env->has_key ? env->goal_distance : env->key_distance;
    if (kind != KEY_WALL && kind != KEY_DOOR) {
        env->player = next;
    }
    int died = env->player == env->mimic;
    float reward = died ? -1 : .02f * (distance[previous] - distance[env->player]) - .001f;
    if (env->grid[env->player] == KEY_CHEST) {
        env->has_key = 1;
        env->grid[env->key] = key_floor(env, env->key);
        env->grid[env->door] = key_floor(env, env->door);
        reward += .25f;
    }
    int success = env->player == env->goal;
    reward += success;
    env->tick++;
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
        int dirty[] = {previous, env->player, env->key, env->door};
        for (int i = 0; i < 4; i++) {
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
        "PufferLib Sprite Key", PUF_STEPS_PER_SEC);
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
