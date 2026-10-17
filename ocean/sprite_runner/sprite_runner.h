#include <stdlib.h>
#include <string.h>
typedef float obs_t;
#include "pufferenv.h"
#ifdef SPRITE_ATTENTION
#define ACT_SIZES {2, 20, 15}
#define NUM_ATNS 3
#else
#define ACT_SIZES {2}
#define NUM_ATNS 1
#endif
#define PUF_STEPS_PER_SEC 12
#define GRID_WIDTH 20
#define GRID_HEIGHT 15
#define STATE_OBS_SIZE 7
#define SPRITE_RULES "sprite-runner-v3"
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
    unsigned int object_rng;
    unsigned int floor_seed;
    unsigned char grid[GRID_WIDTH * GRID_HEIGHT];
    unsigned char pixel_grid[GRID_WIDTH * GRID_HEIGHT];
    SpriteClient* client;
    int observation_mode;
    int representation;
    int items;
    int player;
    int object;
    int kind;
    int vy;
    int tick;
    int count;
    int correct;
#ifdef SPRITE_ATTENTION
    int partial_observation;
    int gaze_idx;
#endif
    float episode_return;
};

#ifdef SPRITE_ATTENTION
#define SPRITE_OUTSIDE RUNNER_SAND
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
    obs[0] = (env->player % GRID_WIDTH) / (float)GRID_WIDTH;
    obs[1] = (env->player / GRID_WIDTH) / (float)GRID_HEIGHT;
    obs[2] = env->vy / 3.f;
    obs[3] = env->player / GRID_WIDTH == 11;
    obs[4] = (env->object % GRID_WIDTH - 3) / (float)GRID_WIDTH;
    obs[5] = env->kind <= RUNNER_BERRY;
    obs[6] = (11 - env->object / GRID_WIDTH) / 3.f;
}

void runner_spawn(Env* env) {
    env->object_rng = env->object_rng * 1664525u + 1013904223u;
    env->kind = 1 + (env->object_rng >> 16) % 4;
    int heights[] = {11, 8, 9};
    int y = env->kind <= RUNNER_BERRY ? heights[(env->object_rng >> 24) % 3] : 11;
    env->object = y * GRID_WIDTH + GRID_WIDTH - 1;
    env->grid[env->object] = env->kind;
}

void puf_init(Env* env, Dict* kwargs) {
    env->rng += (unsigned int)dict_get(kwargs, "seed");
    env->observation_mode = dict_get(kwargs, "observation_mode");
#ifdef SPRITE_ATTENTION
    env->partial_observation = dict_get(kwargs, "partial_observation");
#endif
    env->representation = dict_get(kwargs, "representation");
    env->items = dict_get(kwargs, "items");
    env->num_agents = 1;
    env->agents[0].policy = 0;
    if (env->observation_mode) {
        sprite_make_tiles(env->representation);
        memset(env->pixel_grid, 255, sizeof(env->pixel_grid));
    }
}

void puf_reset(Env* env) {
    // One world draw per episode; item draws never advance the next world.
    env->rng = env->rng * 1664525u + 1013904223u;
    env->object_rng = env->rng ^ 0x27d4eb2du;
    env->object_rng ^= env->object_rng >> 16;
    env->object_rng *= 0x85ebca6bu;
    env->floor_seed = env->rng ^ 0xf0d7b6c3u;
    memset(env->grid, 0, sizeof(env->grid));
    for (int y = 12; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            env->grid[y * GRID_WIDTH + x] = RUNNER_SAND;
        }
    }
    env->player = 11 * GRID_WIDTH + 3;
    env->tick = 0;
    env->count = 0;
    env->correct = 0;
    env->vy = 0;
    env->episode_return = 0;
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 0;
    runner_spawn(env);
#ifdef SPRITE_ATTENTION
    env->gaze_idx = env->player;
#endif
    compute_observations(env);
}

void puf_step(Env* env) {
#ifdef SPRITE_ATTENTION
    sprite_attention_action(env);
#endif
    int previous = env->player, old_object = env->object;
    int y = env->player / GRID_WIDTH;
    if (env->agents[0].actions[0] == 1 && y == 11) {
        env->vy = -2;
    }
    y += env->vy;
    if (y >= 11) {
        y = 11;
        env->vy = 0;
    } else {
        env->vy++;
    }
    env->player = y * GRID_WIDTH + 3;
    env->grid[env->object] = RUNNER_EMPTY;
    env->object--;
    env->tick++;
    float reward = 0;
    int died = 0;
    if (env->object % GRID_WIDTH == 3) {
        int touched = y == env->object / GRID_WIDTH;
        int correct = touched == (env->kind <= RUNNER_BERRY);
        died = touched && env->kind > RUNNER_BERRY;
        env->correct += correct;
        env->count++;
        reward = correct ? 1 : -1;
        if (!died && env->count < env->items) {
            runner_spawn(env);
        }
    } else {
        env->grid[env->object] = env->kind;
    }
    env->episode_return += reward;
    env->agents[0].terminals[0] = 0;
    if (died || env->count == env->items) {
        float perf = env->correct / (float)env->items;
        env->log.perf += perf;
        env->log.successes += perf == 1;
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
        int dirty[] = {previous, env->player, old_object, env->object};
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
        "PufferLib Sprite Runner", PUF_STEPS_PER_SEC);
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
