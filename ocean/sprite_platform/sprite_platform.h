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
#define PUF_STEPS_PER_SEC 10
#define GRID_WIDTH 20
#define GRID_HEIGHT 15
#define STATE_OBS_SIZE 41
#define SPRITE_RULES "sprite-platform-v2"
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
    int goal;
    int vy;
    int tick;
#ifdef SPRITE_ATTENTION
    int partial_observation;
    int gaze_idx;
#endif
    float episode_return;
};

#ifdef SPRITE_ATTENTION
#define SPRITE_OUTSIDE PLATFORM_STONE
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
    int px = env->player % GRID_WIDTH, py = env->player / GRID_WIDTH;
    obs[0] = px / (float)GRID_WIDTH;
    obs[1] = py / (float)GRID_HEIGHT;
    obs[2] = env->vy / 3.f;
    obs[3] = py < GRID_HEIGHT - 1 && env->grid[env->player + GRID_WIDTH] == PLATFORM_STONE;
    obs[4] = (env->goal % GRID_WIDTH - px) / (float)GRID_WIDTH;
    obs[5] = (env->goal / GRID_WIDTH - py) / (float)GRID_HEIGHT;
    int index = 6;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 4; x++) {
            int sx = px + x, sy = py + y;
            int kind = sx < 0 || sx >= GRID_WIDTH || sy < 0 || sy >= GRID_HEIGHT
                ? PLATFORM_STONE : env->grid[sy * GRID_WIDTH + sx];
            obs[index++] = kind == PLATFORM_STONE ? -1 : kind == PLATFORM_FLAG ? 1 : 0;
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
    env->floor_seed = env->rng ^ 0x98a296b5u;
    memset(env->grid, 0, sizeof(env->grid));
    env->rng = env->rng * 1664525u + 1013904223u;
    int ground = 10 + (env->rng >> 16) % 4;
    int step = 1 + (env->rng >> 24) % 2;
    for (int x = 0; x < GRID_WIDTH; x++) {
        int top = ground - (x >= 8 && x <= 9) - (x >= 15 ? step : 0);
        for (int y = top; y < GRID_HEIGHT; y++) {
            env->grid[y * GRID_WIDTH + x] = PLATFORM_STONE;
        }
    }
    for (int gap = 0; gap < 2; gap++) {
        env->rng = env->rng * 1664525u + 1013904223u;
        int start = 4 + gap * 7 + (env->rng >> 16) % 2;
        int width = 1 + (env->rng >> 24) % 2;
        for (int y = 0; y < GRID_HEIGHT; y++) {
            for (int x = start; x < start + width; x++) {
                env->grid[y * GRID_WIDTH + x] = PLATFORM_EMPTY;
            }
        }
    }
    env->goal = (ground - step - 1) * GRID_WIDTH + GRID_WIDTH - 2;
    env->grid[env->goal] = PLATFORM_FLAG;
    env->player = (ground - 1) * GRID_WIDTH + 1;
    env->tick = 0;
    env->vy = 0;
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
    // 0 wait;1 left;2 right;3 jump-left;4 jump-right.
    int action = env->agents[0].actions[0];
    int previous = env->player;
    int x = previous % GRID_WIDTH, y = previous / GRID_WIDTH;
    int grounded = y < GRID_HEIGHT - 1 && env->grid[previous + GRID_WIDTH] == PLATFORM_STONE;
    if (grounded && action >= 3) {
        env->vy = -2;
    }
    int dx = action == 1 || action == 3 ? -1 : action == 2 || action == 4 ? 1 : 0;
    int nx = x + dx;
    if (nx >= 0 && nx < GRID_WIDTH && env->grid[y * GRID_WIDTH + nx] != PLATFORM_STONE) {
        x = nx;
    }
    int speed = abs(env->vy), direction = env->vy < 0 ? -1 : 1;
    for (int i = 0; i < speed; i++) {
        int ny = y + direction;
        if (ny < 0 || ny >= GRID_HEIGHT || env->grid[ny * GRID_WIDTH + x] == PLATFORM_STONE) {
            env->vy = 0;
            break;
        }
        y = ny;
    }
    if (env->vy < 3) {
        env->vy++;
    }
    env->player = y * GRID_WIDTH + x;
    env->tick++;
    int success = env->player == env->goal;
    int died = y >= GRID_HEIGHT - 1;
    float reward = .05f * (x - previous % GRID_WIDTH) - .001f + success - died;
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
        float* obs = env->agents[0].observations;
        sprite_write_cell(obs, env->grid, env->player, previous, env->representation,
            env->floor_seed, env->pixel_grid);
        sprite_write_cell(obs, env->grid, env->player, env->player, env->representation,
            env->floor_seed, env->pixel_grid);
    } else {
        compute_observations(env);
    }
    env->agents[0].rewards[0] = reward;
}

void puf_render(Env* env) {
    sprite_render(&env->client, env->grid, env->player, env->representation, env->floor_seed,
        "PufferLib Sprite Platform", PUF_STEPS_PER_SEC);
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
