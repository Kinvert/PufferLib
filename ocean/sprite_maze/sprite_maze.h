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
#define STATE_OBS_SIZE 29
#define SPRITE_RULES "sprite-maze-v3"
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
    short distance[GRID_WIDTH * GRID_HEIGHT];
    SpriteClient* client;
    int observation_mode;
    int representation;
    int max_steps;
    int player;
    int goal;
    int tick;
#ifdef SPRITE_ATTENTION
    int partial_observation;
    int gaze_idx;
#endif
    float episode_return;
};

#ifdef SPRITE_ATTENTION
#define SPRITE_OUTSIDE MAZE_HEDGE
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
    obs[2] = (env->goal % GRID_WIDTH - px) / (float)GRID_WIDTH;
    obs[3] = (env->goal / GRID_WIDTH - py) / (float)GRID_HEIGHT;
    int index = 4;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            int sx = px + x, sy = py + y;
            int kind = sx < 0 || sx >= GRID_WIDTH || sy < 0 || sy >= GRID_HEIGHT
                ? MAZE_HEDGE : env->grid[sy * GRID_WIDTH + sx];
            obs[index++] = kind == MAZE_HEDGE ? -1 : kind == MAZE_TRAP ? -.5f
                : kind == MAZE_GOLD ? 1 : 0;
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
    env->floor_seed = env->rng ^ 0x9fe289d7u;
    // Growing-tree carving, like ocean/maze, using this game's own RNG.
    memset(env->grid, MAZE_HEDGE, sizeof(env->grid));
    int stack[GRID_WIDTH * GRID_HEIGHT], count = 1;
    env->player = GRID_WIDTH + 1;
    stack[0] = env->player;
    env->grid[env->player] = MAZE_PATH;
    int steps[] = {-2 * GRID_WIDTH, 2 * GRID_WIDTH, -2, 2};
    while (count) {
        int cell = stack[count - 1], neighbors[4], n = 0;
        for (int i = 0; i < 4; i++) {
            int next = cell + steps[i];
            int x = next % GRID_WIDTH, y = next / GRID_WIDTH;
            if (next > 0 && x > 0 && x < GRID_WIDTH - 1 && y > 0
                && y < GRID_HEIGHT - 1 && env->grid[next] == MAZE_HEDGE) {
                neighbors[n++] = next;
            }
        }
        if (!n) {
            count--;
            continue;
        }
        env->rng = env->rng * 1664525u + 1013904223u;
        int next = neighbors[(env->rng >> 16) % n];
        env->grid[(cell + next) / 2] = env->grid[next] = MAZE_PATH;
        stack[count++] = next;
    }
    // Extra side passages guarantee enough leaves; traps can be anywhere.
    for (int i = 0; i < 7; i++) {
        env->grid[(2 * i + 1) * GRID_WIDTH + 18] = MAZE_PATH;
    }
    env->goal = 13 * GRID_WIDTH + 17;
    env->grid[env->goal] = MAZE_GOLD;
    int leaves[GRID_WIDTH * GRID_HEIGHT], leaf_count = 0;
    int delta[] = {-GRID_WIDTH, GRID_WIDTH, -1, 1};
    for (int cell = GRID_WIDTH + 1; cell < (GRID_HEIGHT - 1) * GRID_WIDTH; cell++) {
        if (env->grid[cell] != MAZE_PATH || cell == env->player) {
            continue;
        }
        int exits = 0;
        for (int i = 0; i < 4; i++) {
            exits += env->grid[cell + delta[i]] != MAZE_HEDGE;
        }
        if (exits == 1) {
            leaves[leaf_count++] = cell;
        }
    }
    // Removing leaves other than start/goal cannot disconnect their safe path.
    for (int i = 0; i < 3; i++) {
        env->rng = env->rng * 1664525u + 1013904223u;
        int pick = i + (env->rng >> 16) % (leaf_count - i);
        int cell = leaves[pick];
        leaves[pick] = leaves[i];
        env->grid[cell] = MAZE_TRAP;
    }
    // Reset-only distances provide potential shaping, never observation hints.
    for (int i = 0; i < GRID_WIDTH * GRID_HEIGHT; i++) {
        env->distance[i] = -1;
    }
    int queue[GRID_WIDTH * GRID_HEIGHT], head = 0, tail = 1;
    queue[0] = env->goal;
    env->distance[env->goal] = 0;
    while (head < tail) {
        int cell = queue[head++];
        for (int a = 0; a < 4; a++) {
            int next = cell + delta[a];
            int kind = env->grid[next];
            if (kind != MAZE_HEDGE && kind != MAZE_TRAP && env->distance[next] < 0) {
                env->distance[next] = env->distance[cell] + 1;
                queue[tail++] = next;
            }
        }
    }
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
    int previous = env->player;
    int next = previous + delta[(int)env->agents[0].actions[0]];
    if (env->grid[next] != MAZE_HEDGE) {
        env->player = next;
    }
    env->tick++;
    int success = env->player == env->goal;
    int died = env->grid[env->player] == MAZE_TRAP;
    float reward = died ? -1 : .02f * (env->distance[previous]
        - env->distance[env->player]) - .001f + success;
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
        "PufferLib Sprite Maze", PUF_STEPS_PER_SEC);
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
