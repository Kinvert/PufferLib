#include <stdlib.h>
#include <string.h>
typedef float obs_t;
#include "pufferenv.h"
#ifdef SPRITE_ATTENTION
#define ACT_SIZES {3, 20, 15}
#define NUM_ATNS 3
#else
#define ACT_SIZES {3}
#define NUM_ATNS 1
#endif
#define PUF_STEPS_PER_SEC 16
#define GRID_WIDTH 20
#define GRID_HEIGHT 15
#define STATE_OBS_SIZE 6
#define SPRITE_RULES "sprite-catch-v4"
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
    int drops;
    int player;
    int object;
    int good;
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
#define SPRITE_OUTSIDE CATCH_EMPTY
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
    int px = env->player % GRID_WIDTH;
    obs[0] = px / (float)GRID_WIDTH;
    obs[1] = (env->object % GRID_WIDTH) / (float)GRID_WIDTH;
    obs[2] = (env->object / GRID_WIDTH) / (float)GRID_HEIGHT;
    obs[3] = env->good;
    obs[4] = (env->object % GRID_WIDTH - px) / (float)GRID_WIDTH;
    obs[5] = env->tick % 2;
}

void catch_spawn(Env* env) {
    env->object_rng = env->object_rng * 1664525u + 1013904223u;
    env->object = GRID_WIDTH + (env->object_rng >> 16) % GRID_WIDTH;
    env->object_rng = env->object_rng * 1664525u + 1013904223u;
    env->good = (env->object_rng >> 16) % 2;
    env->grid[env->object] = env->good ? CATCH_GEM : CATCH_SKULL;
}

void puf_init(Env* env, Dict* kwargs) {
    env->rng += (unsigned int)dict_get(kwargs, "seed");
    env->observation_mode = dict_get(kwargs, "observation_mode");
#ifdef SPRITE_ATTENTION
    env->partial_observation = dict_get(kwargs, "partial_observation");
#endif
    env->representation = dict_get(kwargs, "representation");
    env->drops = dict_get(kwargs, "drops");
    env->num_agents = 1;
    env->agents[0].policy = 0;
    if (env->observation_mode) {
        sprite_make_tiles(env->representation);
        memset(env->pixel_grid, 255, sizeof(env->pixel_grid));
    }
}

void puf_reset(Env* env) {
    // One world draw per episode; early death cannot skip the next world's RNG.
    env->rng = env->rng * 1664525u + 1013904223u;
    env->object_rng = env->rng ^ 0x9e3779b9u;
    env->object_rng ^= env->object_rng >> 16;
    env->object_rng *= 0x85ebca6bu;
    env->floor_seed = env->rng ^ 0xd168aaadu;
    memset(env->grid, 0, sizeof(env->grid));
    for (int x = 0; x < GRID_WIDTH; x++) {
        env->grid[(GRID_HEIGHT - 1) * GRID_WIDTH + x] = CATCH_FLOOR;
    }
    env->player = (GRID_HEIGHT - 2) * GRID_WIDTH + GRID_WIDTH / 2;
    env->tick = 0;
    env->count = 0;
    env->correct = 0;
    env->episode_return = 0;
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 0;
    catch_spawn(env);
#ifdef SPRITE_ATTENTION
    env->gaze_idx = env->player;
#endif
    compute_observations(env);
}

void puf_step(Env* env) {
#ifdef SPRITE_ATTENTION
    sprite_attention_action(env);
#endif
    int action = env->agents[0].actions[0];
    int old_player = env->player, old_object = env->object;
    int px = env->player % GRID_WIDTH;
    int x = env->object % GRID_WIDTH, y = env->object / GRID_WIDTH;
    int before = abs(x - px);
    px += (action == 2) - (action == 1);
    if (px < 0) {
        px = 0;
    }
    if (px >= GRID_WIDTH) {
        px = GRID_WIDTH - 1;
    }
    env->player = (GRID_HEIGHT - 2) * GRID_WIDTH + px;
    float progress = .02f * (before - abs(x - px));
    float reward = env->good ? progress : -progress;
    env->grid[old_object] = CATCH_EMPTY;
    env->tick++;
    y += env->tick % 2 == 0;
    env->agents[0].terminals[0] = 0;
    int died = 0;
    if (y == GRID_HEIGHT - 2) {
        int caught = x == px;
        int correct = caught == env->good;
        died = caught && !env->good;
        env->correct += correct;
        env->count++;
        reward += correct ? 1 : -1;
        if (!died && env->count < env->drops) {
            catch_spawn(env);
        }
    } else {
        env->object = y * GRID_WIDTH + x;
        env->grid[env->object] = env->good ? CATCH_GEM : CATCH_SKULL;
    }
    env->episode_return += reward;
    if (died || env->count == env->drops) {
        float perf = env->correct / (float)env->drops;
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
        float* obs = env->agents[0].observations;
        int dirty[] = {old_player, env->player, old_object, env->object};
        for (int i = 0; i < 4; i++) {
            sprite_write_cell(obs, env->grid, env->player, dirty[i], env->representation,
                env->floor_seed, env->pixel_grid);
        }
    } else {
        compute_observations(env);
    }
    env->agents[0].rewards[0] = reward;
}

void puf_render(Env* env) {
    sprite_render(&env->client, env->grid, env->player, env->representation, env->floor_seed,
        "PufferLib Sprite Catch", PUF_STEPS_PER_SEC);
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
