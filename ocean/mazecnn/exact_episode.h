#pragma once
#include "../connect4cnn/exact_protocol.h"

// Evaluation-only counters. Original Maze step/reset/level generation is unchanged.
typedef struct {
    uint32_t level;
    int width, height, horizon, decisions, success, end_reason, native_logs;
    float episode_return;
} MazeExactEpisode;

static uint64_t maze_exact_word(uint64_t hash, uint32_t word) {
    for (int i = 0; i < 4; i++) hash = (hash ^ ((word >> (8 * i)) & 255)) * UINT64_C(1099511628211);
    return hash;
}
static uint32_t maze_exact_level(uint32_t seed, uint32_t episode, uint32_t first, uint32_t count) {
    uint32_t rotation = c4_exact_mix((uint64_t)seed ^ UINT64_C(0x6d617a656c657631)) % count;
    return first + ((uint64_t)episode + rotation) % count;
}
static uint64_t maze_exact_world_hash(Env* env) {
    uint64_t h = UINT64_C(14695981039346656037);
    State* s = &env->state;
    h = maze_exact_word(h, s->width); h = maze_exact_word(h, s->height);
    h = maze_exact_word(h, s->spawn_x); h = maze_exact_word(h, s->spawn_y);
    h = maze_exact_word(h, s->x); h = maze_exact_word(h, s->y);
    h = maze_exact_word(h, s->direction); h = maze_exact_word(h, env->tick);
    for (int i = 0; i < MAX_SIZE * MAX_SIZE; i++) h = maze_exact_word(h, s->maze[i]);
    return h;
}
static uint64_t maze_exact_observation_hash(const obs_t* obs) {
    uint64_t h = maze_exact_word(UINT64_C(14695981039346656037), OBS_SIZE);
    for (int i = 0; i < OBS_SIZE; i++) {
#ifdef PUFFER_MAZECNN
        uint32_t word; memcpy(&word, &obs[i], sizeof(word));
#else
        uint32_t word = obs[i];
#endif
        h = maze_exact_word(h, word);
    }
    return h;
}
static int maze_exact_start(Env* env, MazeExactEpisode* e, uint32_t seed,
        uint32_t episode, uint32_t first, uint32_t count) {
    if (!env->levels || env->num_levels < 1 || env->num_levels > 8192
            || !count || (uint64_t)first + count > (uint32_t)env->num_levels) return 0;
    memset(e, 0, sizeof(*e));
    e->level = maze_exact_level(seed, episode, first, count);
    env->state = env->levels[e->level]; env->tick = 0;
    env->rng = c4_exact_env_seed(seed, episode); env->log = (Log){0};
    env->agents[0].actions[0] = 0; env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 1; // Recurrent reset before the first policy action.
    e->width = env->state.width; e->height = env->state.height;
    e->horizon = 2 * e->width * e->height;
    if (e->width < 5 || e->width > MAX_SIZE || e->height != e->width
            || !(e->width & 1) || env->state.x != 1 || env->state.y != 1
            || env->state.direction != 0 || env->state.maze[MAX_SIZE + 1] != AGENT) return 0;
    compute_observations(env);
    return 1;
}
static int maze_exact_step(Env* env, MazeExactEpisode* e, float action) {
    if (!env->levels || env->num_levels < 1 || env->num_levels > 8192 || e->end_reason
            || e->width < 5 || e->width > MAX_SIZE || e->height != e->width
            || e->level >= (uint32_t)env->num_levels || e->decisions < 0
            || e->horizon != 2 * e->width * e->height || e->episode_return != 0
            || e->success != 0 || e->native_logs != 0
            || !isfinite(action) || action < 0 || action > 4 || action != floorf(action)
            || e->decisions != env->tick || e->decisions >= e->horizon || env->log.n != 0
            || env->state.width != e->width || env->state.height != e->height) return 0;
    unsigned int rng = env->rng;
    env->agents[0].actions[0] = action;
    puf_step(env); e->decisions++;
    float reward = env->agents[0].rewards[0], terminal = env->agents[0].terminals[0];
    if ((reward != 0 && reward != 1) || (terminal != 0 && terminal != 1)) return 0;
    e->episode_return += reward;
    if (!terminal) {
        return reward == 0 && env->tick == e->decisions && env->rng == rng && env->log.n == 0;
    }
    e->success = reward == 1; e->end_reason = e->success ? 1 : 2;
    e->native_logs = 1 + (e->success && e->decisions == e->horizon);
    // Validate original reset draws/logs without counting duplicated logs as episodes.
    rand_r(&rng); unsigned int level = rand_r(&rng) % env->num_levels;
    return (e->success || e->decisions == e->horizon) && env->tick == 0 && env->rng == rng
        && env->log.n == e->native_logs && env->log.perf == e->native_logs * reward
        && env->log.score == e->native_logs * reward && env->log.episode_return == e->native_logs * reward
        && env->log.episode_length == e->native_logs * e->decisions
        && e->episode_return == e->success
        && memcmp(&env->state, &env->levels[level], sizeof(State)) == 0;
}
