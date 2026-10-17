#pragma once
#include "../connect4cnn/exact_protocol.h"

// Evaluation-only counters. Shared game/learner behavior is unchanged.
typedef struct {
    int decisions, foods, score, end_reason;
    float episode_return;
} SnakeExactEpisode;

static uint64_t snake_exact_word(uint64_t hash, uint32_t word) {
    for (int i = 0; i < 4; i++) hash = (hash ^ ((word >> (8*i)) & 255))*UINT64_C(1099511628211);
    return hash;
}
static uint64_t snake_exact_float(uint64_t hash, float value) {
    uint32_t word; memcpy(&word, &value, sizeof(word));
    return snake_exact_word(hash, word);
}
static uint64_t snake_exact_world_hash(Env* env) {
    uint64_t hash = UINT64_C(14695981039346656037);
    uint32_t integers[] = {(uint32_t)env->width, (uint32_t)env->height, (uint32_t)env->food,
        (uint32_t)env->max_snake_length, (uint32_t)env->max_steps, (uint32_t)env->length,
        (uint32_t)env->head_ptr, (uint32_t)env->tick, (uint32_t)env->food_collected,
        (uint32_t)env->last_score, (uint32_t)env->last_food, (uint32_t)env->last_decisions,
        (uint32_t)env->last_end_reason, env->rng};
    float values[] = {env->reward_food, env->reward_death, env->episode_return, env->last_return};
    for (size_t i = 0; i < sizeof(integers)/sizeof(integers[0]); i++) hash = snake_exact_word(hash, integers[i]);
    for (size_t i = 0; i < sizeof(values)/sizeof(values[0]); i++) hash = snake_exact_float(hash, values[i]);
    for (int i = 0; i < env->width*env->height; i++) hash = (hash ^ env->grid[i])*UINT64_C(1099511628211);
    for (int i = 0; i < env->max_snake_length; i++) hash = snake_exact_word(hash, (uint32_t)env->snake[i]);
    return hash;
}
static uint64_t snake_exact_observation_hash(const obs_t* obs) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int i = 0; i < OBS_SIZE; i++) hash = snake_exact_float(hash, obs[i]);
    return hash;
}
static void snake_exact_start(Env* env, SnakeExactEpisode* e, uint32_t seed, uint32_t id) {
    *e = (SnakeExactEpisode){0}; e->score = 1;
    env->rng = c4_exact_env_seed(seed, id);
    env->log = (Log){0};
    puf_reset(env);
    // Game reset clears this flag; the actor must reset recurrent state on its first input.
    env->agents[0].terminals[0] = 1;
}
static int snake_exact_valid(Env* env, const SnakeExactEpisode* e) {
    if (e->decisions < 0 || e->decisions > env->max_steps || e->foods < 0
            || e->foods > e->decisions || e->score < 1 || e->end_reason < 0 || e->end_reason > 2
            || !isfinite(e->episode_return) || env->max_steps < 1 || env->max_steps > 16777216
            || env->max_snake_length < 2) return 0;
    int expected = e->foods + 1 < env->max_snake_length ? e->foods + 1 : env->max_snake_length - 1;
    if (e->score != expected) return 0;
    if (e->end_reason) {
        return e->decisions > 0 && (e->end_reason != 2 || e->decisions == env->max_steps)
            && (e->end_reason != 1 || e->foods < e->decisions)
            && env->agents[0].terminals[0] == 1 && env->log.n == 1
            && env->last_score == e->score && env->last_food == e->foods
            && env->last_decisions == e->decisions && env->last_end_reason == e->end_reason
            && env->last_return == e->episode_return && env->log.episode_return == e->episode_return
            && env->log.score == e->score && env->log.food_collected == e->foods
            && env->log.episode_length == e->decisions && env->log.perf == fminf(e->score/120.0f, 1)
            && env->log.deaths == (e->end_reason == 1) && env->log.horizon_ends == (e->end_reason == 2)
            && env->tick == 0 && env->food_collected == 0 && env->length == 1 && env->episode_return == 0;
    }
    return e->decisions < env->max_steps && env->log.n == 0 && env->tick == e->decisions
        && env->food_collected == e->foods && env->length == e->score && env->episode_return == e->episode_return
        && env->last_end_reason == 0 && isfinite(env->agents[0].rewards[0])
        && (e->decisions != 0 || env->agents[0].rewards[0] == 0)
        && env->agents[0].terminals[0] == (e->decisions == 0 ? 1 : 0);
}
static int snake_exact_step(Env* env, SnakeExactEpisode* e, float action) {
    if (e->end_reason || !snake_exact_valid(env, e) || !isfinite(action)
            || action < 0 || action > 3 || action != floorf(action)) return 0;
    env->agents[0].actions[0] = action;
    e->decisions++;
    puf_step(env);
    int ended = env->agents[0].terminals[0] == 1;
    int foods = ended ? env->last_food : env->food_collected;
    int added = foods - e->foods;
    if (added < 0 || added > 1) return 0;
    e->end_reason = ended ? env->last_end_reason : 0;
    float expected_reward = e->end_reason == 1 ? env->reward_death : added ? env->reward_food : 0;
    if ((e->end_reason == 1 && added) || !isfinite(env->agents[0].rewards[0])
            || env->agents[0].rewards[0] != expected_reward) return 0;
    e->foods = foods;
    e->score = foods + 1 < env->max_snake_length ? foods + 1 : env->max_snake_length - 1;
    e->episode_return += env->agents[0].rewards[0];
    return snake_exact_valid(env, e);
}
