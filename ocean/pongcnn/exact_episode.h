#pragma once
#include "../connect4cnn/exact_protocol.h"

// Host evaluation receipts. Original puf_step/training/game rules are unchanged.
typedef struct {
    int decisions;
    int rally_decisions;
    unsigned int right;
    unsigned int left;
    float episode_return;
    int completed;
    int capped;
} PongExactEpisode;

static uint64_t pong_exact_word(uint64_t hash, uint32_t word) {
    for (int i = 0; i < 4; i++) hash = (hash ^ ((word >> (8*i)) & 255))*UINT64_C(1099511628211);
    return hash;
}
static uint64_t pong_exact_float(uint64_t hash, float value) {
    uint32_t word; memcpy(&word, &value, sizeof(word));
    return pong_exact_word(hash, word);
}
static uint64_t pong_exact_world_hash(Env* env) {
    uint64_t hash = UINT64_C(14695981039346656037);
    float values[] = {env->paddle_yl, env->paddle_yr, env->ball_x, env->ball_y,
        env->ball_vx, env->ball_vy, env->width, env->height, env->paddle_width,
        env->paddle_height, env->ball_width, env->ball_height, env->paddle_speed,
        env->ball_initial_speed_x, env->ball_initial_speed_y, env->ball_max_speed_y,
        env->ball_speed_y_increment, env->min_paddle_y, env->max_paddle_y, env->paddle_dir};
    uint32_t integers[] = {env->score_l, env->score_r, env->max_score,
        (uint32_t)env->tick, (uint32_t)env->n_bounces, (uint32_t)env->win,
        (uint32_t)env->frameskip, (uint32_t)env->continuous, env->rng};
    for (size_t i = 0; i < sizeof(values)/sizeof(values[0]); i++) hash = pong_exact_float(hash, values[i]);
    for (size_t i = 0; i < sizeof(integers)/sizeof(integers[0]); i++) hash = pong_exact_word(hash, integers[i]);
    return hash;
}
static uint64_t pong_exact_observation_hash(const obs_t* obs) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int i = 0; i < OBS_SIZE; i++) hash = pong_exact_float(hash, obs[i]);
    return hash;
}
static void pong_exact_start(Env* env, PongExactEpisode* episode, uint32_t seed, uint32_t id) {
    *episode = (PongExactEpisode){0};
    env->rng = c4_exact_env_seed(seed, id);
    env->log = (Log){0};
    env->paddle_dir = 0; env->win = 0;
    env->agents[0].actions[0] = 0;
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 1;
    puf_reset(env);
}
static int pong_exact_valid(Env* env, const PongExactEpisode* e, int cap) {
    if (cap <= 0 || cap > 16777216 || e->decisions <= 0 || e->decisions > cap
            || e->rally_decisions < 0 || e->rally_decisions > e->decisions
            || env->max_score == 0 || env->max_score > 1000000
            || e->right > env->max_score || e->left > env->max_score
            || e->right + e->left > (unsigned)e->decisions
            || (e->right + e->left == 0 && e->rally_decisions != e->decisions)
            || !isfinite(e->episode_return) || e->episode_return != (float)e->right - (float)e->left
            || (e->completed != 0 && e->completed != 1) || (e->capped != 0 && e->capped != 1)
            || (e->completed && e->capped)) return 0;
    if (e->completed) {
        return (e->right == env->max_score) != (e->left == env->max_score)
            && e->decisions - e->rally_decisions >= (int)(e->right + e->left) - 1
            && env->agents[0].terminals[0] == 1 && env->log.n == 1
            && env->log.score == e->episode_return && env->log.episode_return == e->episode_return
            && e->rally_decisions > 0 && env->log.episode_length == e->rally_decisions
            && env->log.perf == (float)e->right / ((float)e->right + (float)e->left)
            && env->score_l == 0 && env->score_r == 0 && env->tick == 0;
    }
    return e->right < env->max_score && e->left < env->max_score
        && e->decisions - e->rally_decisions >= (int)(e->right + e->left)
        && env->agents[0].terminals[0] == 0 && env->log.n == 0
        && env->score_r == e->right && env->score_l == e->left
        && env->tick == e->rally_decisions && e->capped == (e->decisions == cap);
}
static int pong_exact_step(Env* env, PongExactEpisode* e, float action, int cap) {
    if (e->completed || e->capped || e->decisions < 0 || e->decisions >= cap
            || cap <= 0 || cap > 16777216 || env->max_score == 0 || env->max_score > 1000000
            || env->continuous != 0 || env->frameskip <= 0
            || env->score_l != e->left || env->score_r != e->right
            || e->rally_decisions < 0 || e->rally_decisions > e->decisions || env->tick != e->rally_decisions
            || e->right >= env->max_score || e->left >= env->max_score
            || e->right + e->left > (unsigned)e->decisions || env->log.n != 0
            || !isfinite(e->episode_return) || e->episode_return != (float)e->right - (float)e->left
            || !isfinite(action) || action < 0 || action > 2 || action != floorf(action)) return 0;
    env->agents[0].actions[0] = action;
    e->decisions++; e->rally_decisions++;
    puf_step(env);
    float reward = env->agents[0].rewards[0];
    if (reward != -1 && reward != 0 && reward != 1) return 0;
    e->right += reward == 1; e->left += reward == -1;
    e->episode_return += reward;
    e->completed = env->agents[0].terminals[0] == 1;
    e->capped = !e->completed && e->decisions == cap;
    if (reward != 0 && !e->completed) e->rally_decisions = 0;
    return pong_exact_valid(env, e, cap);
}

// Deterministic possible-final-fraction bounds, not confidence intervals.
// Point counts, not float32 native perf, define the rational score metric.
static int pong_exact_bounds(unsigned int right, unsigned int left, unsigned int target,
        int completed, double* lower, double* upper) {
    if (target == 0 || target > 1000000 || right > target || left > target
            || (completed != 0 && completed != 1)) return 0;
    if (completed) {
        if ((right == target) == (left == target)) return 0;
        *lower = *upper = (double)right / (right + left);
    } else {
        if (right >= target || left >= target) return 0;
        *lower = (double)right / (target + right);
        *upper = (double)target / (target + left);
    }
    return 1;
}
