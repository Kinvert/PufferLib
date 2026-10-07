#pragma once
#include "../connect4cnn/exact_protocol.h"

// Evaluation only. Training retains the original puf_step frame-skip/reset path.
typedef struct {
    int decisions;
    int frames;
    int score;
    float episode_return;
    int completed;
    int capped;
} BreakoutExactEpisode;

static uint64_t breakout_exact_word(uint64_t hash, uint32_t word) {
    for (int i = 0; i < 4; i++) hash = (hash ^ ((word >> (8*i)) & 255))*UINT64_C(1099511628211);
    return hash;
}

static uint64_t breakout_exact_float(uint64_t hash, float value) {
    uint32_t word; memcpy(&word, &value, sizeof(word));
    return breakout_exact_word(hash, word);
}

static uint64_t breakout_exact_world_hash(Env* env) {
    uint64_t hash = UINT64_C(14695981039346656037);
    // Explicit fields: independent of pointers, padding, raster and policy shape.
    int integers[] = {env->score, env->balls_fired, env->hits, env->width,
        env->height, env->num_bricks, env->brick_rows, env->brick_cols,
        env->ball_width, env->ball_height, env->brick_width, env->brick_height,
        env->num_balls, env->max_score, env->half_max_score, env->tick,
        env->frameskip, env->hit_brick, env->continuous};
    float values[] = {env->paddle_x, env->paddle_y, env->ball_x, env->ball_y,
        env->ball_vx, env->ball_vy, env->initial_paddle_width, env->paddle_width,
        env->paddle_height, env->paddle_speed, env->ball_speed,
        env->initial_ball_speed, env->max_ball_speed};
    for (size_t i = 0; i < sizeof(integers)/sizeof(integers[0]); i++)
        hash = breakout_exact_word(hash, integers[i]);
    for (size_t i = 0; i < sizeof(values)/sizeof(values[0]); i++)
        hash = breakout_exact_float(hash, values[i]);
    hash = breakout_exact_word(hash, env->rng);
    for (int i = 0; i < env->num_bricks; i++) {
        hash = breakout_exact_float(hash, env->brick_x[i]);
        hash = breakout_exact_float(hash, env->brick_y[i]);
        hash = breakout_exact_float(hash, env->brick_states[i]);
    }
    return hash;
}

static uint64_t breakout_exact_observation_hash(const obs_t* obs) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int i = 0; i < OBS_SIZE; i++) hash = breakout_exact_float(hash, obs[i]);
    return hash;
}

static void breakout_exact_start(Env* env, BreakoutExactEpisode* episode,
        uint32_t seed, uint32_t id) {
    *episode = (BreakoutExactEpisode){0};
    env->rng = c4_exact_env_seed(seed, id);
    env->log = (Log){0};
    env->agents[0].actions[0] = 0;
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 1;
    puf_reset(env);
}

static int breakout_exact_valid(Env* env, const BreakoutExactEpisode* episode,
        int frame_cap) {
    if (frame_cap <= 0 || frame_cap > 16777216 || env->frameskip <= 0 || env->max_score <= 0
            || episode->frames <= 0 || episode->frames > frame_cap
            || episode->decisions <= 0
            || episode->decisions != 1 + (episode->frames - 1)/env->frameskip
            || episode->score < 0 || episode->score > env->max_score
            || !isfinite(episode->episode_return)
            || episode->episode_return != episode->score
            || (episode->completed != 0 && episode->completed != 1)
            || (episode->capped != 0 && episode->capped != 1)
            || (episode->completed && episode->capped)) return 0;
    if (episode->completed) {
        return env->agents[0].terminals[0] == 1 && env->log.n == 1
            && env->log.score == episode->score
            && env->log.episode_return == episode->episode_return
            && env->log.episode_length == episode->frames
            && env->log.perf == episode->score/(float)env->max_score
            && env->tick == 0 && env->score == 0 && env->balls_fired == 0;
    }
    return env->agents[0].terminals[0] == 0 && env->log.n == 0
        && env->tick == episode->frames && env->score == episode->score
        && episode->capped == (episode->frames == frame_cap);
}

// Returns false on invalid inputs/accounting. Never steps a finished episode.
// A natural terminal on the cap frame takes precedence over administrative cap.
static int breakout_exact_step(Env* env, BreakoutExactEpisode* episode,
        float action, int frame_cap) {
    if (episode->completed || episode->capped || episode->frames < 0
            || frame_cap <= 0 || frame_cap > 16777216
            || episode->frames >= frame_cap || env->frameskip <= 0 || env->max_score <= 0
            || !isfinite(action) || action < NOOP || action > RIGHT
            || action != floorf(action)) return 0;
    env->agents[0].actions[0] = action;
    env->agents[0].terminals[0] = 0;
    env->agents[0].rewards[0] = 0;
    episode->decisions++;
    for (int i = 0; i < env->frameskip && episode->frames < frame_cap; i++) {
        env->tick++;
        episode->frames++;
        step_frame(env, action);
        if (env->agents[0].terminals[0]) {
            episode->completed = 1;
            break;
        }
    }
    episode->episode_return += env->agents[0].rewards[0];
    if (episode->completed && (!isfinite(env->log.score) || env->log.score < 0
            || env->log.score > env->max_score || env->log.score != floorf(env->log.score))) return 0;
    episode->score = episode->completed ? (int)env->log.score : env->score;
    episode->capped = !episode->completed && episode->frames == frame_cap;
    compute_observations(env);
    return breakout_exact_valid(env, episode, frame_cap);
}
