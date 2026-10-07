#pragma once
#include "../connect4cnn/exact_protocol.h"

// Host-side episode receipts, shared with the GPU-free simulation probe.
static uint64_t flappy_exact_word(uint64_t hash, uint32_t word) {
    for (int i = 0; i < 4; i++) hash = (hash ^ ((word >> (8*i)) & 255))*UINT64_C(1099511628211);
    return hash;
}
static uint64_t flappy_exact_float(uint64_t hash, float value) {
    uint32_t word; memcpy(&word,&value,sizeof(word));
    return flappy_exact_word(hash,word);
}
static uint64_t flappy_exact_world_hash(Env* env) {
    uint64_t hash = UINT64_C(14695981039346656037);
    hash = flappy_exact_float(hash,env->bird_y);
    hash = flappy_exact_float(hash,env->bird_vy);
    hash = flappy_exact_float(hash,env->episode_return);
    hash = flappy_exact_word(hash,env->tick);
    hash = flappy_exact_word(hash,env->score);
    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) {
        hash = flappy_exact_float(hash,env->pipes[i].x);
        hash = flappy_exact_float(hash,env->pipes[i].gap_y);
        hash = flappy_exact_word(hash,env->pipes[i].passed);
    }
    return hash;
}
static uint64_t flappy_exact_observation_hash(const obs_t* obs) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int i = 0; i < OBS_SIZE; i++) hash = flappy_exact_float(hash,obs[i]);
    return hash;
}
static void flappy_exact_start(Env* env, uint32_t seed, uint32_t episode) {
    env->rng = c4_exact_env_seed(seed,episode);
    env->log = (Log){0};
    env->agents[0].actions[0] = 0;
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 1;
    puf_reset(env);
}
static int flappy_exact_terminal(Env* env, int decision, float episode_return) {
    return env->log.n == 1 && env->log.episode_length == decision
        && env->log.score >= 0 && env->log.score == floorf(env->log.score)
        && env->log.perf == flappy_clampf(env->log.score/20.0f,0,1)
        && isfinite(env->log.episode_return) && env->log.episode_return == episode_return
        && env->tick == 0 && env->score == 0 && env->episode_return == 0;
}
