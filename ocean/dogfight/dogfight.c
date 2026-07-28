#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "dogfight.h"

static void bind_agent(Env* env, obs_t* observations, float* actions,
        float* rewards, float* terminals) {
    env->agents[0].observations = observations;
    env->agents[0].actions = actions;
    env->agents[0].rewards = rewards;
    env->agents[0].terminals = terminals;
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
}

int main(void) {
    srand(42);

    Env env = {
        .num_agents = 1,
        .max_steps = 300,
        .configured_max_steps = 300,
        .rng = 42,
    };
    obs_t observations[OBS_SIZE] = {0};
    float actions[NUM_ATNS] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    bind_agent(&env, observations, actions, rewards, terminals);

    RewardConfig reward_config = {
        .aim_scale = 0.001695f,
        .closing_scale = 0.0001f,
        .neg_g = 0.035f,
        .control_rate_penalty = 0.002f,
        .low_altitude_threshold = 1200.0f,
        .low_altitude_penalty = 0.005f,
        .speed_min = 50.0f,
        .aim_decay_stage = 30.0f,
        .shaping_decay_start = 100000000,
        .shaping_decay_end = 150000000,
        .energy_gain_scale = 0.001f,
        .energy_loss_scale = 0.0005f,
        .energy_advantage_scale = 0.004f,
    };

    init(&env, OBS_OPPONENT_AWARE, &reward_config, 1, 0, 0);
    puf_reset(&env);

    actions[0] = 0.0f;
    actions[1] = 0.0f;
    actions[2] = 0.0f;
    actions[3] = 0.0f;
    actions[4] = -1.0f;

    for (int step = 0; step < 512; step++) {
        puf_step(&env);
        assert(isfinite(rewards[0]));
        for (int i = 0; i < OBS_SIZE; i++) {
            assert(isfinite(observations[i]));
        }
    }

    puf_close(&env);
    puts("dogfight phase1 smoke: ok");
    return 0;
}
