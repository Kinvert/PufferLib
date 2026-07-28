#pragma once

#include <assert.h>

static inline double dogfight_dict_get(
        Dict* kwargs, const char* key, double fallback) {
    for (int i = 0; i < kwargs->size; i++) {
        if (strcmp(kwargs->items[i].key, key) == 0) {
            return kwargs->items[i].value;
        }
    }
    return fallback;
}

static inline void dogfight_sync_agent_buffers(Env* env) {
    if (env->agents[0].observations == NULL) {
        return;
    }
    env->observations = (float*)env->agents[0].observations;
    env->actions = env->agents[0].actions;
    env->rewards = env->agents[0].rewards;
    env->terminals = env->agents[0].terminals;
}

void puf_init(Env* env, Dict* kwargs) {
    dogfight_bind_rng(&env->rng);
    env->num_agents = (int)dogfight_dict_get(kwargs, "num_agents", 1);
    assert(env->num_agents == 1 && "Phase 1 Dogfight supports num_agents=1");

    env->max_steps = (int)dogfight_dict_get(kwargs, "max_steps", 300);
    env->configured_max_steps = env->max_steps;
    int obs_scheme = (int)dogfight_dict_get(
        kwargs, "obs_scheme", OBS_OPPONENT_AWARE);
    int curriculum_enabled = (int)dogfight_dict_get(
        kwargs, "curriculum_enabled", 1);
    int curriculum_randomize = (int)dogfight_dict_get(
        kwargs, "curriculum_randomize", 0);

    RewardConfig reward_config = {
        .aim_scale = (float)dogfight_dict_get(
            kwargs, "reward_aim_scale", 0.001695),
        .closing_scale = (float)dogfight_dict_get(
            kwargs, "reward_closing_scale", 0.0001),
        .neg_g = (float)dogfight_dict_get(kwargs, "penalty_neg_g", 0.035),
        .control_rate_penalty = (float)dogfight_dict_get(
            kwargs, "control_rate_penalty", 0.002),
        .low_altitude_threshold = (float)dogfight_dict_get(
            kwargs, "low_altitude_threshold", 1200.0),
        .low_altitude_penalty = (float)dogfight_dict_get(
            kwargs, "low_altitude_penalty", 0.005),
        .speed_min = (float)dogfight_dict_get(kwargs, "speed_min", 50.0),
        .aim_decay_stage = (float)dogfight_dict_get(
            kwargs, "aim_decay_stage", 30.0),
        .shaping_decay_start = (long)dogfight_dict_get(
            kwargs, "shaping_decay_start", 100000000.0),
        .shaping_decay_end = (long)dogfight_dict_get(
            kwargs, "shaping_decay_end", 150000000.0),
        .energy_gain_scale = (float)dogfight_dict_get(
            kwargs, "energy_gain_scale", 0.001),
        .energy_loss_scale = (float)dogfight_dict_get(
            kwargs, "energy_loss_scale", 0.0005),
        .energy_advantage_scale = (float)dogfight_dict_get(
            kwargs, "energy_advantage_scale", 0.004),
    };

    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
    env->agents[1].policy = 1;
    env->agents[1].action_mask = NULL;

    init(env, obs_scheme, &reward_config, curriculum_enabled,
        curriculum_randomize, (int)env->rng);

    env->eval_spawn_mode = (int)dogfight_dict_get(
        kwargs, "eval_spawn_mode", 0);
    env->domain_randomization = (float)dogfight_dict_get(
        kwargs, "domain_randomization", 0.0);
    env->vertical_spawn_prob = (float)dogfight_dict_get(
        kwargs, "vertical_spawn_prob", 0.0);

    int recovery_enabled = (int)dogfight_dict_get(
        kwargs, "recovery_enabled", 1);
    env->recovery_altitude_threshold = recovery_enabled
        ? (float)dogfight_dict_get(
            kwargs, "recovery_altitude_threshold", 500.0)
        : -9999.0f;
    env->recovery_trigger_prob = (float)dogfight_dict_get(
        kwargs, "recovery_trigger_prob", 0.067);
    env->recovery_speed_threshold = (float)dogfight_dict_get(
        kwargs, "recovery_speed_threshold", 70.0);
    env->recovery_bank_deg = (float)dogfight_dict_get(
        kwargs, "recovery_bank_deg", 60.0);

    // Phase 1 always uses the internal scripted opponent.
    env->selfplay_active = 0;
    env->use_opponent_override = 0;
    env->opponent_observations = NULL;
    env->opponent_rewards = NULL;
}

void puf_reset(Env* env) {
    dogfight_bind_rng(&env->rng);
    dogfight_sync_agent_buffers(env);
    c_reset(env);
}

void puf_step(Env* env) {
    dogfight_bind_rng(&env->rng);
    dogfight_sync_agent_buffers(env);
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    c_step(env);
}

void puf_render(Env* env) {
    dogfight_sync_agent_buffers(env);
    c_render(env);
}

void puf_close(Env* env) {
    c_close(env);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "shots_fired", log->shots_fired);
    dict_set(out, "accuracy", log->accuracy);
    dict_set(out, "stage", log->stage);
    dict_set(out, "avg_stage_weight", log->total_stage_weight);
    dict_set(out, "avg_abs_bias", log->total_abs_bias);
    dict_set(out, "avg_stage", log->stage_sum);
    dict_set(out, "avg_control_rate", log->total_control_rate);
    dict_set(out, "base_stage_kills", log->base_stage_kills);
    dict_set(out, "base_stage_eps", log->base_stage_eps);
    dict_set(out, "player_ground", log->player_ground_hits);
    dict_set(out, "opponent_ground", log->opponent_ground_hits);
    dict_set(out, "recovery_triggers", log->recovery_triggers);
    dict_set(out, "clean_fights", log->clean_fights);
    dict_set(out, "altitude_kills", log->altitude_kills);
    dict_set(out, "n", log->n);
}
