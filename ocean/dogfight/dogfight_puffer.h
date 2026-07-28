#pragma once

#include <assert.h>
#include <stdint.h>

#include "curriculum_mix.h"
#include "dogfight_two_agent.h"

#define PUFFER_ENV_GLOBAL_STEP
#define PUFFER_ENV_CURRICULUM

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
    if (env->num_agents == 2) {
        assert(env->agents[1].observations != NULL);
        assert(env->agents[1].actions != NULL);
        assert(env->agents[1].rewards != NULL);
        assert(env->agents[1].terminals != NULL);
        dogfight_two_agent_bind_slots(env);
        return;
    }
    env->observations = (float*)env->agents[0].observations;
    env->actions = env->agents[0].actions;
    env->rewards = env->agents[0].rewards;
    env->terminals = env->agents[0].terminals;
}

void puf_init(Env* env, Dict* kwargs) {
    dogfight_bind_rng(&env->rng);
    env->num_agents = (int)dogfight_dict_get(kwargs, "num_agents", 2);
    assert((env->num_agents == 1 || env->num_agents == 2)
        && "Dogfight supports one legacy slot or two direct slots");

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
    env->agents[1].policy = 0;
    env->agents[1].action_mask = NULL;

    init(env, obs_scheme, &reward_config, curriculum_enabled,
        curriculum_randomize, (int)env->rng);

    float curriculum_target = (float)dogfight_dict_get(
        kwargs, "curriculum_target", 0.9);
    int fixed_stage = (int)dogfight_dict_get(kwargs, "fixed_stage", -1);
    int rehearsal_stage = (int)dogfight_dict_get(
        kwargs, "rehearsal_stage", -1);
    int rehearsal_stride = (int)dogfight_dict_get(
        kwargs, "rehearsal_stride", 0);
    assert(fixed_stage < CURRICULUM_COUNT);
    assert(rehearsal_stage < CURRICULUM_COUNT);
    assert(rehearsal_stride >= 0);
    if (fixed_stage >= 0) {
        fixed_stage = dogfight_select_fixed_stage(
            (uint32_t)env->rng,
            fixed_stage,
            rehearsal_stage,
            rehearsal_stride);
        curriculum_target = (float)fixed_stage;
    }
    set_curriculum_target(env, curriculum_target);

    env->eval_spawn_mode = (int)dogfight_dict_get(
        kwargs, "eval_spawn_mode", 0);
    env->domain_randomization = (float)dogfight_dict_get(
        kwargs, "domain_randomization", 0.0);
    env->vertical_spawn_prob = (float)dogfight_dict_get(
        kwargs, "vertical_spawn_prob", 0.0);
    env->two_agent_reward_version = (int)dogfight_dict_get(
        kwargs, "reward_version", 1);
    assert(env->two_agent_reward_version == 1
        && "Unsupported Dogfight two-agent reward version");
    env->two_agent_role_randomization = (int)dogfight_dict_get(
        kwargs, "role_randomization", 1);
    env->two_agent_player_slot = 0;

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

    // Phase 4 binds slot 1 directly. Explicit one-slot runs retain the
    // internal scripted opponent for curriculum and regression compatibility.
    env->selfplay_active = 0;
    env->use_opponent_override = env->num_agents == 2;
    env->opponent_observations = NULL;
    env->opponent_rewards = NULL;
}

typedef struct {
    int enabled;
    int fixed_stage;
    int max_stage;
    int mastered_stage;
    int min_eval_episodes;
    long warmup_steps;
    long eval_interval;
    long last_eval_step;
    long finalize_at_steps;
    float target;
    float mastery_threshold;
    float last_base_stage_perf;
    double base_stage_kills;
    double base_stage_eps;
    double last_base_stage_eps;
    double observed_base_stage_kills;
    double observed_base_stage_eps;
} PufCurriculumState;

static inline void puf_set_global_step(Env* env, long global_step) {
    env->global_step = global_step;
}

static inline void puf_curriculum_init(
        PufCurriculumState* state, Dict* kwargs, long total_timesteps) {
    memset(state, 0, sizeof(*state));
    state->enabled = (int)dogfight_dict_get(
        kwargs, "curriculum_enabled", 1);
    state->fixed_stage = (int)dogfight_dict_get(
        kwargs, "fixed_stage", -1);
    state->max_stage = (int)dogfight_dict_get(
        kwargs, "max_stage", CURRICULUM_COUNT - 1);
    if (state->max_stage < 0) {
        state->max_stage = 0;
    } else if (state->max_stage >= CURRICULUM_COUNT) {
        state->max_stage = CURRICULUM_COUNT - 1;
    }

    state->target = (float)dogfight_dict_get(
        kwargs, "curriculum_target", 0.9);
    if (state->fixed_stage >= 0) {
        state->target = (float)state->fixed_stage;
    }
    state->target = fminf(
        fmaxf(state->target, 0.0f), (float)state->max_stage);
    state->mastered_stage = -1;
    state->min_eval_episodes = (int)dogfight_dict_get(
        kwargs, "min_eval_episodes", 50);
    state->warmup_steps = (long)dogfight_dict_get(
        kwargs, "warmup_steps", 3000000);
    state->eval_interval = (long)dogfight_dict_get(
        kwargs, "eval_interval", 2500000);
    if (state->eval_interval < 1) {
        state->eval_interval = 1;
    }
    state->last_eval_step = state->warmup_steps;
    state->mastery_threshold = (float)dogfight_dict_get(
        kwargs, "mastery_threshold", 0.90);

    long finalize_margin = (long)dogfight_dict_get(
        kwargs, "finalize_margin", 0);
    state->finalize_at_steps = finalize_margin > 0
        ? total_timesteps - finalize_margin
        : -1;
}

static inline int puf_curriculum_observe(
        PufCurriculumState* state, long global_step,
        double base_stage_kills, double base_stage_eps) {
    if (!state->enabled || state->fixed_stage >= 0 ||
            global_step < state->warmup_steps) {
        return 0;
    }

    state->base_stage_kills += base_stage_kills;
    state->base_stage_eps += base_stage_eps;
    if (global_step - state->last_eval_step < state->eval_interval) {
        return 0;
    }
    state->last_eval_step = global_step;

    float base_stage_perf = state->base_stage_eps > 0.0
        ? (float)(state->base_stage_kills / state->base_stage_eps)
        : 0.0f;
    state->last_base_stage_perf = base_stage_perf;
    state->last_base_stage_eps = state->base_stage_eps;

    int mastery_stage = (int)(state->target + 0.5f);
    if (base_stage_perf >= state->mastery_threshold &&
            state->base_stage_eps >= state->min_eval_episodes &&
            mastery_stage > state->mastered_stage) {
        state->mastered_stage = mastery_stage;
        state->base_stage_kills = 0.0;
        state->base_stage_eps = 0.0;
    }

    int in_finalization = state->finalize_at_steps >= 0 &&
        global_step >= state->finalize_at_steps;
    float new_target;
    if (in_finalization) {
        new_target = state->mastered_stage >= 19
            ? 20.0f
            : (float)state->mastered_stage + 0.01f;
    } else {
        new_target = (float)state->mastered_stage + 0.9f;
    }
    if (state->mastered_stage >= 19) {
        new_target = fmaxf(new_target, 19.0f);
    }
    new_target = fminf(new_target, (float)state->max_stage);
    if (fabsf(state->target - new_target) > 0.01f) {
        state->target = new_target;
    }

    state->base_stage_kills *= 0.9;
    state->base_stage_eps *= 0.9;
    return 1;
}

static inline void puf_curriculum_apply(
        PufCurriculumState* state, Env* envs, int num_envs) {
    for (int i = 0; i < num_envs; i++) {
        set_curriculum_target(&envs[i], state->target);
    }
}

static inline double puf_curriculum_delta(
        double current, double* observed) {
    double tolerance = 1.0e-4 * fmax(1.0, fabs(*observed));
    double delta = current > *observed + tolerance
        ? current - *observed
        : 0.0;
    *observed = fmax(*observed, current);
    return delta;
}

static inline void puf_curriculum_counters_cleared(
        PufCurriculumState* state) {
    state->observed_base_stage_kills = 0.0;
    state->observed_base_stage_eps = 0.0;
}

static inline void puf_curriculum_update(
        PufCurriculumState* state, Env* envs, int num_envs,
        long global_step, Dict* log) {
    double n = dogfight_dict_get(log, "env/n", 0.0);
    double raw_base_stage_kills = dogfight_dict_get(
        log, "env/base_stage_kills", 0.0) * n;
    double raw_base_stage_eps = dogfight_dict_get(
        log, "env/base_stage_eps", 0.0) * n;
    double base_stage_kills = puf_curriculum_delta(
        raw_base_stage_kills, &state->observed_base_stage_kills);
    double base_stage_eps = puf_curriculum_delta(
        raw_base_stage_eps, &state->observed_base_stage_eps);
    int previous_mastered = state->mastered_stage;
    float previous_target = state->target;

    int evaluated = puf_curriculum_observe(
        state, global_step, base_stage_kills, base_stage_eps);
    if (evaluated && state->mastered_stage != previous_mastered) {
        fprintf(stderr,
            "[CURRICULUM] event=mastered stage=%d perf=%.3f eps=%.0f\n",
            state->mastered_stage, state->last_base_stage_perf,
            state->last_base_stage_eps);
    }
    if (evaluated && fabsf(state->target - previous_target) > 0.01f) {
        fprintf(stderr,
            "[CURRICULUM] event=target old=%.2f new=%.2f mastered=%d\n",
            previous_target, state->target, state->mastered_stage);
        puf_curriculum_apply(state, envs, num_envs);
    }
    if (evaluated) {
        fprintf(stderr,
            "[CURRICULUM] step=%ld stage=%.2f base=%.3f(%.0feps) mastered=%d\n",
            global_step, state->target, state->last_base_stage_perf,
            state->last_base_stage_eps, state->mastered_stage);
    }

    dict_set(log, "env/curriculum_target", state->target);
    dict_set(log, "env/mastered_stage", state->mastered_stage);
}

void puf_reset(Env* env) {
    dogfight_bind_rng(&env->rng);
    if (env->num_agents == 2) {
        dogfight_two_agent_select_roles(env);
    } else {
        dogfight_sync_agent_buffers(env);
    }
    c_reset(env);
    if (env->num_agents == 2) {
        compute_opponent_observations(env, env->opponent_observations);
    }
}

void puf_step(Env* env) {
    dogfight_bind_rng(&env->rng);
    dogfight_sync_agent_buffers(env);
    for (int i = 0; i < env->num_agents; i++) {
        *env->agents[i].rewards = 0.0f;
        *env->agents[i].terminals = 0.0f;
    }
    if (env->num_agents == 2) {
        c_step_two_agent(env);
    } else {
        c_step(env);
    }
    if (env->num_agents == 2) {
        if (*env->agents[0].terminals != 0.0f) {
            *env->agents[1].terminals = *env->agents[0].terminals;
        }
        compute_opponent_observations(env, env->opponent_observations);
    }
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
    dict_set(out, "slot_0_score", log->slot_0_score);
    dict_set(out, "slot_1_score", log->slot_1_score);
    dict_set(out, "draw_rate", log->draw_rate);
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
