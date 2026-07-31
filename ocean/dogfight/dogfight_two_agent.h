#pragma once

// Native direct two-agent dynamics. This path intentionally does not call the
// scripted-opponent, recovery, or legacy self-play branches in c_step().

static inline void dogfight_two_agent_combine_dense_rewards(
        int reward_version,
        const float perspective_rewards[2],
        float published_rewards[2]) {
    if (reward_version == 0 || reward_version == 1) {
        published_rewards[0] = perspective_rewards[0];
        published_rewards[1] = perspective_rewards[1];
        return;
    }
    assert(reward_version == 2);
    float advantage = clampf(
        0.5f * (perspective_rewards[0] - perspective_rewards[1]),
        -1.0f, 1.0f);
    published_rewards[0] = advantage;
    published_rewards[1] = -advantage;
}

static inline void dogfight_two_agent_bind_slots(Dogfight* env) {
    int player_slot = env->two_agent_player_slot;
    int opponent_slot = 1 - player_slot;
    env->observations = (float*)env->agents[player_slot].observations;
    env->actions = env->agents[player_slot].actions;
    env->rewards = env->agents[player_slot].rewards;
    env->terminals = env->agents[player_slot].terminals;
    env->opponent_observations =
        (float*)env->agents[opponent_slot].observations;
    env->opponent_rewards = env->agents[opponent_slot].rewards;
    memcpy(env->opponent_actions_override,
        env->agents[opponent_slot].actions,
        NUM_ATNS * sizeof(float));
    env->use_opponent_override = 1;
}

static inline void dogfight_two_agent_select_roles(Dogfight* env) {
    env->two_agent_scripted_episode =
        env->two_agent_bootstrap_steps > 0
        && env->global_step < env->two_agent_bootstrap_steps;
    env->two_agent_player_slot = env->two_agent_scripted_episode
        ? 0
        : (env->two_agent_role_randomization
        ? (int)(dogfight_rand() & 1U)
        : 0);
    dogfight_two_agent_bind_slots(env);
}

static inline void dogfight_two_agent_pursuit_teacher_actions(
        Dogfight* env,
        Plane* self,
        Plane* target,
        float actions[NUM_ATNS]) {
    // Bootstrap must teach the policy how to fly toward its observation
    // target. The curriculum opponent's stage behavior is not a valid teacher:
    // at stage 0 it only flies straight, so its controls contain no
    // target-conditioned left/right signal.
    execute_pursuit_pure(
        &env->opponent_ap,
        &env->opponent_ace,
        self,
        target,
        actions);
}

static inline void dogfight_two_agent_scripted_actions(
        Dogfight* env, float actions[NUM_ATNS]) {
    dogfight_two_agent_pursuit_teacher_actions(
        env, &env->opponent, &env->player, actions);
}

static inline float dogfight_two_agent_imitation_reward(
        const float policy_actions[NUM_ATNS],
        const float teacher_actions[NUM_ATNS],
        float scale) {
    // Teach flight controls, not weapons policy. Pure pursuit intentionally
    // leaves the trigger off, and imitating that would fight the kill reward.
    const int num_flight_controls = 4;
    float mse = 0.0f;
    for (int action = 0; action < num_flight_controls; action++) {
        float error = policy_actions[action] - teacher_actions[action];
        mse += error * error;
    }
    return -scale * mse / (float)num_flight_controls;
}

static inline float dogfight_two_agent_flight_school_reward(
        const float policy_actions[NUM_ATNS],
        const float teacher_actions[NUM_ATNS],
        float scale) {
    // Teach one contextual control at a time. Averaging all four controls let
    // the policy improve its score through easy throttle/elevator imitation
    // while retaining the target-independent aileron bias visible in flight.
    float aileron_error = policy_actions[2] - teacher_actions[2];
    return clampf(
        scale * (1.0f - 0.5f * aileron_error * aileron_error),
        -1.0f, 1.0f);
}

static inline float dogfight_two_agent_steering_alignment_reward(
        float observed_target_azimuth,
        float aileron,
        float scale) {
    float desired_aileron = observed_target_azimuth < -1.0e-4f
        ? 1.0f
        : (observed_target_azimuth > 1.0e-4f ? -1.0f : 0.0f);
    return scale * desired_aileron * clampf(aileron, -1.0f, 1.0f);
}

static inline float dogfight_two_agent_steering_scale_for_stage(
        float configured_scale, int stage) {
    float difficulty = clampf((float)stage / 10.0f, 0.0f, 1.0f);
    return configured_scale * (1.0f - difficulty);
}

static inline float dogfight_two_agent_publish_steering_reward(
        float competitive_reward,
        float observed_target_azimuth,
        float aileron,
        float steering_scale) {
    return clampf(
        competitive_reward + dogfight_two_agent_steering_alignment_reward(
            observed_target_azimuth, aileron, steering_scale),
        -1.0f,
        1.0f);
}

static inline float dogfight_two_agent_update_roll_travel(
        float previous_radians, float roll_rate, float dt) {
    float updated = previous_radians + fabsf(roll_rate) * dt;
    float limit = 4.0f * 2.0f * (float)M_PI;
    return clampf(updated, 0.0f, limit);
}

static inline float dogfight_two_agent_roll_stage_scale(int stage) {
    if (stage <= 2) return 1.0f;
    if (stage >= 6) return 0.0f;
    return (6.0f - (float)stage) / 4.0f;
}

static inline float dogfight_two_agent_roll_time_scale(
        long global_step,
        long decay_start,
        long decay_end,
        float final_fraction) {
    if (global_step <= decay_start) return 1.0f;
    if (global_step >= decay_end) return final_fraction;
    float progress = (float)(global_step - decay_start)
        / (float)(decay_end - decay_start);
    return 1.0f - progress * (1.0f - final_fraction);
}

static inline float dogfight_two_agent_roll_discipline_penalty(
        float travel_radians,
        float configured_scale,
        float free_rotations,
        float full_rotations,
        int stage,
        long global_step,
        long decay_start,
        long decay_end,
        float final_fraction) {
    if (configured_scale <= 0.0f) return 0.0f;
    float rotations = travel_radians / (2.0f * (float)M_PI);
    if (rotations <= free_rotations) return 0.0f;
    float severity = clampf(
        (rotations - free_rotations)
            / (full_rotations - free_rotations),
        0.0f,
        1.0f);
    float stage_scale = dogfight_two_agent_roll_stage_scale(stage);
    float time_scale = dogfight_two_agent_roll_time_scale(
        global_step, decay_start, decay_end, final_fraction);
    return -configured_scale * stage_scale * time_scale
        * severity * severity;
}

static inline float dogfight_two_agent_publish_roll_discipline(
        float competitive_reward,
        float travel_radians,
        const Dogfight* env) {
    float penalty = dogfight_two_agent_roll_discipline_penalty(
        travel_radians,
        env->native_roll_discipline_scale,
        env->native_roll_free_rotations,
        env->native_roll_full_rotations,
        env->stage,
        env->global_step,
        env->native_roll_decay_start,
        env->native_roll_decay_end,
        env->native_roll_final_fraction);
    return clampf(competitive_reward + penalty, -1.0f, 1.0f);
}

static inline float dogfight_two_agent_signed_bank(const Plane* plane) {
    Vec3 up = quat_rotate(plane->ori, vec3(0.0f, 0.0f, 1.0f));
    Vec3 body_y = quat_rotate(plane->ori, vec3(0.0f, 1.0f, 0.0f));
    float magnitude = acosf(clampf(up.z, -1.0f, 1.0f));
    return body_y.z >= 0.0f ? magnitude : -magnitude;
}

static inline float dogfight_two_agent_bank_guidance_penalty(
        const Plane* self,
        const Plane* other,
        float bank_scale,
        float roll_scale) {
    if (bank_scale <= 0.0f && roll_scale <= 0.0f) return 0.0f;

    const float max_roll_rate = 60.0f * DEG_TO_RAD;
    const float roll_gain = 2.0f;
    Quat inverse = {
        self->ori.w,
        -self->ori.x,
        -self->ori.y,
        -self->ori.z,
    };
    Vec3 relative_body = quat_rotate(
        inverse, sub3(other->pos, self->pos));
    float cross_track = sqrtf(
        relative_body.y * relative_body.y
            + relative_body.z * relative_body.z);
    float range = sqrtf(
        relative_body.x * relative_body.x
            + cross_track * cross_track);
    float off_axis = range > 1.0e-6f ? cross_track / range : 0.0f;
    float alignment_weight = clampf(
        off_axis / sinf(5.0f * DEG_TO_RAD), 0.0f, 1.0f);
    float alignment_error = cross_track > 1.0e-6f
        ? acosf(clampf(relative_body.z / cross_track, -1.0f, 1.0f))
        : 0.0f;
    float signed_alignment_error = relative_body.y > 1.0e-6f
        ? alignment_error
        : (relative_body.y < -1.0e-6f ? -alignment_error : 0.0f);
    // Body +Y is left and positive omega.x rolls right. Roll through the
    // shortest direction until the target projection is on body +Z, then
    // damp roll instead of continuing around another revolution.
    float desired_roll_rate = clampf(
        -roll_gain * signed_alignment_error * alignment_weight,
        -max_roll_rate,
        max_roll_rate);
    float alignment_error_norm =
        alignment_error / (float)M_PI;
    float roll_error_norm = clampf(
        (desired_roll_rate - self->omega.x)
            / (2.0f * max_roll_rate),
        -1.0f,
        1.0f);
    return -bank_scale * alignment_weight
            * alignment_error_norm * alignment_error_norm
        - roll_scale * roll_error_norm * roll_error_norm;
}

static inline float dogfight_two_agent_publish_bank_guidance(
        float competitive_reward,
        const Plane* self,
        const Plane* other,
        const Dogfight* env) {
    float schedule = dogfight_two_agent_roll_stage_scale(env->stage)
        * dogfight_two_agent_roll_time_scale(
            env->global_step,
            env->native_roll_decay_start,
            env->native_roll_decay_end,
            env->native_roll_final_fraction);
    float penalty = schedule * dogfight_two_agent_bank_guidance_penalty(
        self,
        other,
        env->native_bank_guidance_scale,
        env->native_roll_guidance_scale);
    return clampf(competitive_reward + penalty, -1.0f, 1.0f);
}

static inline bool dogfight_two_agent_in_native_acquisition(
        const Dogfight* env) {
    if (!env->native_spawn_curriculum) {
        return false;
    }
    // Acquisition is a current-policy training task, not a match outcome.
    // Native frozen-bank training and final pool evaluation assign distinct
    // policy rows, so they must run real combat even when global_step starts
    // inside the acquisition window.
    if (env->agents[0].policy != env->agents[1].policy) {
        return false;
    }
    if (env->native_acquisition_steps > 0
            && env->global_step < env->native_acquisition_steps) {
        return true;
    }
    if (env->native_acquisition_rehearsal_cycle_steps <= 0
            || env->native_acquisition_rehearsal_steps <= 0
            || env->native_acquisition_rehearsal_steps
                >= env->native_acquisition_rehearsal_cycle_steps) {
        return false;
    }
    long selfplay_step = env->global_step - env->native_acquisition_steps;
    if (selfplay_step < 0) {
        return false;
    }
    long cycle_step =
        selfplay_step % env->native_acquisition_rehearsal_cycle_steps;
    return cycle_step >= env->native_acquisition_rehearsal_cycle_steps
        - env->native_acquisition_rehearsal_steps;
}

static inline float dogfight_two_agent_native_acquisition_reward(
        float observed_target_azimuth,
        const float actions[NUM_ATNS],
        float scale,
        float neutral_control_scale) {
    // This is an environment reward, not imitation: no policy, controller,
    // trajectory, or action label is queried. Mirrored target geometry defines
    // the control objective directly. One-decision episodes make its credit
    // causal before normal competitive self-play starts. Aileron remains the
    // dominant contextual target, while neutral throttle/elevator/rudder raw
    // inputs anchor the other control means to stable half-throttle flight.
    float desired_aileron = observed_target_azimuth < -1.0e-4f
        ? 1.0f
        : (observed_target_azimuth > 1.0e-4f ? -1.0f : 0.0f);
    float aileron_error =
        clampf(actions[2], -1.0f, 1.0f) - desired_aileron;
    float neutral_control_penalty =
        actions[0] * actions[0]
        + actions[1] * actions[1]
        + actions[3] * actions[3];
    return clampf(
        scale * (
            1.0f
            - 0.5f * aileron_error * aileron_error
            - neutral_control_scale * neutral_control_penalty),
        -1.0f,
        1.0f);
}

static inline float dogfight_two_agent_dense_reward(
        Dogfight* env,
        const Plane* self,
        const Plane* other,
        const float* actions,
        int slot,
        float self_energy,
        float other_energy) {
    Vec3 rel_pos = sub3(other->pos, self->pos);
    float dist = norm3(rel_pos);
    Vec3 rel_pos_norm = normalize3(rel_pos);
    Vec3 rel_vel = sub3(self->vel, other->vel);
    float shaping_decay = calc_shaping_decay(
        env->global_step,
        env->rcfg.shaping_decay_start,
        env->rcfg.shaping_decay_end);

    float reward = clampf(
        dot3(rel_vel, rel_pos_norm) * env->rcfg.closing_scale,
        -0.05f, 0.05f) * shaping_decay;

    Vec3 forward = quat_rotate(self->ori, vec3(1, 0, 0));
    float aim_dot = dot3(rel_pos_norm, forward);
    if (dist < GUN_RANGE * 2.0f) {
        reward += (aim_dot + 1.0f) * 0.5f
            * env->rcfg.aim_scale * shaping_decay;
    }

    float g_deficit = fmaxf(0.0f, 0.5f - self->g_force);
    reward -= g_deficit * env->rcfg.neg_g;

    float speed = norm3(self->vel);
    if (speed < env->rcfg.speed_min) {
        reward -= (env->rcfg.speed_min - speed) * PENALTY_STALL;
    }
    reward -= fabsf(actions[3]) * PENALTY_RUDDER;
    reward -= fabsf(actions[2]) * env->rcfg.aileron_magnitude_penalty;

    float d_e = actions[1] - env->two_agent_prev_controls[slot][0];
    float d_a = actions[2] - env->two_agent_prev_controls[slot][1];
    float d_r = actions[3] - env->two_agent_prev_controls[slot][2];
    float delta_sq = d_e*d_e + d_a*d_a + d_r*d_r;
    env->two_agent_pool_control_rate_sum[slot] += delta_sq;
    env->episode_control_rate += 0.5f * delta_sq;
    reward -= delta_sq * env->rcfg.control_rate_penalty;
    env->two_agent_prev_controls[slot][0] = actions[1];
    env->two_agent_prev_controls[slot][1] = actions[2];
    env->two_agent_prev_controls[slot][2] = actions[3];

    float alt_threshold = env->rcfg.low_altitude_threshold;
    float alt_deficit = fmaxf(0.0f, alt_threshold - self->pos.z);
    float alt_ratio = alt_deficit / fmaxf(alt_threshold, 1.0f);
    float altitude_reward = -env->rcfg.low_altitude_penalty
        * alt_ratio * alt_ratio;
    if (self->vel.z < 0.0f && alt_deficit > 0.0f) {
        altitude_reward *= 1.0f + fminf(-self->vel.z / 30.0f, 1.0f);
    }
    reward += altitude_reward;
    reward -= 0.00001f;

    reward += self_energy > self->prev_energy
        ? env->rcfg.energy_gain_scale
        : -env->rcfg.energy_loss_scale;
    float energy_advantage = clampf(
        (self_energy - other_energy) / 1000.0f, -1.0f, 1.0f);
    reward += env->rcfg.energy_advantage_scale * energy_advantage;

    return clampf(reward, -1.0f, 1.0f);
}

/*
 * Pool fitness version 2.
 *
 * Protein must select for combat outcomes, but an outcome alone is not enough
 * in Dogfight: two collapsed policies can otherwise make a persistent
 * one-direction roll look strong relative to an even worse checkpoint. These
 * fixed, reflection-symmetric gates reduce pool score without changing PPO
 * rewards.
 */
#define DOGFIGHT_POOL_GROUND_WIN_CREDIT 0.25f
#define DOGFIGHT_POOL_QUALITY_MIN_STEPS 64
#define DOGFIGHT_POOL_QUALITY_MIN_SIDE_STEPS 16
#define DOGFIGHT_POOL_AILERON_BIAS_GOOD 0.20f
#define DOGFIGHT_POOL_AILERON_BIAS_BAD 0.65f
#define DOGFIGHT_POOL_COMMON_RESPONSE_GOOD 0.15f
#define DOGFIGHT_POOL_COMMON_RESPONSE_BAD 0.60f
#define DOGFIGHT_POOL_DIRECTIONAL_RESPONSE_BAD -0.02f
#define DOGFIGHT_POOL_DIRECTIONAL_RESPONSE_GOOD 0.05f
#define DOGFIGHT_POOL_EXCESSIVE_ROLL_RATE 2.50f
#define DOGFIGHT_POOL_EXCESSIVE_ROLL_FRACTION_GOOD 0.10f
#define DOGFIGHT_POOL_EXCESSIVE_ROLL_FRACTION_BAD 0.60f
#define DOGFIGHT_POOL_ROLL_ROTATIONS_GOOD 0.75f
#define DOGFIGHT_POOL_ROLL_ROTATIONS_BAD 2.00f
#define DOGFIGHT_POOL_CONTROL_RATE_GOOD_EARLY 0.45f
#define DOGFIGHT_POOL_CONTROL_RATE_BAD_EARLY 0.80f
#define DOGFIGHT_POOL_CONTROL_RATE_GOOD_LATE 0.80f
#define DOGFIGHT_POOL_CONTROL_RATE_BAD_LATE 1.20f
#define DOGFIGHT_POOL_CONTROLLED_FRACTION_BAD 0.60f
#define DOGFIGHT_POOL_CONTROLLED_FRACTION_GOOD 0.90f
#define DOGFIGHT_POOL_CONTROLLED_MIN_ALTITUDE 100.0f

static inline float dogfight_pool_quality_descending(
        float value, float good, float bad) {
    if (value <= good) return 1.0f;
    if (value >= bad) return 0.0f;
    return (bad - value) / (bad - good);
}

static inline float dogfight_pool_quality_ascending(
        float value, float bad, float good) {
    if (value <= bad) return 0.0f;
    if (value >= good) return 1.0f;
    return (value - bad) / (good - bad);
}

static inline void dogfight_pool_control_rate_limits(
        int stage, float* good, float* bad) {
    float difficulty = clampf((float)stage / 10.0f, 0.0f, 1.0f);
    *good = DOGFIGHT_POOL_CONTROL_RATE_GOOD_EARLY
        + difficulty * (
            DOGFIGHT_POOL_CONTROL_RATE_GOOD_LATE
            - DOGFIGHT_POOL_CONTROL_RATE_GOOD_EARLY);
    *bad = DOGFIGHT_POOL_CONTROL_RATE_BAD_EARLY
        + difficulty * (
            DOGFIGHT_POOL_CONTROL_RATE_BAD_LATE
            - DOGFIGHT_POOL_CONTROL_RATE_BAD_EARLY);
}

static inline void dogfight_two_agent_reset_pool_quality(Dogfight* env) {
    memset(env->two_agent_pool_aileron_sum, 0,
        sizeof(env->two_agent_pool_aileron_sum));
    memset(env->two_agent_pool_control_rate_sum, 0,
        sizeof(env->two_agent_pool_control_rate_sum));
    memset(env->two_agent_pool_target_negative_aileron_sum, 0,
        sizeof(env->two_agent_pool_target_negative_aileron_sum));
    memset(env->two_agent_pool_target_positive_aileron_sum, 0,
        sizeof(env->two_agent_pool_target_positive_aileron_sum));
    memset(env->two_agent_pool_steps, 0,
        sizeof(env->two_agent_pool_steps));
    memset(env->two_agent_pool_target_negative_steps, 0,
        sizeof(env->two_agent_pool_target_negative_steps));
    memset(env->two_agent_pool_target_positive_steps, 0,
        sizeof(env->two_agent_pool_target_positive_steps));
    memset(env->two_agent_pool_excessive_roll_steps, 0,
        sizeof(env->two_agent_pool_excessive_roll_steps));
    memset(env->two_agent_pool_controlled_steps, 0,
        sizeof(env->two_agent_pool_controlled_steps));
}

static inline void dogfight_two_agent_track_pool_quality(
        Dogfight* env, Plane* planes[2], float* actions[2]) {
    const float target_azimuths[2] = {
        env->observations[13],
        env->opponent_observations[13],
    };

    for (int physical = 0; physical < 2; physical++) {
        Plane* plane = planes[physical];
        float aileron = actions[physical][2];
        float target_azimuth = target_azimuths[physical];

        env->two_agent_pool_steps[physical] += 1;
        env->two_agent_pool_aileron_sum[physical] += aileron;
        if (target_azimuth < -1.0e-4f) {
            env->two_agent_pool_target_negative_aileron_sum[physical] += aileron;
            env->two_agent_pool_target_negative_steps[physical] += 1;
        } else if (target_azimuth > 1.0e-4f) {
            env->two_agent_pool_target_positive_aileron_sum[physical] += aileron;
            env->two_agent_pool_target_positive_steps[physical] += 1;
        }

        if (fabsf(plane->omega.x) > DOGFIGHT_POOL_EXCESSIVE_ROLL_RATE) {
            env->two_agent_pool_excessive_roll_steps[physical] += 1;
        }

        float speed_squared =
            plane->vel.x * plane->vel.x +
            plane->vel.y * plane->vel.y +
            plane->vel.z * plane->vel.z;
        float minimum_speed = env->rcfg.speed_min;
        bool controlled =
            isfinite(plane->pos.z) &&
            isfinite(speed_squared) &&
            plane->pos.z >= DOGFIGHT_POOL_CONTROLLED_MIN_ALTITUDE &&
            plane->pos.z <= WORLD_MAX_Z - DOGFIGHT_POOL_CONTROLLED_MIN_ALTITUDE &&
            speed_squared >= minimum_speed * minimum_speed;
        if (controlled) {
            env->two_agent_pool_controlled_steps[physical] += 1;
        }
    }
}

static inline float dogfight_two_agent_pool_flight_quality(
        const Dogfight* env, int physical) {
    int steps = env->two_agent_pool_steps[physical];
    if (steps <= 0) return 1.0f;

    float controlled_fraction =
        (float)env->two_agent_pool_controlled_steps[physical] / (float)steps;
    float quality = dogfight_pool_quality_ascending(
        controlled_fraction,
        DOGFIGHT_POOL_CONTROLLED_FRACTION_BAD,
        DOGFIGHT_POOL_CONTROLLED_FRACTION_GOOD);

    /* Do not reject legitimate quick kills based on a tiny control sample. */
    if (steps < DOGFIGHT_POOL_QUALITY_MIN_STEPS) return quality;

    float control_rate =
        env->two_agent_pool_control_rate_sum[physical] / (float)steps;
    float control_rate_good;
    float control_rate_bad;
    dogfight_pool_control_rate_limits(
        env->stage, &control_rate_good, &control_rate_bad);
    quality = fminf(quality, dogfight_pool_quality_descending(
        control_rate, control_rate_good, control_rate_bad));

    float signed_bias = fabsf(
        env->two_agent_pool_aileron_sum[physical] / (float)steps);
    quality = fminf(quality, dogfight_pool_quality_descending(
        signed_bias,
        DOGFIGHT_POOL_AILERON_BIAS_GOOD,
        DOGFIGHT_POOL_AILERON_BIAS_BAD));

    float excessive_roll_fraction =
        (float)env->two_agent_pool_excessive_roll_steps[physical] /
        (float)steps;
    quality = fminf(quality, dogfight_pool_quality_descending(
        excessive_roll_fraction,
        DOGFIGHT_POOL_EXCESSIVE_ROLL_FRACTION_GOOD,
        DOGFIGHT_POOL_EXCESSIVE_ROLL_FRACTION_BAD));
    float roll_rotations =
        env->two_agent_roll_travel_radians[physical]
        / (2.0f * (float)M_PI);
    quality = fminf(quality, dogfight_pool_quality_descending(
        roll_rotations,
        DOGFIGHT_POOL_ROLL_ROTATIONS_GOOD,
        DOGFIGHT_POOL_ROLL_ROTATIONS_BAD));

    int negative_steps =
        env->two_agent_pool_target_negative_steps[physical];
    int positive_steps =
        env->two_agent_pool_target_positive_steps[physical];
    if (negative_steps >= DOGFIGHT_POOL_QUALITY_MIN_SIDE_STEPS &&
            positive_steps >= DOGFIGHT_POOL_QUALITY_MIN_SIDE_STEPS) {
        float negative_mean =
            env->two_agent_pool_target_negative_aileron_sum[physical] /
            (float)negative_steps;
        float positive_mean =
            env->two_agent_pool_target_positive_aileron_sum[physical] /
            (float)positive_steps;
        float common_response = fabsf(0.5f * (
            negative_mean + positive_mean));
        float directional_response = 0.5f * (
            negative_mean - positive_mean);

        quality = fminf(quality, dogfight_pool_quality_descending(
            common_response,
            DOGFIGHT_POOL_COMMON_RESPONSE_GOOD,
            DOGFIGHT_POOL_COMMON_RESPONSE_BAD));
        quality = fminf(quality, dogfight_pool_quality_ascending(
            directional_response,
            DOGFIGHT_POOL_DIRECTIONAL_RESPONSE_BAD,
            DOGFIGHT_POOL_DIRECTIONAL_RESPONSE_GOOD));
    }

    return clampf(quality, 0.0f, 1.0f);
}

static inline void dogfight_two_agent_remember_native_aileron_bias(
        Dogfight* env) {
    if (!env->native_spawn_curriculum || env->num_agents != 2) {
        return;
    }

    // Logical slot 0 is the current trainable row. Convert its executed,
    // physical-frame aileron mean back into the episode-fixed policy frame so
    // the next spawn can present a corrective state independent of world side.
    int physical = env->two_agent_player_slot == 0 ? 0 : 1;
    int steps = env->two_agent_pool_steps[physical];
    if (steps < DOGFIGHT_POOL_QUALITY_MIN_STEPS) {
        return;
    }
    float physical_bias =
        env->two_agent_pool_aileron_sum[physical] / (float)steps;
    env->native_previous_canonical_aileron_bias =
        env->lateral_frame_mirror ? -physical_bias : physical_bias;
}

static inline void dogfight_two_agent_raw_pool_scores(
        int physical_winner, float physical_scores[2]) {
    physical_scores[0] = physical_winner == 0
        ? 0.5f
        : (physical_winner == 1 ? 1.0f : 0.0f);
    physical_scores[1] = physical_winner == 0
        ? 0.5f
        : (physical_winner == -1 ? 1.0f : 0.0f);
}

static inline void dogfight_two_agent_adjusted_pool_scores(
        DeathReason reason,
        int physical_winner,
        const float physical_quality[2],
        float physical_scores[2]) {
    physical_scores[0] = 0.0f;
    physical_scores[1] = 0.0f;

    if (physical_winner == 0) {
        if (reason == DEATH_KILL) {
            physical_scores[0] = 0.5f * physical_quality[0];
            physical_scores[1] = 0.5f * physical_quality[1];
        }
        return;
    }

    int winner_index = physical_winner == 1 ? 0 : 1;
    float outcome_credit = 0.0f;
    if (reason == DEATH_KILL) {
        outcome_credit = 1.0f;
    } else if (reason == DEATH_OOB) {
        outcome_credit = DOGFIGHT_POOL_GROUND_WIN_CREDIT;
    }
    physical_scores[winner_index] =
        outcome_credit * physical_quality[winner_index];
}

static inline void dogfight_two_agent_finish(
        Dogfight* env,
        float reward_0,
        float reward_1,
        DeathReason reason,
        int winner,
        int clean_fight) {
    // Match Robocode's native self-play contract. Only tagged historical
    // matches participate in frozen-opponent alignment, and the trainer owns
    // clearing this signal after it has processed the boundary.
    if (env->tag > 0) {
        env->boundary_reached = 1;
    }
    env->rewards[0] = reward_0;
    env->opponent_rewards[0] = reward_1;
    *env->agents[0].terminals = 1.0f;
    *env->agents[1].terminals = 1.0f;
    env->two_agent_episode_returns[0] += reward_0;
    env->two_agent_episode_returns[1] += reward_1;
    env->episode_return = env->two_agent_episode_returns[0];
    env->kill = winner == 1;
    env->opp_kill = winner == -1;
    env->death_reason = reason;
    env->last_death_reason = reason;
    env->last_winner = winner;
    if (clean_fight) {
        env->log.clean_fights += 1.0f;
    }
    int logical_winner = winner == 0
        ? -1
        : (winner == 1
            ? env->two_agent_player_slot
            : 1 - env->two_agent_player_slot);
    if (reason == DEATH_KILL && logical_winner >= 0) {
        if (logical_winner == 0) {
            env->log.slot_0_gun_kills += 1.0f;
        } else {
            env->log.slot_1_gun_kills += 1.0f;
        }
    }
    float physical_raw_scores[2];
    dogfight_two_agent_raw_pool_scores(winner, physical_raw_scores);
    float logical_raw_scores[2] = {0.0f, 0.0f};
    logical_raw_scores[env->two_agent_player_slot] = physical_raw_scores[0];
    logical_raw_scores[1 - env->two_agent_player_slot] =
        physical_raw_scores[1];
    env->log.slot_0_score += logical_raw_scores[0];
    env->log.slot_1_score += logical_raw_scores[1];
    env->log.draw_rate += winner == 0 ? 1.0f : 0.0f;

    float physical_quality[2] = {
        dogfight_two_agent_pool_flight_quality(env, 0),
        dogfight_two_agent_pool_flight_quality(env, 1),
    };
    int candidate_physical = env->two_agent_player_slot == 0 ? 0 : 1;
    env->log.pool_flight_quality += physical_quality[candidate_physical];

    float physical_scores[2];
    dogfight_two_agent_adjusted_pool_scores(
        reason, winner, physical_quality, physical_scores);
    float logical_scores[2] = {0.0f, 0.0f};
    logical_scores[env->two_agent_player_slot] = physical_scores[0];
    logical_scores[1 - env->two_agent_player_slot] = physical_scores[1];
    env->log.slot_0_fitness += logical_scores[0];

    dogfight_two_agent_remember_native_aileron_bias(env);
    add_log(env);
    dogfight_two_agent_select_roles(env);
    c_reset(env);
}

static inline void c_step_two_agent(Dogfight* env) {
    float teacher_actions[2][NUM_ATNS];
    float* actions[2] = {
        env->actions,
        env->opponent_actions_override,
    };
    Plane* planes[2] = {
        &env->player,
        &env->opponent,
    };

    if (env->tick == 0) {
        memset(env->two_agent_prev_controls, 0,
            sizeof(env->two_agent_prev_controls));
        memset(env->two_agent_episode_returns, 0,
            sizeof(env->two_agent_episode_returns));
        memset(env->two_agent_episode_shots, 0,
            sizeof(env->two_agent_episode_shots));
        memset(env->two_agent_roll_travel_radians, 0,
            sizeof(env->two_agent_roll_travel_radians));
        dogfight_two_agent_reset_pool_quality(env);
    }

    for (int slot = 0; slot < 2; slot++) {
        for (int action = 0; action < NUM_ATNS; action++) {
            actions[slot][action] = clampf(
            actions[slot][action], -1.0f, 1.0f);
        }
    }

    if (dogfight_two_agent_in_native_acquisition(env)) {
        float neutral_control_scale =
            env->global_step < env->native_acquisition_steps
            ? env->native_acquisition_neutral_scale
            : 0.0f;
        float acquisition_rewards[2] = {
            dogfight_two_agent_native_acquisition_reward(
                env->observations[13],
                actions[0],
                env->native_acquisition_reward_scale,
                neutral_control_scale),
            dogfight_two_agent_native_acquisition_reward(
                env->opponent_observations[13],
                actions[1],
                env->native_acquisition_reward_scale,
                neutral_control_scale),
        };
        env->total_aileron_usage += fabsf(actions[0][2]);
        env->aileron_bias += actions[0][2];
        if (env->observations[13] < -1.0e-4f) {
            env->target_az_neg_aileron_sum += actions[0][2];
            env->target_az_neg_steps++;
        } else if (env->observations[13] > 1.0e-4f) {
            env->target_az_pos_aileron_sum += actions[0][2];
            env->target_az_pos_steps++;
        }
        memcpy(
            env->last_opp_actions,
            actions[1],
            NUM_ATNS * sizeof(float));
        env->tick++;
        dogfight_two_agent_finish(
            env,
            acquisition_rewards[0],
            acquisition_rewards[1],
            DEATH_TIMEOUT,
            0,
            1);
        return;
    }

    if (env->two_agent_scripted_episode) {
        dogfight_two_agent_pursuit_teacher_actions(
            env, planes[0], planes[1], teacher_actions[0]);
        dogfight_two_agent_pursuit_teacher_actions(
            env, planes[1], planes[0], teacher_actions[1]);
        float flight_school_rewards[2] = {
            dogfight_two_agent_flight_school_reward(
                actions[0],
                teacher_actions[0],
                env->two_agent_bootstrap_imitation_scale),
            dogfight_two_agent_flight_school_reward(
                actions[1],
                teacher_actions[1],
                env->two_agent_bootstrap_imitation_scale),
        };
        env->total_aileron_usage += fabsf(actions[0][2]);
        env->aileron_bias += actions[0][2];
        if (env->observations[13] < -1.0e-4f) {
            env->target_az_neg_aileron_sum += actions[0][2];
            env->target_az_neg_steps++;
        } else if (env->observations[13] > 1.0e-4f) {
            env->target_az_pos_aileron_sum += actions[0][2];
            env->target_az_pos_steps++;
        }
        memcpy(
            env->last_opp_actions,
            actions[1],
            NUM_ATNS * sizeof(float));
        env->tick++;
        dogfight_two_agent_finish(
            env,
            flight_school_rewards[0],
            flight_school_rewards[1],
            DEATH_TIMEOUT,
            0,
            1);
        return;
    }

    dogfight_two_agent_track_pool_quality(env, planes, actions);
    env->total_aileron_usage += fabsf(actions[0][2]);
    env->aileron_bias += actions[0][2];
    if (env->observations[13] < -1.0e-4f) {
        env->target_az_neg_aileron_sum += actions[0][2];
        env->target_az_neg_steps++;
    } else if (env->observations[13] > 1.0e-4f) {
        env->target_az_pos_aileron_sum += actions[0][2];
        env->target_az_pos_steps++;
    }

    env->tick++;
    env->rewards[0] = 0.0f;
    env->opponent_rewards[0] = 0.0f;
    *env->agents[0].terminals = 0.0f;
    *env->agents[1].terminals = 0.0f;

    step_plane_with_params(
        planes[0], actions[0], DT, &env->flight_params);
    step_plane_with_params(
        planes[1], actions[1], DT, &env->flight_params);
    for (int physical = 0; physical < 2; physical++) {
        env->two_agent_roll_travel_radians[physical] =
            dogfight_two_agent_update_roll_travel(
                env->two_agent_roll_travel_radians[physical],
                planes[physical]->omega.x,
                DT);
    }
    env->episode_roll_travel_radians = 0.5f * (
        env->two_agent_roll_travel_radians[0]
        + env->two_agent_roll_travel_radians[1]);
    memcpy(env->last_opp_actions, actions[1], NUM_ATNS * sizeof(float));

    if (env->head_on_lockout) {
        Vec3 rel_pos = sub3(planes[1]->pos, planes[0]->pos);
        Vec3 rel_vel = sub3(planes[1]->vel, planes[0]->vel);
        float rel_dot = dot3(rel_pos, rel_vel);
        if (env->prev_rel_dot < 0.0f && rel_dot >= 0.0f) {
            env->head_on_lockout = 0;
        }
        env->prev_rel_dot = rel_dot;
    }

    bool fired[2] = {false, false};
    bool hit[2] = {false, false};
    for (int slot = 0; slot < 2; slot++) {
        if (planes[slot]->fire_cooldown > 0) {
            planes[slot]->fire_cooldown--;
        }
        fired[slot] = actions[slot][4] > 0.5f
            && planes[slot]->fire_cooldown == 0
            && !env->head_on_lockout;
        if (fired[slot]) {
            planes[slot]->fire_cooldown = FIRE_COOLDOWN;
            env->two_agent_episode_shots[slot] += 1.0f;
            if (slot == 0) {
                env->episode_shots_fired += 1.0f;
            }
        }
    }

    // Both hit intents are sampled from the same post-motion state before any
    // death is resolved, removing player/opponent evaluation order.
    hit[0] = fired[0]
        && check_hit(planes[0], planes[1], env->cos_gun_cone);
    hit[1] = fired[1]
        && check_hit(planes[1], planes[0], env->cos_gun_cone);

    bool crashed[2] = {
        planes[0]->pos.z < 0.0f || planes[0]->pos.z > WORLD_MAX_Z,
        planes[1]->pos.z < 0.0f || planes[1]->pos.z > WORLD_MAX_Z,
    };
    bool supersonic[2] = {
        norm3(planes[0]->vel) > 340.0f,
        norm3(planes[1]->vel) > 340.0f,
    };
    bool dead[2] = {
        hit[1] || crashed[0] || supersonic[0],
        hit[0] || crashed[1] || supersonic[1],
    };

    if (dead[0] || dead[1]) {
        if (crashed[0] && planes[0]->pos.z < 0.0f) {
            env->log.player_ground_hits += 1.0f;
        }
        if (crashed[1] && planes[1]->pos.z < 0.0f) {
            env->log.opponent_ground_hits += 1.0f;
        }

        DeathReason reason = hit[0] || hit[1]
            ? DEATH_KILL
            : (supersonic[0] || supersonic[1]
                ? DEATH_SUPERSONIC
                : DEATH_OOB);
        int winner = dead[0] == dead[1] ? 0 : (dead[1] ? 1 : -1);
        float terminal_rewards[2];
        if (dead[0] && dead[1]) {
            terminal_rewards[0] = 0.0f;
            terminal_rewards[1] = 0.0f;
        } else if (dead[0]) {
            terminal_rewards[0] = -1.0f;
            terminal_rewards[1] = 1.0f;
        } else {
            terminal_rewards[0] = 1.0f;
            terminal_rewards[1] = -1.0f;
        }
        dogfight_two_agent_finish(
            env,
            terminal_rewards[0],
            terminal_rewards[1],
            reason,
            winner,
            hit[0] || hit[1]);
        return;
    }

    if (env->tick >= env->max_steps) {
        dogfight_two_agent_finish(
            env, 0.0f, 0.0f, DEATH_TIMEOUT, 0, 1);
        return;
    }

    float energy[2] = {
        calc_specific_energy_with_params(planes[0], &env->flight_params),
        calc_specific_energy_with_params(planes[1], &env->flight_params),
    };
    float dense_rewards[2] = {
        dogfight_two_agent_dense_reward(
            env, planes[0], planes[1], actions[0], 0,
            energy[0], energy[1]),
        dogfight_two_agent_dense_reward(
            env, planes[1], planes[0], actions[1], 1,
            energy[1], energy[0]),
    };
    planes[0]->prev_energy = energy[0];
    planes[1]->prev_energy = energy[1];
    float published_rewards[2];
    dogfight_two_agent_combine_dense_rewards(
        env->two_agent_reward_version,
        dense_rewards,
        published_rewards);
    float steering_scale = dogfight_two_agent_steering_scale_for_stage(
        env->two_agent_steering_alignment_scale,
        env->stage);
    published_rewards[0] = dogfight_two_agent_publish_steering_reward(
        published_rewards[0],
        env->observations[13],
        actions[0][2],
        steering_scale);
    published_rewards[1] = dogfight_two_agent_publish_steering_reward(
        published_rewards[1],
        env->opponent_observations[13],
        actions[1][2],
        steering_scale);
    published_rewards[0] = dogfight_two_agent_publish_bank_guidance(
        published_rewards[0], planes[0], planes[1], env);
    published_rewards[1] = dogfight_two_agent_publish_bank_guidance(
        published_rewards[1], planes[1], planes[0], env);
    published_rewards[0] = dogfight_two_agent_publish_roll_discipline(
        published_rewards[0],
        env->two_agent_roll_travel_radians[0],
        env);
    published_rewards[1] = dogfight_two_agent_publish_roll_discipline(
        published_rewards[1],
        env->two_agent_roll_travel_radians[1],
        env);
    env->rewards[0] = published_rewards[0];
    env->opponent_rewards[0] = published_rewards[1];
    env->two_agent_episode_returns[0] += published_rewards[0];
    env->two_agent_episode_returns[1] += published_rewards[1];
    env->episode_return = env->two_agent_episode_returns[0];
    compute_observations(env);
}
