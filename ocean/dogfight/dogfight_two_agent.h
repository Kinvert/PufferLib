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

static inline bool dogfight_two_agent_in_native_acquisition(
        const Dogfight* env) {
    if (!env->native_spawn_curriculum) {
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
    env->log.slot_0_score += winner == 0
        ? 0.5f
        : (logical_winner == 0 ? 1.0f : 0.0f);
    env->log.slot_1_score += winner == 0
        ? 0.5f
        : (logical_winner == 1 ? 1.0f : 0.0f);
    env->log.draw_rate += winner == 0 ? 1.0f : 0.0f;
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
    env->rewards[0] = published_rewards[0];
    env->opponent_rewards[0] = published_rewards[1];
    env->two_agent_episode_returns[0] += published_rewards[0];
    env->two_agent_episode_returns[1] += published_rewards[1];
    env->episode_return = env->two_agent_episode_returns[0];
    compute_observations(env);
}
