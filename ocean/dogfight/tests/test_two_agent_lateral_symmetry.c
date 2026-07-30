#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../dogfight.h"

#define TOL 5.0e-4f

typedef struct {
    Env env;
    obs_t observations[2][OBS_SIZE];
    float actions[2][NUM_ATNS];
    float rewards[2];
    float terminals[2];
} DualEnv;

static int closef(float a, float b) {
    return fabsf(a - b) <= TOL;
}

static void bind_agents(DualEnv* t) {
    for (int i = 0; i < 2; i++) {
        t->env.agents[i].observations = t->observations[i];
        t->env.agents[i].actions = t->actions[i];
        t->env.agents[i].rewards = &t->rewards[i];
        t->env.agents[i].terminals = &t->terminals[i];
        t->env.agents[i].action_mask = NULL;
        t->env.agents[i].policy = i;
    }
}

static void setup_dual(DualEnv* t, unsigned int seed) {
    memset(t, 0, sizeof(*t));
    t->env.num_agents = 2;
    t->env.max_steps = 300;
    t->env.configured_max_steps = 300;
    t->env.rng = seed;
    bind_agents(t);
    dogfight_bind_rng(&t->env.rng);

    RewardConfig rcfg = {0};
    rcfg.speed_min = 50.0f;
    rcfg.low_altitude_threshold = 1200.0f;
    init(&t->env, OBS_OPPONENT_AWARE, &rcfg, 1, 0, 0);
    t->env.two_agent_role_randomization = 0;
    t->env.two_agent_reward_version = 1;
    t->env.use_opponent_override = 1;
    set_curriculum_target(&t->env, 0.0f);
    puf_reset(&t->env);
}

static int test_stage_zero_resets_are_laterally_balanced(void) {
    DualEnv t;
    setup_dual(&t, 42U);

    const int episodes = 8192;
    int left = 0;
    int right = 0;
    double azimuth_sum = 0.0;
    for (int episode = 0; episode < episodes; episode++) {
        puf_reset(&t.env);
        if (t.env.stage != CURRICULUM_TAIL_CHASE) {
            fprintf(stderr, "fixed stage-0 reset produced stage %d\n",
                t.env.stage);
            return 1;
        }
        float azimuth = t.observations[0][13];
        azimuth_sum += azimuth;
        left += azimuth < 0.0f;
        right += azimuth > 0.0f;
    }

    float imbalance = fabsf((float)(right - left)) / (float)episodes;
    float mean_azimuth = (float)(azimuth_sum / (double)episodes);
    if (left == 0 || right == 0 || imbalance > 0.05f
            || fabsf(mean_azimuth) > 0.01f) {
        fprintf(stderr,
            "stage-0 lateral reset imbalance: left=%d right=%d "
            "imbalance=%.4f mean_azimuth=%.6f\n",
            left, right, imbalance, mean_azimuth);
        return 1;
    }
    return 0;
}

static Quat lateral_mirror_quat(Quat q) {
    return (Quat){
        .w = q.w,
        .x = -q.x,
        .y = q.y,
        .z = -q.z,
    };
}

static Vec3 lateral_mirror_vec(Vec3 v) {
    return vec3(v.x, -v.y, v.z);
}

static void lateral_mirror_plane(const Plane* src, Plane* dst) {
    *dst = *src;
    dst->pos = lateral_mirror_vec(src->pos);
    dst->vel = lateral_mirror_vec(src->vel);
    dst->prev_vel = lateral_mirror_vec(src->prev_vel);
    dst->ori = lateral_mirror_quat(src->ori);
    dst->omega = vec3(-src->omega.x, src->omega.y, -src->omega.z);
}

static int mirrored_vec_close(Vec3 original, Vec3 mirrored) {
    Vec3 expected = lateral_mirror_vec(original);
    return closef(expected.x, mirrored.x)
        && closef(expected.y, mirrored.y)
        && closef(expected.z, mirrored.z);
}

static int mirrored_plane_close(const Plane* original, const Plane* mirrored) {
    Vec3 original_forward = quat_rotate(original->ori, vec3(1, 0, 0));
    Vec3 mirrored_forward = quat_rotate(mirrored->ori, vec3(1, 0, 0));
    Vec3 original_up = quat_rotate(original->ori, vec3(0, 0, 1));
    Vec3 mirrored_up = quat_rotate(mirrored->ori, vec3(0, 0, 1));
    return mirrored_vec_close(original->pos, mirrored->pos)
        && mirrored_vec_close(original->vel, mirrored->vel)
        && closef(-original->omega.x, mirrored->omega.x)
        && closef(original->omega.y, mirrored->omega.y)
        && closef(-original->omega.z, mirrored->omega.z)
        && mirrored_vec_close(original_forward, mirrored_forward)
        && mirrored_vec_close(original_up, mirrored_up);
}

static void set_asymmetric_state(DualEnv* t) {
    reset_plane(
        &t->env.player,
        vec3(-80.0f, 35.0f, 2100.0f),
        vec3(92.0f, 7.0f, 4.0f));
    reset_plane(
        &t->env.opponent,
        vec3(360.0f, 190.0f, 2225.0f),
        vec3(78.0f, -11.0f, -3.0f));

    Quat player_yaw = quat_from_axis_angle(vec3(0, 0, 1), 0.17f);
    Quat player_pitch = quat_from_axis_angle(vec3(0, 1, 0), -0.09f);
    Quat player_roll = quat_from_axis_angle(vec3(1, 0, 0), 0.21f);
    t->env.player.ori =
        quat_mul(player_yaw, quat_mul(player_pitch, player_roll));
    t->env.player.omega = vec3(0.13f, -0.07f, 0.05f);
    t->env.player.prev_vel = t->env.player.vel;

    Quat opponent_yaw = quat_from_axis_angle(vec3(0, 0, 1), -0.31f);
    Quat opponent_pitch = quat_from_axis_angle(vec3(0, 1, 0), 0.12f);
    Quat opponent_roll = quat_from_axis_angle(vec3(1, 0, 0), -0.16f);
    t->env.opponent.ori =
        quat_mul(opponent_yaw, quat_mul(opponent_pitch, opponent_roll));
    t->env.opponent.omega = vec3(-0.09f, 0.04f, -0.06f);
    t->env.opponent.prev_vel = t->env.opponent.vel;
    t->env.tick = 0;
    t->env.head_on_lockout = 0;
    compute_observations(&t->env);
    compute_opponent_observations(
        &t->env, t->env.opponent_observations);
}

static int mirrored_observations_close(
        const obs_t original[OBS_SIZE],
        const obs_t mirrored[OBS_SIZE]) {
    static const float signs[OBS_SIZE] = {
        1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1,
        -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1,
    };
    for (int i = 0; i < OBS_SIZE; i++) {
        if (!closef(signs[i] * original[i], mirrored[i])) {
            fprintf(stderr,
                "mirrored observation mismatch at %d: %.6f -> %.6f "
                "(expected %.6f)\n",
                i, original[i], mirrored[i], signs[i] * original[i]);
            return 0;
        }
    }
    return 1;
}

static int observations_close(
        const obs_t a[OBS_SIZE], const obs_t b[OBS_SIZE]) {
    for (int i = 0; i < OBS_SIZE; i++) {
        if (!closef(a[i], b[i])) {
            fprintf(stderr,
                "canonical observation mismatch at %d: %.6f != %.6f\n",
                i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

static void mirror_actions(
        const float original[NUM_ATNS],
        float mirrored[NUM_ATNS]) {
    memcpy(mirrored, original, NUM_ATNS * sizeof(float));
    mirrored[2] = -original[2];
    mirrored[3] = -original[3];
}

static int test_two_agent_step_is_laterally_symmetric(void) {
    DualEnv original;
    DualEnv mirrored;
    setup_dual(&original, 7U);
    setup_dual(&mirrored, 7U);
    set_asymmetric_state(&original);
    lateral_mirror_plane(&original.env.player, &mirrored.env.player);
    lateral_mirror_plane(&original.env.opponent, &mirrored.env.opponent);
    mirrored.env.tick = original.env.tick;
    mirrored.env.head_on_lockout = original.env.head_on_lockout;
    compute_observations(&mirrored.env);
    compute_opponent_observations(
        &mirrored.env, mirrored.observations[1]);

    if (!mirrored_observations_close(
                original.observations[0], mirrored.observations[0])
            || !mirrored_observations_close(
                original.observations[1], mirrored.observations[1])) {
        return 1;
    }

    const float player_actions[NUM_ATNS] =
        {0.30f, -0.25f, 0.55f, 0.20f, -1.0f};
    const float opponent_actions[NUM_ATNS] =
        {-0.10f, 0.35f, -0.45f, -0.15f, -1.0f};
    memcpy(original.actions[0], player_actions, sizeof(player_actions));
    memcpy(original.actions[1], opponent_actions, sizeof(opponent_actions));
    mirror_actions(player_actions, mirrored.actions[0]);
    mirror_actions(opponent_actions, mirrored.actions[1]);

    puf_step(&original.env);
    puf_step(&mirrored.env);

    if (original.env.target_az_pos_steps != 1
            || original.env.target_az_neg_steps != 0
            || !closef(original.env.target_az_pos_aileron_sum, 0.55f)
            || mirrored.env.target_az_neg_steps != 1
            || mirrored.env.target_az_pos_steps != 0
            || !closef(mirrored.env.target_az_neg_aileron_sum, -0.55f)) {
        fprintf(stderr,
            "target-conditioned aileron telemetry lost lateral sign\n");
        return 1;
    }
    if (!mirrored_plane_close(
                &original.env.player, &mirrored.env.player)
            || !mirrored_plane_close(
                &original.env.opponent, &mirrored.env.opponent)) {
        fprintf(stderr, "two-agent mirrored actions broke flight symmetry\n");
        return 1;
    }
    if (!closef(original.rewards[0], mirrored.rewards[0])
            || !closef(original.rewards[1], mirrored.rewards[1])) {
        fprintf(stderr,
            "two-agent mirrored rewards differ: %.6f/%.6f %.6f/%.6f\n",
            original.rewards[0], mirrored.rewards[0],
            original.rewards[1], mirrored.rewards[1]);
        return 1;
    }
    if (!mirrored_observations_close(
                original.observations[0], mirrored.observations[0])
            || !mirrored_observations_close(
                original.observations[1], mirrored.observations[1])) {
        return 1;
    }
    return 0;
}

static void publish_canonical_state(DualEnv* t) {
    t->env.lateral_canonicalization = 1;
    t->env.lateral_observations_published = 0;
    dogfight_select_lateral_frame(&t->env);
    dogfight_publish_canonical_observations(&t->env);
}

static int telemetry_matches_executed_aileron(
        const Dogfight* env, float executed_aileron) {
    int total_steps =
        (int)(env->target_az_neg_steps + env->target_az_pos_steps);
    float total_aileron =
        env->target_az_neg_aileron_sum
        + env->target_az_pos_aileron_sum;
    return total_steps == 1 && closef(total_aileron, executed_aileron);
}

static int test_canonical_adapter_is_equivariant_and_non_mutating(void) {
    for (int player_slot = 0; player_slot < 2; player_slot++) {
        DualEnv original;
        DualEnv mirrored;
        setup_dual(&original, 71U + (unsigned int)player_slot);
        setup_dual(&mirrored, 71U + (unsigned int)player_slot);
        original.env.two_agent_player_slot = player_slot;
        mirrored.env.two_agent_player_slot = player_slot;
        dogfight_two_agent_bind_slots(&original.env);
        dogfight_two_agent_bind_slots(&mirrored.env);

        set_asymmetric_state(&original);
        lateral_mirror_plane(
            &original.env.player, &mirrored.env.player);
        lateral_mirror_plane(
            &original.env.opponent, &mirrored.env.opponent);
        mirrored.env.tick = original.env.tick;
        mirrored.env.head_on_lockout = original.env.head_on_lockout;
        compute_observations(&mirrored.env);
        compute_opponent_observations(
            &mirrored.env, mirrored.env.opponent_observations);
        publish_canonical_state(&original);
        publish_canonical_state(&mirrored);

        if (original.env.lateral_frame_mirror
                == mirrored.env.lateral_frame_mirror
                || !observations_close(
                    original.observations[0], mirrored.observations[0])
                || !observations_close(
                    original.observations[1], mirrored.observations[1])) {
            fprintf(stderr,
                "canonical reset did not collapse mirrored worlds "
                "for player slot %d\n", player_slot);
            return 1;
        }

        const float slot_actions[2][NUM_ATNS] = {
            {0.30f, -0.25f, 0.55f, 0.20f, -1.0f},
            {-0.10f, 0.35f, -0.45f, -0.15f, -1.0f},
        };
        float original_action_copy[2][NUM_ATNS];
        float mirrored_action_copy[2][NUM_ATNS];
        memcpy(original.actions, slot_actions, sizeof(slot_actions));
        memcpy(mirrored.actions, slot_actions, sizeof(slot_actions));
        memcpy(original_action_copy, original.actions,
            sizeof(original_action_copy));
        memcpy(mirrored_action_copy, mirrored.actions,
            sizeof(mirrored_action_copy));

        puf_step(&original.env);
        puf_step(&mirrored.env);

        if (memcmp(original.actions, original_action_copy,
                    sizeof(original_action_copy)) != 0
                || memcmp(mirrored.actions, mirrored_action_copy,
                    sizeof(mirrored_action_copy)) != 0) {
            fprintf(stderr,
                "canonical adapter mutated Protein action buffers\n");
            return 1;
        }
        if (!closef(original.env.executed_player_actions[2],
                    -mirrored.env.executed_player_actions[2])
                || !closef(original.env.opponent_actions_override[2],
                    -mirrored.env.opponent_actions_override[2])
                || !telemetry_matches_executed_aileron(
                    &original.env,
                    original.env.executed_player_actions[2])
                || !telemetry_matches_executed_aileron(
                    &mirrored.env,
                    mirrored.env.executed_player_actions[2])) {
            fprintf(stderr,
                "canonical adapter lost physical action signs "
                "for player slot %d\n", player_slot);
            return 1;
        }
        if (!mirrored_plane_close(
                    &original.env.player, &mirrored.env.player)
                || !mirrored_plane_close(
                    &original.env.opponent, &mirrored.env.opponent)
                || !closef(original.rewards[0], mirrored.rewards[0])
                || !closef(original.rewards[1], mirrored.rewards[1])
                || !observations_close(
                    original.observations[0], mirrored.observations[0])
                || !observations_close(
                    original.observations[1], mirrored.observations[1])) {
            fprintf(stderr,
                "canonical adapter broke mirrored transition "
                "for player slot %d\n", player_slot);
            return 1;
        }
    }
    return 0;
}

static int test_canonical_frame_refreshes_after_terminal_reset(void) {
    DualEnv t;
    setup_dual(&t, 991U);
    t.env.lateral_canonicalization = 1;
    puf_reset(&t.env);
    t.env.max_steps = 1;
    memset(t.actions, 0, sizeof(t.actions));
    puf_step(&t.env);

    if (t.terminals[0] == 0.0f || t.terminals[1] == 0.0f
            || !t.env.lateral_observations_published
            || (t.env.lateral_frame_mirror != 0
                && t.env.lateral_frame_mirror != 1)) {
        fprintf(stderr,
            "canonical frame was not republished after terminal reset\n");
        return 1;
    }

    static const int selectors[] = {13, 11, 1, 3, 5, 20, 22};
    for (size_t i = 0; i < sizeof(selectors) / sizeof(selectors[0]); i++) {
        float value = t.env.observations[selectors[i]];
        if (fabsf(value) > 1.0e-8f) {
            if (value < 0.0f) {
                fprintf(stderr,
                    "terminal reset published a non-canonical frame\n");
                return 1;
            }
            break;
        }
    }
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_stage_zero_resets_are_laterally_balanced();
    failures += test_two_agent_step_is_laterally_symmetric();
    failures += test_canonical_adapter_is_equivariant_and_non_mutating();
    failures += test_canonical_frame_refreshes_after_terminal_reset();
    if (failures == 0) {
        puts("two-agent lateral symmetry: ok");
    }
    return failures;
}
