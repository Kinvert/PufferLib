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
    compute_opponent_observations(&t->env, t->observations[1]);
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

int main(void) {
    int failures = 0;
    failures += test_stage_zero_resets_are_laterally_balanced();
    failures += test_two_agent_step_is_laterally_symmetric();
    if (failures == 0) {
        puts("two-agent lateral symmetry: ok");
    }
    return failures;
}
