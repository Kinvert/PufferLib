#include <math.h>
#include <stdio.h>
#include <string.h>

#include "dogfight.h"

#define TEST_OBS_SIZE 26
#define TEST_NUM_ATNS 5

typedef struct TestEnv {
    Dogfight env;
    float observations[TEST_OBS_SIZE];
    float actions[TEST_NUM_ATNS];
    float rewards[1];
    float terminals[1];
} TestEnv;

typedef struct ExpectedDecay {
    long global_step;
    float reward;
    float r_closing;
    float r_aim;
} ExpectedDecay;

static int nearly(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

static RewardConfig test_rcfg(void) {
    RewardConfig r = {0};
    r.aim_scale = 0.001695f;
    r.closing_scale = 0.0001f;
    r.neg_g = 0.035f;
    r.control_rate_penalty = 0.002f;
    r.low_altitude_threshold = 1200.0f;
    r.low_altitude_penalty = 0.005f;
    r.speed_min = 50.0f;
    r.aim_decay_stage = 30.0f;
    r.shaping_decay_start = 100000000L;
    r.shaping_decay_end = 150000000L;
    r.energy_gain_scale = 0.001f;
    r.energy_loss_scale = 0.0005f;
    r.energy_advantage_scale = 0.004f;
    return r;
}

static Quat make_ori(float yaw, float pitch, float roll) {
    Quat q_yaw = quat_from_axis_angle(vec3(0.0f, 0.0f, 1.0f), yaw);
    Quat q_pitch = quat_from_axis_angle(vec3(0.0f, 1.0f, 0.0f), pitch);
    Quat q_roll = quat_from_axis_angle(vec3(1.0f, 0.0f, 0.0f), roll);
    Quat q = quat_mul(q_yaw, quat_mul(q_pitch, q_roll));
    quat_normalize(&q);
    return q;
}

static void setup_env(TestEnv* t, long global_step) {
    memset(t, 0, sizeof(*t));
    t->env.num_agents = 1;
    t->env.max_steps = 1200;
    t->env.rng = 12345;
    t->env.observations = t->observations;
    t->env.actions = t->actions;
    t->env.rewards = t->rewards;
    t->env.terminals = t->terminals;

    RewardConfig rcfg = test_rcfg();
    init(&t->env, OBS_OPPONENT_AWARE, &rcfg, 1, 0, 0);

    t->env.tick = 123;
    t->env.max_steps = 1200;
    t->env.stage = CURRICULUM_ANGLED;
    t->env.curriculum_target = 5.0f;
    t->env.global_step = global_step;
    t->env.use_opponent_override = 1;
    t->env.head_on_lockout = 0;
    t->env.prev_elevator = -0.08f;
    t->env.prev_aileron = 0.11f;
    t->env.prev_rudder = -0.05f;

    t->env.player.pos = vec3(1200.0f, -350.0f, 1800.0f);
    t->env.player.vel = vec3(155.0f, 18.0f, -22.0f);
    t->env.player.prev_vel = vec3(151.0f, 16.0f, -21.0f);
    t->env.player.omega = vec3(0.41f, -0.27f, 0.08f);
    t->env.player.ori = make_ori(0.63f, -0.21f, 0.34f);
    t->env.player.throttle = 0.74f;
    t->env.player.g_force = 3.2f;

    t->env.opponent.pos = vec3(1780.0f, 420.0f, 2050.0f);
    t->env.opponent.vel = vec3(-120.0f, 72.0f, 34.0f);
    t->env.opponent.prev_vel = vec3(-118.0f, 70.0f, 32.0f);
    t->env.opponent.omega = vec3(-0.33f, 0.22f, -0.11f);
    t->env.opponent.ori = make_ori(-1.10f, 0.18f, -0.42f);
    t->env.opponent.throttle = 0.61f;
    t->env.opponent.g_force = 2.4f;

    t->env.player.prev_energy =
        calc_specific_energy_with_params(&t->env.player, &t->env.flight_params);
    t->env.opponent.prev_energy =
        calc_specific_energy_with_params(&t->env.opponent, &t->env.flight_params);

    t->actions[0] = 0.36f;
    t->actions[1] = -0.22f;
    t->actions[2] = 0.31f;
    t->actions[3] = -0.17f;
    t->actions[4] = -1.0f;

    t->env.opponent_actions_override[0] = -0.10f;
    t->env.opponent_actions_override[1] = 0.18f;
    t->env.opponent_actions_override[2] = -0.24f;
    t->env.opponent_actions_override[3] = 0.13f;
    t->env.opponent_actions_override[4] = -1.0f;
}

static int check_case(const ExpectedDecay* expected) {
    TestEnv t;
    setup_env(&t, expected->global_step);
    c_step(&t.env);

    if (t.terminals[0] != 0.0f || t.env.death_reason != DEATH_NONE) {
        printf("reward_shaping_decay: unexpected terminal step=%ld term=%.1f death=%d [FAIL]\n",
            expected->global_step, t.terminals[0], t.env.death_reason);
        return 1;
    }

    if (!nearly(t.rewards[0], expected->reward, 1e-6f) ||
            !nearly(t.env.r_closing, expected->r_closing, 1e-6f) ||
            !nearly(t.env.r_aim, expected->r_aim, 1e-6f)) {
        printf("reward_shaping_decay: step=%ld reward=%.9f closing=%.9f aim=%.9f [FAIL]\n",
            expected->global_step, t.rewards[0], t.env.r_closing, t.env.r_aim);
        return 1;
    }

    if (!nearly(t.env.r_rate, -0.000148000f, 1e-7f) ||
            !nearly(t.env.r_altitude, -0.000000000f, 1e-7f) ||
            !nearly(t.env.r_player_energy, -0.000500000f, 1e-7f) ||
            !nearly(t.env.r_energy_adv, -0.000172380f, 1e-7f)) {
        printf("reward_shaping_decay: nonshaping terms drifted step=%ld rate=%.9f alt=%.9f energy=%.9f adv=%.9f [FAIL]\n",
            expected->global_step, t.env.r_rate, t.env.r_altitude,
            t.env.r_player_energy, t.env.r_energy_adv);
        return 1;
    }

    return 0;
}

static int test_reward_shaping_decay_matches_df36_fixture(void) {
    const ExpectedDecay cases[] = {
        {25L, 0.010970011f, 0.010312480f, 0.001657911f},
        {125000000L, 0.004984815f, 0.005156240f, 0.000828956f},
        {150000000L, -0.001000380f, 0.000000000f, 0.000000000f},
    };

    for (int i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++) {
        if (check_case(&cases[i])) return 1;
    }

    printf("reward_shaping_decay: aim/closing decay and nonshaping terms match df36 fixture [OK]\n");
    return 0;
}

int main(void) {
    return test_reward_shaping_decay_matches_df36_fixture();
}
