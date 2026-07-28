#include <math.h>
#include <stdio.h>
#include <string.h>

#include "dogfight.h"

#define TEST_OBS_SIZE 26
#define TEST_NUM_ATNS 5
#define TRACE_STEPS 12

static const float PLAYER_ACTIONS[TRACE_STEPS][TEST_NUM_ATNS] = {
    {0.10f, -0.12f, 0.24f, -0.08f, -1.0f},
    {0.18f, -0.05f, 0.18f, -0.02f, -1.0f},
    {0.25f, 0.04f, 0.12f, 0.05f, -1.0f},
    {0.08f, 0.11f, 0.02f, 0.10f, -1.0f},
    {-0.04f, 0.16f, -0.08f, 0.04f, -1.0f},
    {-0.12f, 0.06f, -0.18f, -0.05f, -1.0f},
    {-0.02f, -0.04f, -0.10f, -0.12f, -1.0f},
    {0.14f, -0.13f, 0.06f, -0.04f, -1.0f},
    {0.22f, -0.06f, 0.20f, 0.03f, -1.0f},
    {0.05f, 0.08f, 0.14f, 0.11f, -1.0f},
    {-0.08f, 0.13f, -0.04f, 0.06f, -1.0f},
    {0.02f, 0.02f, -0.16f, -0.03f, -1.0f},
};

static const float OPPONENT_ACTIONS[TRACE_STEPS][TEST_NUM_ATNS] = {
    {-0.10f, 0.06f, -0.16f, 0.03f, -1.0f},
    {-0.06f, 0.12f, -0.10f, 0.09f, -1.0f},
    {0.04f, 0.18f, 0.00f, 0.12f, -1.0f},
    {0.12f, 0.09f, 0.11f, 0.05f, -1.0f},
    {0.18f, 0.00f, 0.18f, -0.02f, -1.0f},
    {0.06f, -0.08f, 0.12f, -0.10f, -1.0f},
    {-0.04f, -0.14f, 0.02f, -0.14f, -1.0f},
    {-0.14f, -0.06f, -0.10f, -0.07f, -1.0f},
    {-0.02f, 0.03f, -0.18f, 0.00f, -1.0f},
    {0.10f, 0.10f, -0.06f, 0.08f, -1.0f},
    {0.16f, 0.04f, 0.08f, 0.13f, -1.0f},
    {0.00f, -0.05f, 0.16f, 0.04f, -1.0f},
};

static const float DF36_STEP_REWARDS[TRACE_STEPS] = {
    0.009889215f,
    0.010071762f,
    0.009924255f,
    0.009766961f,
    0.009728002f,
    0.009596534f,
    0.009446928f,
    0.009399904f,
    0.009344898f,
    0.009179410f,
    0.009128619f,
    0.009081302f,
};

static const float DF36_TRACE_PLAYER_POS[3] = {531.150024414f, -274.979827881f, 2198.815185547f};
static const float DF36_TRACE_PLAYER_VEL[3] = {129.611358643f, 20.070404053f, -3.518382311f};
static const float DF36_TRACE_PLAYER_OMEGA[3] = {1.023152351f, 0.211644724f, -0.654094100f};
static const float DF36_TRACE_PLAYER_ORI[4] = {0.961600006f, 0.208749831f, 0.014986308f, 0.177551448f};
static const float DF36_TRACE_OPPONENT_POS[3] = {937.236633301f, 405.971588135f, 2261.731689453f};
static const float DF36_TRACE_OPPONENT_VEL[3] = {-94.852821350f, 64.655654907f, 0.573884726f};
static const float DF36_TRACE_OPPONENT_OMEGA[3] = {1.158403635f, 5.000000000f, -0.935623825f};
static const float DF36_TRACE_OPPONENT_ORI[4] = {0.707984924f, 0.169368535f, 0.523158550f, -0.443144202f};

static const float DF36_TRACE_OBS[TEST_OBS_SIZE] = {
    0.512803376f,
    -0.111289971f,
    0.008729712f,
    0.341050804f,
    0.070548244f,
    -0.218031377f,
    -0.428688854f,
    0.439763039f,
    0.235523552f,
    0.357595950f,
    0.102949329f,
    -0.396145999f,
    0.912397981f,
    0.205878407f,
    -0.137579575f,
    0.397668123f,
    0.304445028f,
    0.026000857f,
    0.425421983f,
    1.000000000f,
    0.386134565f,
    0.590667367f,
    -0.703490078f,
    0.395238847f,
    0.459177405f,
    0.040799335f,
};

typedef struct TestEnv {
    Dogfight env;
    float observations[TEST_OBS_SIZE];
    float actions[TEST_NUM_ATNS];
    float rewards[1];
    float opponent_rewards[1];
    float terminals[1];
} TestEnv;

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

static void setup_env(TestEnv* t) {
    memset(t, 0, sizeof(*t));
    t->env.num_agents = 1;
    t->env.max_steps = 1200;
    t->env.rng = 112233;
    t->env.observations = t->observations;
    t->env.actions = t->actions;
    t->env.rewards = t->rewards;
    t->env.terminals = t->terminals;

    RewardConfig rcfg = test_rcfg();
    init(&t->env, OBS_OPPONENT_AWARE, &rcfg, 1, 0, 0);
    t->env.opponent_rewards = t->opponent_rewards;

    t->env.tick = 37;
    t->env.max_steps = 1200;
    t->env.configured_max_steps = 1200;
    t->env.stage = CURRICULUM_GENTLE_TURNS;
    t->env.curriculum_target = 3.0f;
    t->env.global_step = 25L;
    t->env.use_opponent_override = 1;
    t->env.head_on_lockout = 0;
    t->env.selfplay_active = 0;
    t->env.prev_elevator = 0.03f;
    t->env.prev_aileron = -0.07f;
    t->env.prev_rudder = 0.02f;

    t->env.player.pos = vec3(500.0f, -280.0f, 2200.0f);
    t->env.player.vel = vec3(130.0f, 22.0f, -8.0f);
    t->env.player.prev_vel = vec3(128.0f, 21.0f, -7.5f);
    t->env.player.omega = vec3(0.18f, -0.09f, 0.04f);
    t->env.player.ori = make_ori(0.42f, -0.10f, 0.21f);
    t->env.player.throttle = 0.62f;
    t->env.player.g_force = 1.8f;

    t->env.opponent.pos = vec3(960.0f, 390.0f, 2260.0f);
    t->env.opponent.vel = vec3(-95.0f, 68.0f, 12.0f);
    t->env.opponent.prev_vel = vec3(-94.0f, 67.0f, 11.0f);
    t->env.opponent.omega = vec3(-0.14f, 0.13f, -0.06f);
    t->env.opponent.ori = make_ori(-0.84f, 0.08f, -0.24f);
    t->env.opponent.throttle = 0.54f;
    t->env.opponent.g_force = 1.6f;

    t->env.player.prev_energy =
        calc_specific_energy_with_params(&t->env.player, &t->env.flight_params);
    t->env.opponent.prev_energy =
        calc_specific_energy_with_params(&t->env.opponent, &t->env.flight_params);
}

static void copy_action(float* dst, const float* src) {
    for (int i = 0; i < TEST_NUM_ATNS; i++) {
        dst[i] = src[i];
    }
}

static int compare_vec3(const char* label, Vec3 got, const float expected[3]) {
    const float values[3] = {got.x, got.y, got.z};
    for (int i = 0; i < 3; i++) {
        if (!nearly(values[i], expected[i], 1e-4f)) {
            printf("scripted_trace: %s[%d]=%.9f expected %.9f [FAIL]\n",
                label, i, values[i], expected[i]);
            return 1;
        }
    }
    return 0;
}

static int compare_quat(const char* label, Quat got, const float expected[4]) {
    const float values[4] = {got.w, got.x, got.y, got.z};
    for (int i = 0; i < 4; i++) {
        if (!nearly(values[i], expected[i], 1e-4f)) {
            printf("scripted_trace: %s[%d]=%.9f expected %.9f [FAIL]\n",
                label, i, values[i], expected[i]);
            return 1;
        }
    }
    return 0;
}

static int compare_observations(TestEnv* t) {
    for (int i = 0; i < TEST_OBS_SIZE; i++) {
        if (!nearly(t->observations[i], DF36_TRACE_OBS[i], 1e-5f)) {
            printf("scripted_trace: obs[%d]=%.9f expected %.9f [FAIL]\n",
                i, t->observations[i], DF36_TRACE_OBS[i]);
            return 1;
        }
    }
    return 0;
}

static int test_scripted_trace_matches_df36_fixture(void) {
    TestEnv t;
    setup_env(&t);

    for (int step = 0; step < TRACE_STEPS; step++) {
        copy_action(t.actions, PLAYER_ACTIONS[step]);
        copy_action(t.env.opponent_actions_override, OPPONENT_ACTIONS[step]);
        c_step(&t.env);
        if (!nearly(t.rewards[0], DF36_STEP_REWARDS[step], 1e-6f) ||
                t.terminals[0] != 0.0f ||
                t.env.tick != 38 + step ||
                t.env.death_reason != DEATH_NONE) {
            printf("scripted_trace: step=%d reward=%.9f term=%.1f tick=%d death=%d [FAIL]\n",
                step, t.rewards[0], t.terminals[0], t.env.tick, t.env.death_reason);
            return 1;
        }
    }

    if (t.env.tick != 49 || t.env.kill != 0 || t.env.opp_kill != 0) {
        printf("scripted_trace: final flags tick=%d kill=%d opp=%d [FAIL]\n",
            t.env.tick, t.env.kill, t.env.opp_kill);
        return 1;
    }

    if (!nearly(t.rewards[0], 0.009081302f, 1e-6f) ||
            !nearly(t.env.episode_return, 0.114557788f, 1e-6f) ||
            !nearly(t.env.r_closing, 0.007611125f, 1e-6f) ||
            !nearly(t.env.r_aim, 0.001508094f, 1e-6f) ||
            !nearly(t.env.r_rate, -0.000069200f, 1e-7f) ||
            !nearly(t.env.r_altitude, -0.000000000f, 1e-7f) ||
            !nearly(t.env.r_player_energy, -0.000500000f, 1e-7f) ||
            !nearly(t.env.r_energy_adv, 0.000571282f, 1e-7f) ||
            !nearly(t.env.sum_r_closing, 0.097289592f, 1e-6f) ||
            !nearly(t.env.sum_r_aim, 0.018521061f, 1e-6f) ||
            !nearly(t.env.sum_r_rate, -0.000822600f, 1e-7f) ||
            !nearly(t.env.sum_r_player_energy, -0.006000001f, 1e-7f) ||
            !nearly(t.env.sum_r_energy_adv, 0.006419738f, 1e-7f)) {
        printf("scripted_trace: reward summary ret=%.9f closing=%.9f aim=%.9f rate=%.9f energy=%.9f adv=%.9f [FAIL]\n",
            t.env.episode_return, t.env.r_closing, t.env.r_aim, t.env.r_rate,
            t.env.r_player_energy, t.env.r_energy_adv);
        return 1;
    }

    if (!nearly(t.env.prev_elevator, 0.020000000f, 1e-6f) ||
            !nearly(t.env.prev_aileron, -0.159999996f, 1e-6f) ||
            !nearly(t.env.prev_rudder, -0.029999999f, 1e-6f)) {
        printf("scripted_trace: prev actions %.9f %.9f %.9f [FAIL]\n",
            t.env.prev_elevator, t.env.prev_aileron, t.env.prev_rudder);
        return 1;
    }

    if (compare_vec3("player_pos", t.env.player.pos, DF36_TRACE_PLAYER_POS)) return 1;
    if (compare_vec3("player_vel", t.env.player.vel, DF36_TRACE_PLAYER_VEL)) return 1;
    if (compare_vec3("player_omega", t.env.player.omega, DF36_TRACE_PLAYER_OMEGA)) return 1;
    if (compare_quat("player_ori", t.env.player.ori, DF36_TRACE_PLAYER_ORI)) return 1;
    if (compare_vec3("opponent_pos", t.env.opponent.pos, DF36_TRACE_OPPONENT_POS)) return 1;
    if (compare_vec3("opponent_vel", t.env.opponent.vel, DF36_TRACE_OPPONENT_VEL)) return 1;
    if (compare_vec3("opponent_omega", t.env.opponent.omega, DF36_TRACE_OPPONENT_OMEGA)) return 1;
    if (compare_quat("opponent_ori", t.env.opponent.ori, DF36_TRACE_OPPONENT_ORI)) return 1;
    if (compare_observations(&t)) return 1;

    printf("scripted_trace: 12-step physics, rewards, and observations match df36 fixture [OK]\n");
    return 0;
}

int main(void) {
    return test_scripted_trace_matches_df36_fixture();
}
