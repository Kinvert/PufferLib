#include <math.h>
#include <stdio.h>
#include <string.h>

#include "dogfight.h"

#define TEST_OBS_SIZE 26
#define TEST_NUM_ATNS 5

static const float DOGFIGHT3_STEP_OBS[TEST_OBS_SIZE] = {
    0.511763453f,
    -0.355744153f,
    -0.078487128f,
    0.361220658f,
    -0.062026381f,
    -0.040070232f,
    1.000000000f,
    0.359915257f,
    1.000000000f,
    0.377265602f,
    0.043334901f,
    -0.398718685f,
    0.916048884f,
    0.093049392f,
    -0.033674814f,
    0.496904492f,
    0.412499219f,
    0.007537454f,
    0.481735140f,
    0.690473557f,
    -0.018423419f,
    0.450565934f,
    0.021002144f,
    0.892496109f,
    0.573868096f,
    0.103247292f,
};

static const float DOGFIGHT3_STEP_PLAYER_POS[3] = {1203.097412109f, -349.645965576f, 1799.576293945f};
static const float DOGFIGHT3_STEP_PLAYER_VEL[3] = {154.623001099f, 17.679334641f, -21.039566040f};
static const float DOGFIGHT3_STEP_PLAYER_OMEGA[3] = {1.083661914f, -0.186079144f, -0.120210692f};
static const float DOGFIGHT3_STEP_PLAYER_ORI[4] = {0.924915016f, 0.199712366f, -0.045722447f, 0.320244700f};
static const float DOGFIGHT3_STEP_OPPONENT_POS[3] = {1777.597045898f, 421.433807373f, 2050.658691406f};
static const float DOGFIGHT3_STEP_OPPONENT_VEL[3] = {-119.812904358f, 71.537590027f, 33.317016602f};
static const float DOGFIGHT3_STEP_OPPONENT_OMEGA[3] = {-0.055270255f, 2.071420670f, -0.248537004f};
static const float DOGFIGHT3_STEP_OPPONENT_ORI[4] = {0.836919188f, -0.127327174f, 0.193751782f, -0.495796651f};

typedef struct TestEnv {
    Dogfight env;
    float observations[TEST_OBS_SIZE];
    float actions[TEST_NUM_ATNS];
    float rewards[1];
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
    t->env.global_step = 25L;
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

    t->env.player.prev_energy = calc_specific_energy_with_params(&t->env.player, &t->env.flight_params);
    t->env.opponent.prev_energy = calc_specific_energy_with_params(&t->env.opponent, &t->env.flight_params);

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

static int compare_vec3(const char* label, Vec3 got, const float expected[3]) {
    const float values[3] = {got.x, got.y, got.z};
    for (int i = 0; i < 3; i++) {
        if (!nearly(values[i], expected[i], 1e-4f)) {
            printf("scripted_step: %s[%d]=%.9f expected %.9f [FAIL]\n",
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
            printf("scripted_step: %s[%d]=%.9f expected %.9f [FAIL]\n",
                label, i, values[i], expected[i]);
            return 1;
        }
    }
    return 0;
}

static int compare_observations(TestEnv* t) {
    for (int i = 0; i < TEST_OBS_SIZE; i++) {
        if (!nearly(t->observations[i], DOGFIGHT3_STEP_OBS[i], 1e-5f)) {
            printf("scripted_step: obs[%d]=%.9f expected %.9f [FAIL]\n",
                i, t->observations[i], DOGFIGHT3_STEP_OBS[i]);
            return 1;
        }
    }
    return 0;
}

static int test_scripted_step_matches_dogfight3_fixture(void) {
    TestEnv t;
    setup_env(&t);
    c_step(&t.env);

    if (t.env.tick != 124 || t.terminals[0] != 0.0f ||
            t.env.death_reason != DEATH_NONE || t.env.kill != 0 || t.env.opp_kill != 0) {
        printf("scripted_step: flags tick=%d term=%.1f death=%d kill=%d opp=%d [FAIL]\n",
            t.env.tick, t.terminals[0], t.env.death_reason, t.env.kill, t.env.opp_kill);
        return 1;
    }

    if (!nearly(t.rewards[0], 0.010970011f, 1e-6f) ||
            !nearly(t.env.episode_return, 0.010970011f, 1e-6f) ||
            !nearly(t.env.r_closing, 0.010312480f, 1e-6f) ||
            !nearly(t.env.r_aim, 0.001657911f, 1e-6f) ||
            !nearly(t.env.r_rate, -0.000148000f, 1e-7f) ||
            !nearly(t.env.r_altitude, -0.000000000f, 1e-7f) ||
            !nearly(t.env.r_player_energy, -0.000500000f, 1e-7f) ||
            !nearly(t.env.r_energy_adv, -0.000172380f, 1e-7f)) {
        printf("scripted_step: reward breakdown mismatch total=%.9f closing=%.9f aim=%.9f rate=%.9f energy=%.9f adv=%.9f [FAIL]\n",
            t.rewards[0], t.env.r_closing, t.env.r_aim, t.env.r_rate,
            t.env.r_player_energy, t.env.r_energy_adv);
        return 1;
    }

    if (compare_vec3("player_pos", t.env.player.pos, DOGFIGHT3_STEP_PLAYER_POS)) return 1;
    if (compare_vec3("player_vel", t.env.player.vel, DOGFIGHT3_STEP_PLAYER_VEL)) return 1;
    if (compare_vec3("player_omega", t.env.player.omega, DOGFIGHT3_STEP_PLAYER_OMEGA)) return 1;
    if (compare_quat("player_ori", t.env.player.ori, DOGFIGHT3_STEP_PLAYER_ORI)) return 1;
    if (compare_vec3("opponent_pos", t.env.opponent.pos, DOGFIGHT3_STEP_OPPONENT_POS)) return 1;
    if (compare_vec3("opponent_vel", t.env.opponent.vel, DOGFIGHT3_STEP_OPPONENT_VEL)) return 1;
    if (compare_vec3("opponent_omega", t.env.opponent.omega, DOGFIGHT3_STEP_OPPONENT_OMEGA)) return 1;
    if (compare_quat("opponent_ori", t.env.opponent.ori, DOGFIGHT3_STEP_OPPONENT_ORI)) return 1;
    if (compare_observations(&t)) return 1;

    printf("scripted_step: step, reward, and next observations match Dogfight3 fixture [OK]\n");
    return 0;
}

int main(void) {
    return test_scripted_step_matches_dogfight3_fixture();
}
