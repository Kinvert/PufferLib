#include <math.h>
#include <stdio.h>
#include <string.h>

#include "dogfight.h"

#define TEST_OBS_SIZE 26
#define TEST_NUM_ATNS 5

static const float DOGFIGHT3_SCHEME1_OBS[TEST_OBS_SIZE] = {
    0.513113201f,
    -0.355990529f,
    -0.085520305f,
    0.136666670f,
    -0.090000004f,
    0.026666667f,
    1.000000000f,
    0.359999985f,
    0.639999986f,
    0.378663987f,
    0.037673328f,
    -0.385249376f,
    0.922043383f,
    0.091425002f,
    -0.028315552f,
    0.497945786f,
    0.417394847f,
    0.007743984f,
    0.462966979f,
    0.073333338f,
    -0.110000007f,
    0.437548459f,
    0.039272949f,
    0.898336768f,
    0.576055586f,
    0.102414653f,
};

typedef struct TestEnv {
    Dogfight env;
    float observations[TEST_OBS_SIZE];
    float actions[TEST_NUM_ATNS];
    float rewards[1];
    float terminals[1];
} TestEnv;

static RewardConfig test_default_rcfg(void) {
    RewardConfig r = {0};
    r.speed_min = 50.0f;
    r.low_altitude_threshold = 1200.0f;
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

    RewardConfig rcfg = test_default_rcfg();
    init(&t->env, OBS_OPPONENT_AWARE, &rcfg, 1, 0, 0);

    t->env.tick = 123;
    t->env.max_steps = 1200;
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
}

static int test_scheme1_observations_match_dogfight3_fixture(void) {
    TestEnv t;
    setup_env(&t);

    if (t.env.obs_scheme != OBS_OPPONENT_AWARE || t.env.obs_size != TEST_OBS_SIZE) {
        printf("scheme1_reference: scheme/size %d/%d expected %d/%d [FAIL]\n",
            t.env.obs_scheme, t.env.obs_size, OBS_OPPONENT_AWARE, TEST_OBS_SIZE);
        return 1;
    }

    compute_observations(&t.env);
    for (int i = 0; i < TEST_OBS_SIZE; i++) {
        float got = t.observations[i];
        float expected = DOGFIGHT3_SCHEME1_OBS[i];
        if (fabsf(got - expected) > 1e-5f) {
            printf("scheme1_reference: obs[%d]=%.9f expected %.9f [FAIL]\n",
                i, got, expected);
            return 1;
        }
    }

    printf("scheme1_reference: observations match Dogfight3 fixture [OK]\n");
    return 0;
}

int main(void) {
    return test_scheme1_observations_match_dogfight3_fixture();
}
