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

static RewardConfig test_default_rcfg(void) {
    RewardConfig r = {0};
    r.speed_min = 50.0f;
    r.low_altitude_threshold = 1500.0f;
    return r;
}

static void setup_env(TestEnv* t) {
    memset(t, 0, sizeof(*t));
    t->env.num_agents = 1;
    t->env.max_steps = 300;
    t->env.rng = 42;
    t->env.observations = t->observations;
    t->env.actions = t->actions;
    t->env.rewards = t->rewards;
    t->env.terminals = t->terminals;

    RewardConfig rcfg = test_default_rcfg();
    init(&t->env, 1, &rcfg, 1, 0, 0);
    c_reset(&t->env);
}

static int test_unbounded_native_actions_are_clamped(void) {
    TestEnv t;
    setup_env(&t);

    t.env.actions[0] = 3.0f;
    t.env.actions[1] = -4.0f;
    t.env.actions[2] = 5.0f;
    t.env.actions[3] = -6.0f;
    t.env.actions[4] = 7.0f;

    c_step(&t.env);

    int fail = 0;
    for (int i = 0; i < TEST_NUM_ATNS; i++) {
        if (t.env.actions[i] < -1.0f || t.env.actions[i] > 1.0f) {
            printf("action_bounds: action[%d]=%.2f outside Box(-1,1) [FAIL]\n",
                i, t.env.actions[i]);
            fail = 1;
        }
    }

    if (fabsf(t.env.prev_elevator - t.env.actions[1]) > 1e-6f ||
            fabsf(t.env.prev_aileron - t.env.actions[2]) > 1e-6f ||
            fabsf(t.env.prev_rudder - t.env.actions[3]) > 1e-6f) {
        printf("action_bounds: previous controls stored unclamped values [FAIL]\n");
        fail = 1;
    }

    if (!isfinite(t.env.rewards[0]) || fabsf(t.env.rewards[0]) > 1.0f) {
        printf("action_bounds: reward=%.3f not finite/clamped [FAIL]\n", t.env.rewards[0]);
        fail = 1;
    }

    if (!fail) {
        printf("action_bounds: native continuous actions clamp to legacy Box(-1,1) [OK]\n");
    }
    return fail;
}

int main(void) {
    return test_unbounded_native_actions_are_clamped();
}
