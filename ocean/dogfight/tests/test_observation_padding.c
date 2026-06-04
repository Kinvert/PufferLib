#include <math.h>
#include <stdio.h>
#include <string.h>

#include "dogfight.h"

#define TEST_OBS_SIZE 26
#define TEST_NUM_ATNS 5

typedef struct TestEnv {
    Dogfight env;
    float observations[TEST_OBS_SIZE];
    float opponent_observations[TEST_OBS_SIZE];
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

static void fill_obs(float* obs, float value) {
    for (int i = 0; i < TEST_OBS_SIZE; i++) obs[i] = value;
}

static int tail_is_unchanged(const char* label, const float* obs, float expected) {
    for (int i = 22; i < TEST_OBS_SIZE; i++) {
        if (fabsf(obs[i] - expected) > 1e-6f) {
            printf("%s: obs[%d]=%.3f expected untouched %.3f [FAIL]\n",
                label, i, obs[i], expected);
            return 0;
        }
    }
    return 1;
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
    init(&t->env, OBS_PILOT, &rcfg, 1, 0, 0);
    t->env.opponent_observations = t->opponent_observations;
}

static int test_scheme0_writes_only_declared_obs_width(void) {
    TestEnv t;
    setup_env(&t);

    if (t.env.obs_size != 22) {
        printf("obs_padding: scheme0 obs_size=%d expected 22 [FAIL]\n", t.env.obs_size);
        return 1;
    }

    fill_obs(t.observations, 123.0f);
    fill_obs(t.opponent_observations, -123.0f);
    c_reset(&t.env);
    compute_opponent_observations(&t.env, t.opponent_observations);
    if (!tail_is_unchanged("obs_width_reset_player", t.observations, 123.0f)) return 1;
    if (!tail_is_unchanged("obs_width_reset_opponent", t.opponent_observations, -123.0f)) return 1;

    const float neutral[TEST_NUM_ATNS] = {0.5f, 0.0f, 0.0f, 0.0f, -1.0f};
    memcpy(t.actions, neutral, sizeof(neutral));
    fill_obs(t.observations, 77.0f);
    fill_obs(t.opponent_observations, -77.0f);
    c_step(&t.env);
    if (!tail_is_unchanged("obs_width_step_player", t.observations, 77.0f)) return 1;
    if (!tail_is_unchanged("obs_width_step_opponent", t.opponent_observations, -77.0f)) return 1;

    printf("obs_width: scheme0 writes only declared observation width [OK]\n");
    return 0;
}

int main(void) {
    return test_scheme0_writes_only_declared_obs_width();
}
