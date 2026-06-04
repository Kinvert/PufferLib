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

static int nearly(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

static int nearly_vec3(Vec3 a, Vec3 b) {
    return nearly(a.x, b.x) && nearly(a.y, b.y) && nearly(a.z, b.z);
}

static int nearly_quat(Quat a, Quat b) {
    return nearly(a.w, b.w) && nearly(a.x, b.x) && nearly(a.y, b.y) && nearly(a.z, b.z);
}

static RewardConfig test_default_rcfg(void) {
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

static void setup_env(TestEnv* t, unsigned int rng) {
    memset(t, 0, sizeof(*t));
    t->env.num_agents = 1;
    t->env.max_steps = 300;
    t->env.rng = rng;
    t->env.observations = t->observations;
    t->env.actions = t->actions;
    t->env.rewards = t->rewards;
    t->env.terminals = t->terminals;

    RewardConfig rcfg = test_default_rcfg();
    init(&t->env, 1, &rcfg, 1, 0, 0);
    set_curriculum_target(&t->env, 5.0f);
    c_reset(&t->env);
}

static void step_with_action(TestEnv* t, const float action[TEST_NUM_ATNS]) {
    memcpy(t->actions, action, TEST_NUM_ATNS * sizeof(float));
    c_step(&t->env);
}

static int compare_observations(TestEnv* a, TestEnv* b) {
    for (int i = 0; i < TEST_OBS_SIZE; i++) {
        if (!nearly(a->observations[i], b->observations[i])) {
            printf("state_roundtrip: obs[%d] %.7f != %.7f [FAIL]\n",
                i, a->observations[i], b->observations[i]);
            return 1;
        }
    }
    return 0;
}

static int compare_future_step(TestEnv* a, TestEnv* b) {
    int fail = 0;
    fail |= compare_observations(a, b);
    if (!nearly(a->rewards[0], b->rewards[0])) {
        printf("state_roundtrip: reward %.7f != %.7f [FAIL]\n",
            a->rewards[0], b->rewards[0]);
        fail = 1;
    }
    if (!nearly(a->terminals[0], b->terminals[0])) {
        printf("state_roundtrip: terminal %.1f != %.1f [FAIL]\n",
            a->terminals[0], b->terminals[0]);
        fail = 1;
    }
    if (a->env.tick != b->env.tick || a->env.death_reason != b->env.death_reason) {
        printf("state_roundtrip: tick/death %d/%d != %d/%d [FAIL]\n",
            a->env.tick, a->env.death_reason, b->env.tick, b->env.death_reason);
        fail = 1;
    }
    if (!nearly_vec3(a->env.player.pos, b->env.player.pos) ||
            !nearly_vec3(a->env.player.vel, b->env.player.vel) ||
            !nearly_quat(a->env.player.ori, b->env.player.ori) ||
            !nearly_vec3(a->env.opponent.pos, b->env.opponent.pos) ||
            !nearly_vec3(a->env.opponent.vel, b->env.opponent.vel) ||
            !nearly_quat(a->env.opponent.ori, b->env.opponent.ori)) {
        printf("state_roundtrip: plane state diverged [FAIL]\n");
        fail = 1;
    }
    return fail;
}

static int compare_terminal_reset_result(TestEnv* a, TestEnv* b) {
    int fail = compare_future_step(a, b);
    if (a->env.last_death_reason != b->env.last_death_reason ||
            a->env.last_winner != b->env.last_winner ||
            a->env.total_episodes != b->env.total_episodes) {
        printf("state_roundtrip_terminal: last result %d/%d/%d != %d/%d/%d [FAIL]\n",
            a->env.last_death_reason, a->env.last_winner, a->env.total_episodes,
            b->env.last_death_reason, b->env.last_winner, b->env.total_episodes);
        fail = 1;
    }
    return fail;
}

static int test_state_roundtrip_restores_future_deterministically(void) {
    TestEnv source;
    TestEnv restored;
    setup_env(&source, 42);
    setup_env(&restored, 999);

    const float warmup_action[TEST_NUM_ATNS] = {0.20f, 0.15f, -0.35f, 0.05f, -1.0f};
    for (int i = 0; i < 8; i++) {
        step_with_action(&source, warmup_action);
    }

    State snapshot = source.env.state;

    const float probe_action[TEST_NUM_ATNS] = {0.35f, -0.20f, 0.40f, -0.10f, 1.0f};
    step_with_action(&source, probe_action);

    restored.env.state = snapshot;
    dogfight_state_restore(&restored.env);
    step_with_action(&restored, probe_action);

    int fail = compare_future_step(&source, &restored);
    if (!fail) {
        printf("state_roundtrip: restored Dogfight state matches scripted future [OK]\n");
    }
    return fail;
}

static int test_state_roundtrip_restores_terminal_future_deterministically(void) {
    TestEnv source;
    TestEnv restored;
    setup_env(&source, 123);
    setup_env(&restored, 456);

    force_state(
        &source.env,
        0.0f, 0.0f, 1000.0f,
        150.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        1.0f,
        250.0f, 0.0f, 1000.0f,
        150.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        3, 0, 0);

    State snapshot = source.env.state;

    const float kill_action[TEST_NUM_ATNS] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    step_with_action(&source, kill_action);

    restored.env.state = snapshot;
    dogfight_state_restore(&restored.env);
    step_with_action(&restored, kill_action);

    int fail = compare_terminal_reset_result(&source, &restored);
    if (!fail) {
        printf("state_roundtrip_terminal: restored terminal future matches scripted kill [OK]\n");
    }
    return fail;
}

int main(void) {
    int fail = 0;
    fail |= test_state_roundtrip_restores_future_deterministically();
    fail |= test_state_roundtrip_restores_terminal_future_deterministically();
    return fail;
}
