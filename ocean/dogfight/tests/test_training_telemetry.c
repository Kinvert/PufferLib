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

static void run_steps(TestEnv* t, int n_steps, const float* action) {
    for (int i = 0; i < n_steps; i++) {
        memcpy(t->env.actions, action, TEST_NUM_ATNS * sizeof(float));
        c_step(&t->env);
    }
}

static int test_action_telemetry_accumulates_from_steps(void) {
    TestEnv t;
    setup_env(&t);

    const float action[TEST_NUM_ATNS] = {0.0f, 0.50f, -1.0f, 0.25f, 1.0f};
    run_steps(&t, 4, action);
    t.env.death_reason = DEATH_TIMEOUT;
    add_log(&t.env);

    int fail = 0;
    fail |= !nearly(t.env.log.action_abs_elevator, 0.50f);
    fail |= !nearly(t.env.log.action_abs_aileron, 1.00f);
    fail |= !nearly(t.env.log.action_abs_rudder, 0.25f);
    fail |= !nearly(t.env.log.action_abs_trigger, 1.00f);
    fail |= !nearly(t.env.log.action_sat_elevator, 0.00f);
    fail |= !nearly(t.env.log.action_sat_aileron, 1.00f);
    fail |= !nearly(t.env.log.action_sat_rudder, 0.00f);
    fail |= !nearly(t.env.log.action_sat_trigger, 1.00f);

    if (fail) {
        printf(
            "action_telemetry: abs=(%.2f %.2f %.2f %.2f) sat=(%.2f %.2f %.2f %.2f) [FAIL]\n",
            t.env.log.action_abs_elevator,
            t.env.log.action_abs_aileron,
            t.env.log.action_abs_rudder,
            t.env.log.action_abs_trigger,
            t.env.log.action_sat_elevator,
            t.env.log.action_sat_aileron,
            t.env.log.action_sat_rudder,
            t.env.log.action_sat_trigger);
        return 1;
    }

    printf("action_telemetry: per-action mean abs and saturation [OK]\n");
    return 0;
}

static int test_curriculum_quality_penalizes_surface_saturation(void) {
    TestEnv healthy;
    setup_env(&healthy);
    set_curriculum_target(&healthy.env, 9.0f);
    healthy.env.tick = 4;
    healthy.env.death_reason = DEATH_TIMEOUT;
    healthy.env.episode_action_sat_elevator = 0.0f;
    healthy.env.episode_action_sat_aileron = 2.0f;
    healthy.env.episode_action_sat_rudder = 0.0f;
    add_log(&healthy.env);

    float progress = 9.0f / (float)(CURRICULUM_COUNT - 1);
    float surface_saturation = (0.0f / 4.0f + 2.0f / 4.0f + 0.0f / 4.0f) / 3.0f;
    float expected = progress * (1.0f - surface_saturation);

    TestEnv saturated;
    setup_env(&saturated);
    set_curriculum_target(&saturated.env, (float)(CURRICULUM_COUNT - 1));
    saturated.env.tick = 4;
    saturated.env.death_reason = DEATH_TIMEOUT;
    saturated.env.episode_action_sat_elevator = 4.0f;
    saturated.env.episode_action_sat_aileron = 4.0f;
    saturated.env.episode_action_sat_rudder = 4.0f;
    add_log(&saturated.env);

    if (!nearly(healthy.env.log.curriculum_quality, expected) ||
            !nearly(saturated.env.log.curriculum_quality, 0.0f)) {
        printf(
            "curriculum_quality: healthy=%.4f expected=%.4f saturated=%.4f [FAIL]\n",
            healthy.env.log.curriculum_quality,
            expected,
            saturated.env.log.curriculum_quality);
        return 1;
    }

    printf("curriculum_quality: target progress penalized by saturation [OK]\n");
    return 0;
}

static int test_low_alt_variant_log_for_stage3(void) {
    TestEnv t;
    setup_env(&t);
    set_curriculum_stage(&t.env, CURRICULUM_GENTLE_TURNS);

    for (int i = 0; i < 1000; i++) {
        c_reset(&t.env);
        if (!t.env.low_altitude_variant) continue;

        if (t.env.max_steps != 2000) {
            printf("low_alt_stage3: max_steps=%d [FAIL]\n", t.env.max_steps);
            return 1;
        }

        t.env.tick = t.env.max_steps;
        t.env.death_reason = DEATH_TIMEOUT;
        add_log(&t.env);

        if (!nearly(t.env.log.low_alt_variant_eps, 1.0f) ||
                !nearly(t.env.log.low_alt_variant_ticks, 2000.0f)) {
            printf(
                "low_alt_stage3: eps=%.1f ticks=%.1f [FAIL]\n",
                t.env.log.low_alt_variant_eps,
                t.env.log.low_alt_variant_ticks);
            return 1;
        }

        printf("low_alt_stage3: low-alt episode/tick telemetry [OK]\n");
        return 0;
    }

    printf("low_alt_stage3: did not sample low-alt variant [FAIL]\n");
    return 1;
}

static int test_mastery_stage_telemetry_accumulates_current_stage_only(void) {
    TestEnv t;
    setup_env(&t);
    set_curriculum_target(&t.env, (float)CURRICULUM_DIVE_ATTACK);

    t.env.stage = CURRICULUM_SIDE_MANEUVERING;
    t.env.tick = 10;
    t.env.death_reason = DEATH_TIMEOUT;
    t.env.episode_action_sat_elevator = 1.0f;
    t.env.episode_action_sat_aileron = 2.0f;
    t.env.episode_action_sat_rudder = 3.0f;
    t.env.aileron_bias = -0.25f;
    add_log(&t.env);

    t.env.stage = CURRICULUM_DIVE_ATTACK;
    t.env.tick = 20;
    t.env.death_reason = DEATH_OOB;
    t.env.player.pos.z = -1.0f;
    t.env.episode_action_sat_elevator = 4.0f;
    t.env.episode_action_sat_aileron = 2.0f;
    t.env.episode_action_sat_rudder = 0.0f;
    t.env.elevator_bias = -0.50f;
    t.env.aileron_bias = 0.50f;
    t.env.rudder_bias = 0.25f;
    add_log(&t.env);

    int fail = 0;
    fail |= !nearly(t.env.log.base_stage_eps, 1.0f);
    fail |= !nearly(t.env.log.base_stage_ground, 1.0f);
    fail |= !nearly(t.env.log.base_stage_timeouts, 0.0f);
    fail |= !nearly(t.env.log.base_stage_episode_length, 20.0f);
    fail |= !nearly(t.env.log.base_stage_action_saturation, 0.1f);
    fail |= !nearly(t.env.log.base_stage_signed_bias, 0.50f);
    fail |= !nearly(t.env.log.base_stage_action_sat_elevator, 0.20f);
    fail |= !nearly(t.env.log.base_stage_action_sat_aileron, 0.10f);
    fail |= !nearly(t.env.log.base_stage_action_sat_rudder, 0.00f);
    fail |= !nearly(t.env.log.base_stage_signed_bias_elevator, -0.50f);
    fail |= !nearly(t.env.log.base_stage_signed_bias_aileron, 0.50f);
    fail |= !nearly(t.env.log.base_stage_signed_bias_rudder, 0.25f);

    if (fail) {
        printf(
            "mastery_stage_telemetry: eps=%.1f ground=%.1f timeout=%.1f len=%.1f sat=%.3f bias=%.3f [FAIL]\n",
            t.env.log.base_stage_eps,
            t.env.log.base_stage_ground,
            t.env.log.base_stage_timeouts,
            t.env.log.base_stage_episode_length,
            t.env.log.base_stage_action_saturation,
            t.env.log.base_stage_signed_bias);
        return 1;
    }

    printf("mastery_stage_telemetry: current-stage diagnostics only [OK]\n");
    return 0;
}

static int test_side_spawn_variant_telemetry_mastery_only(void) {
    TestEnv t;
    setup_env(&t);
    set_curriculum_target(&t.env, (float)CURRICULUM_SIDE_MANEUVERING);

    t.env.stage = CURRICULUM_SIDE_FAR;
    t.env.side_spawn_variant = SIDE_SPAWN_ENERGY;
    t.env.tick = 10;
    t.env.kill = 1;
    t.env.death_reason = DEATH_KILL;
    add_log(&t.env);

    t.env.stage = CURRICULUM_SIDE_MANEUVERING;
    t.env.side_spawn_variant = SIDE_SPAWN_STANDARD;
    t.env.tick = 20;
    t.env.kill = 0;
    t.env.death_reason = DEATH_TIMEOUT;
    t.env.episode_action_sat_elevator = 2.0f;
    t.env.episode_action_sat_aileron = 4.0f;
    t.env.episode_action_sat_rudder = 0.0f;
    t.env.episode_action_sat_trigger = 6.0f;
    t.env.elevator_bias = -0.50f;
    t.env.aileron_bias = -0.25f;
    t.env.rudder_bias = 0.50f;
    add_log(&t.env);

    t.env.stage = CURRICULUM_SIDE_MANEUVERING;
    t.env.side_spawn_variant = SIDE_SPAWN_ENERGY;
    t.env.tick = 40;
    t.env.kill = 1;
    t.env.death_reason = DEATH_OOB;
    t.env.player.pos.z = -1.0f;
    t.env.opponent.pos.z = 1000.0f;
    t.env.episode_action_sat_elevator = 0.0f;
    t.env.episode_action_sat_aileron = 4.0f;
    t.env.episode_action_sat_rudder = 4.0f;
    t.env.episode_action_sat_trigger = 8.0f;
    t.env.elevator_bias = 1.25f;
    t.env.aileron_bias = 0.75f;
    t.env.rudder_bias = -1.50f;
    add_log(&t.env);

    int fail = 0;
    fail |= !nearly(t.env.log.base_stage_side_standard_eps, 1.0f);
    fail |= !nearly(t.env.log.base_stage_side_standard_kills, 0.0f);
    fail |= !nearly(t.env.log.base_stage_side_standard_timeouts, 1.0f);
    fail |= !nearly(t.env.log.base_stage_side_standard_ground, 0.0f);
    fail |= !nearly(t.env.log.base_stage_side_standard_episode_length, 20.0f);
    fail |= !nearly(t.env.log.base_stage_side_standard_action_saturation, 0.1f);
    fail |= !nearly(t.env.log.base_stage_side_standard_signed_bias, -0.25f);
    fail |= !nearly(t.env.log.base_stage_side_standard_action_sat_elevator, 0.1f);
    fail |= !nearly(t.env.log.base_stage_side_standard_action_sat_aileron, 0.2f);
    fail |= !nearly(t.env.log.base_stage_side_standard_action_sat_rudder, 0.0f);
    fail |= !nearly(t.env.log.base_stage_side_standard_action_sat_trigger, 0.3f);
    fail |= !nearly(t.env.log.base_stage_side_standard_signed_bias_elevator, -0.50f);
    fail |= !nearly(t.env.log.base_stage_side_standard_signed_bias_aileron, -0.25f);
    fail |= !nearly(t.env.log.base_stage_side_standard_signed_bias_rudder, 0.50f);

    fail |= !nearly(t.env.log.base_stage_side_energy_eps, 1.0f);
    fail |= !nearly(t.env.log.base_stage_side_energy_kills, 1.0f);
    fail |= !nearly(t.env.log.base_stage_side_energy_timeouts, 0.0f);
    fail |= !nearly(t.env.log.base_stage_side_energy_ground, 1.0f);
    fail |= !nearly(t.env.log.base_stage_side_energy_episode_length, 40.0f);
    fail |= !nearly(t.env.log.base_stage_side_energy_action_saturation, 0.06666667f);
    fail |= !nearly(t.env.log.base_stage_side_energy_signed_bias, 0.75f);
    fail |= !nearly(t.env.log.base_stage_side_energy_action_sat_elevator, 0.0f);
    fail |= !nearly(t.env.log.base_stage_side_energy_action_sat_aileron, 0.1f);
    fail |= !nearly(t.env.log.base_stage_side_energy_action_sat_rudder, 0.1f);
    fail |= !nearly(t.env.log.base_stage_side_energy_action_sat_trigger, 0.2f);
    fail |= !nearly(t.env.log.base_stage_side_energy_signed_bias_elevator, 1.25f);
    fail |= !nearly(t.env.log.base_stage_side_energy_signed_bias_aileron, 0.75f);
    fail |= !nearly(t.env.log.base_stage_side_energy_signed_bias_rudder, -1.50f);

    if (fail) {
        printf(
            "side_variant_telemetry: std eps=%.1f kill=%.1f timeout=%.1f ground=%.1f; energy eps=%.1f kill=%.1f timeout=%.1f ground=%.1f [FAIL]\n",
            t.env.log.base_stage_side_standard_eps,
            t.env.log.base_stage_side_standard_kills,
            t.env.log.base_stage_side_standard_timeouts,
            t.env.log.base_stage_side_standard_ground,
            t.env.log.base_stage_side_energy_eps,
            t.env.log.base_stage_side_energy_kills,
            t.env.log.base_stage_side_energy_timeouts,
            t.env.log.base_stage_side_energy_ground);
        return 1;
    }

    printf("side_variant_telemetry: mastery-stage standard vs energy counters [OK]\n");
    return 0;
}

int main(void) {
    srand(42);
    int fails = 0;
    fails += test_action_telemetry_accumulates_from_steps();
    fails += test_curriculum_quality_penalizes_surface_saturation();
    fails += test_low_alt_variant_log_for_stage3();
    fails += test_mastery_stage_telemetry_accumulates_current_stage_only();
    fails += test_side_spawn_variant_telemetry_mastery_only();
    return fails;
}
