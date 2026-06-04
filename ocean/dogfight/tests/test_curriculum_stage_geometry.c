#include <math.h>
#include <stdio.h>
#include <string.h>

#include "dogfight.h"

#define TEST_OBS_SIZE 26
#define TEST_NUM_ATNS 5
#define TEST_RAD_TO_DEG 57.29577951308232f

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

static void setup_curriculum_env(TestEnv* t, int stage) {
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
    set_curriculum_stage(&t->env, stage);
}

static int altitude_in_bounds(float alt) {
    return alt >= 300.0f && alt <= 4700.0f;
}

static int test_stage_8_17_spawn_safety_and_caps(void) {
    TestEnv t;
    for (int stage = CURRICULUM_SIDE_FAR; stage <= CURRICULUM_HARD_MANEUVERING; stage++) {
        setup_curriculum_env(&t, stage);

        for (int i = 0; i < 64; i++) {
            c_reset(&t.env);
            if (t.env.stage != stage || t.env.max_steps != STAGES[stage].max_steps) {
                printf(
                    "stage%d_spawn_caps: stage=%d max_steps=%d expected=%d [FAIL]\n",
                    stage, t.env.stage, t.env.max_steps, STAGES[stage].max_steps);
                return 1;
            }
            if (!altitude_in_bounds(t.env.player.pos.z)
                    || !altitude_in_bounds(t.env.opponent.pos.z)) {
                printf(
                    "stage%d_spawn_alt: player=%.1f opponent=%.1f [FAIL]\n",
                    stage, t.env.player.pos.z, t.env.opponent.pos.z);
                return 1;
            }

            const float neutral[TEST_NUM_ATNS] = {0.5f, 0.0f, 0.0f, 0.0f, -1.0f};
            for (int step = 0; step < 50; step++) {
                memcpy(t.env.actions, neutral, sizeof(neutral));
                c_step(&t.env);
                if (t.env.terminals[0] != 0.0f || t.env.death_reason != DEATH_NONE) {
                    printf(
                        "stage%d_neutral_trace: terminal=%.0f reason=%d step=%d [FAIL]\n",
                        stage, t.env.terminals[0], t.env.death_reason, step);
                    return 1;
                }
                if (t.env.player.pos.z < 100.0f || t.env.opponent.pos.z < 100.0f) {
                    printf(
                        "stage%d_neutral_ground: player=%.1f opponent=%.1f step=%d [FAIL]\n",
                        stage, t.env.player.pos.z, t.env.opponent.pos.z, step);
                    return 1;
                }
            }
        }
    }

    printf("stage8_17_spawn_safety: caps, altitude, short neutral traces [OK]\n");
    return 0;
}

static void count_side_variants(int stage, const RuntimeConfig* cfg, int* standard, int* energy) {
    TestEnv t;
    setup_curriculum_env(&t, stage);
    apply_runtime_config(&t.env, cfg);

    *standard = 0;
    *energy = 0;
    for (int i = 0; i < 200; i++) {
        c_reset(&t.env);
        if (t.env.side_spawn_variant == SIDE_SPAWN_STANDARD) {
            (*standard)++;
        } else if (t.env.side_spawn_variant == SIDE_SPAWN_ENERGY) {
            (*energy)++;
        }
    }
}

static int test_side_energy_spawn_probability_override(void) {
    RuntimeConfig defaults = default_runtime_config();
    if (fabsf(defaults.side_energy_spawn_prob - 0.2f) > 1e-4f) {
        printf("side_energy_prob_default: got %.2f expected 0.20 [FAIL]\n",
            defaults.side_energy_spawn_prob);
        return 1;
    }

    RuntimeConfig cfg = default_runtime_config();
    int standard = 0;
    int energy = 0;

    cfg.side_energy_spawn_prob = 0.0f;
    count_side_variants(CURRICULUM_SIDE_MANEUVERING, &cfg, &standard, &energy);
    if (standard == 0 || energy != 0) {
        printf("side_energy_prob_zero: standard=%d energy=%d [FAIL]\n", standard, energy);
        return 1;
    }

    cfg.side_energy_spawn_prob = 1.0f;
    count_side_variants(CURRICULUM_SIDE_MANEUVERING, &cfg, &standard, &energy);
    if (standard != 0 || energy == 0) {
        printf("side_energy_prob_one: standard=%d energy=%d [FAIL]\n", standard, energy);
        return 1;
    }

    printf("side_energy_spawn_prob: default and forced variants [OK]\n");
    return 0;
}

static int sample_stage9_standard_bank(float target, const RuntimeConfig* cfg, float* bank_deg, int* mode) {
    TestEnv t;
    setup_curriculum_env(&t, CURRICULUM_SIDE_MANEUVERING);
    apply_runtime_config(&t.env, cfg);
    set_curriculum_target(&t.env, target);

    for (int i = 0; i < 500; i++) {
        c_reset(&t.env);
        if (t.env.stage != CURRICULUM_SIDE_MANEUVERING) continue;
        if (t.env.side_spawn_variant != SIDE_SPAWN_STANDARD) continue;
        *bank_deg = t.env.opponent_ap.target_bank * TEST_RAD_TO_DEG;
        *mode = t.env.opponent_ap.mode;
        return 1;
    }
    return 0;
}

static int test_stage9_bank_curriculum_and_override_scope(void) {
    RuntimeConfig defaults = default_runtime_config();
    if (fabsf(defaults.stage9_bank_deg - -1.0f) > 1e-4f) {
        printf("stage9_bank_auto_default: got %.1f expected -1 auto [FAIL]\n",
            defaults.stage9_bank_deg);
        return 1;
    }

    RuntimeConfig cfg = default_runtime_config();
    cfg.stage9_bank_deg = -1.0f;
    const float targets[] = {8.5f, 8.6f, 8.7f, 8.8f, 8.9f, 9.0f};
    const float expected[] = {0.0f, 5.0f, 10.0f, 15.0f, 30.0f, 30.0f};
    for (int i = 0; i < 6; i++) {
        float bank_deg = -999.0f;
        int mode = -1;
        if (!sample_stage9_standard_bank(targets[i], &cfg, &bank_deg, &mode)) {
            printf("stage9_bank_auto: no stage-9 standard sample for target %.1f [FAIL]\n",
                targets[i]);
            return 1;
        }
        if (fabsf(bank_deg - expected[i]) > 1e-3f) {
            printf("stage9_bank_auto: target %.1f got %.1f expected %.1f [FAIL]\n",
                targets[i], bank_deg, expected[i]);
            return 1;
        }
        if ((expected[i] == 0.0f && mode != AP_STRAIGHT)
                || (expected[i] > 0.0f && mode == AP_STRAIGHT)) {
            printf("stage9_bank_auto: target %.1f bank %.1f mode=%d [FAIL]\n",
                targets[i], bank_deg, mode);
            return 1;
        }
    }

    cfg.stage9_bank_deg = 10.0f;
    float bank_deg = -999.0f;
    int mode = -1;
    if (!sample_stage9_standard_bank(8.5f, &cfg, &bank_deg, &mode)) {
        printf("stage9_bank_override: no standard stage-9 side spawn sampled [FAIL]\n");
        return 1;
    }
    if (fabsf(bank_deg - 10.0f) > 1e-3f || mode == AP_STRAIGHT) {
        printf("stage9_bank_override: got %.1f mode=%d expected 10 turning [FAIL]\n",
            bank_deg, mode);
        return 1;
    }

    printf("stage9_bank_curriculum: auto schedule and override [OK]\n");
    return 0;
}

static int test_stage10_12_vertical_advantage_geometry(void) {
    TestEnv t;

    if (STAGES[CURRICULUM_DIVE_ATTACK].angle_min_deg != 120.0f
            || STAGES[CURRICULUM_DIVE_ATTACK].angle_max_deg != 175.0f) {
        printf(
            "stage10_dive_angles: expected Dogfight3 [120,175], got [%.0f,%.0f] [FAIL]\n",
            STAGES[CURRICULUM_DIVE_ATTACK].angle_min_deg,
            STAGES[CURRICULUM_DIVE_ATTACK].angle_max_deg);
        return 1;
    }

    setup_curriculum_env(&t, CURRICULUM_DIVE_ATTACK);
    for (int i = 0; i < 100; i++) {
        c_reset(&t.env);
        float alt_delta = t.env.player.pos.z - t.env.opponent.pos.z;
        if (alt_delta < 430.0f || alt_delta > 570.0f) {
            printf("stage10_dive_alt_delta: %.1f [FAIL]\n", alt_delta);
            return 1;
        }
    }

    setup_curriculum_env(&t, CURRICULUM_ZOOM_ATTACK);
    for (int i = 0; i < 100; i++) {
        c_reset(&t.env);
        float alt_delta = t.env.opponent.pos.z - t.env.player.pos.z;
        if (alt_delta < 250.0f || alt_delta > 350.0f) {
            printf("stage11_zoom_alt_delta: %.1f [FAIL]\n", alt_delta);
            return 1;
        }
    }

    setup_curriculum_env(&t, CURRICULUM_REAR_CHASE);
    for (int i = 0; i < 100; i++) {
        c_reset(&t.env);
        float alt_delta = t.env.player.pos.z - t.env.opponent.pos.z;
        if (alt_delta < 430.0f || alt_delta > 570.0f) {
            printf("stage12_rear_alt_delta: %.1f [FAIL]\n", alt_delta);
            return 1;
        }
    }

    printf("stage10_12_vertical_geometry: altitude-advantage spawns [OK]\n");
    return 0;
}

int main(void) {
    srand(42);
    int fails = 0;
    fails += test_stage_8_17_spawn_safety_and_caps();
    fails += test_side_energy_spawn_probability_override();
    fails += test_stage9_bank_curriculum_and_override_scope();
    fails += test_stage10_12_vertical_advantage_geometry();
    return fails;
}
