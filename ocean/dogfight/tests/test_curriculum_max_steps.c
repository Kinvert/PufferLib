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

static void setup_curriculum_target(TestEnv* t, float target) {
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
    set_curriculum_target(&t->env, target);
}

static void setup_curriculum_target_with_max_steps(TestEnv* t, float target, int max_steps) {
    setup_curriculum_target(t, target);
    t->env.configured_max_steps = max_steps;
    t->env.max_steps = max_steps;
}

static int expect_stage_cap(float target, CurriculumStage stage, int expected_max_steps,
        const char* label) {
    TestEnv t;
    setup_curriculum_target(&t, target);

    for (int i = 0; i < 1000; i++) {
        c_reset(&t.env);
        if (t.env.stage != stage || t.env.low_altitude_variant ||
                t.env.max_steps != expected_max_steps) {
            printf("%s: stage=%d low_alt=%d max_steps=%d expected_stage=%d expected_steps=%d [FAIL]\n",
                label, t.env.stage, t.env.low_altitude_variant, t.env.max_steps,
                stage, expected_max_steps);
            return 1;
        }
    }

    printf("%s: curriculum max_steps [OK]\n", label);
    return 0;
}

static int expect_low_altitude_variant(float target, CurriculumStage stage,
        const char* label) {
    TestEnv t;
    setup_curriculum_target(&t, target);
    int standard = 0;
    int low_alt = 0;

    for (int i = 0; i < 1000; i++) {
        c_reset(&t.env);
        if (t.env.stage != stage) {
            printf("%s: stage=%d expected_stage=%d [FAIL]\n",
                label, t.env.stage, stage);
            return 1;
        }

        if (t.env.max_steps == t.env.configured_max_steps) {
            standard++;
            continue;
        }

        if (t.env.max_steps == 2000 && t.env.player.pos.z == 400.0f &&
                t.env.opponent.pos.z >= 700.0f && t.env.opponent.pos.z <= 900.0f) {
            low_alt++;
            continue;
        }

        printf(
            "%s: unexpected max_steps=%d player_z=%.1f opponent_z=%.1f [FAIL]\n",
            label, t.env.max_steps, t.env.player.pos.z, t.env.opponent.pos.z);
        return 1;
    }

    if (standard == 0 || low_alt == 0) {
        printf("%s: standard=%d low_alt=%d [FAIL]\n", label, standard, low_alt);
        return 1;
    }

    printf("%s: sampled standard=%d low_alt=%d [OK]\n", label, standard, low_alt);
    return 0;
}

static int expect_configured_or_low_alt_cap(float target, CurriculumStage stage,
        int configured_max_steps, const char* label) {
    TestEnv t;
    setup_curriculum_target_with_max_steps(&t, target, configured_max_steps);
    int standard = 0;
    int low_alt = 0;

    for (int i = 0; i < 1000; i++) {
        c_reset(&t.env);
        if (t.env.stage != stage) {
            printf("%s: stage=%d expected_stage=%d [FAIL]\n",
                label, t.env.stage, stage);
            return 1;
        }

        if (t.env.max_steps == configured_max_steps) {
            standard++;
        } else if (t.env.max_steps == 2000 && t.env.player.pos.z == 400.0f) {
            low_alt++;
        } else {
            printf("%s: max_steps=%d expected=%d or 2000 [FAIL]\n",
                label, t.env.max_steps, configured_max_steps);
            return 1;
        }
    }

    if (standard == 0 || low_alt == 0) {
        printf("%s: standard=%d low_alt=%d [FAIL]\n", label, standard, low_alt);
        return 1;
    }

    printf("%s: configured and low-alt caps [OK]\n", label);
    return 0;
}

static int test_stage_table_preserves_reference_early_caps(void) {
    if (STAGES[CURRICULUM_HEAD_ON].max_steps != 300 ||
            STAGES[CURRICULUM_VERTICAL].max_steps != 500) {
        printf(
            "stage_table_caps: head_on=%d vertical=%d expected=300/500 [FAIL]\n",
            STAGES[CURRICULUM_HEAD_ON].max_steps,
            STAGES[CURRICULUM_VERTICAL].max_steps);
        return 1;
    }

    printf("stage_table_caps: reference early stage caps [OK]\n");
    return 0;
}

static int test_early_stages_use_configured_episode_cap(void) {
    const struct {
        float target;
        CurriculumStage stage;
        const char* label;
    } cases[] = {
        {2.0f, CURRICULUM_VERTICAL, "stage2_configured_max_steps"},
        {(float)CURRICULUM_SIDE_NEAR, CURRICULUM_SIDE_NEAR, "stage6_configured_max_steps"},
        {(float)CURRICULUM_SIDE_MID, CURRICULUM_SIDE_MID, "stage7_configured_max_steps"},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        TestEnv t;
        setup_curriculum_target_with_max_steps(&t, cases[i].target, 3000);
        for (int j = 0; j < 1000; j++) {
            c_reset(&t.env);
            if (t.env.stage != cases[i].stage || t.env.max_steps != 3000) {
                printf("%s: stage=%d max_steps=%d expected_stage=%d expected_steps=3000 [FAIL]\n",
                    cases[i].label, t.env.stage, t.env.max_steps, cases[i].stage);
                return 1;
            }
        }
    }

    printf("early_configured_max_steps: stages 0-7 keep configured cap [OK]\n");
    return 0;
}

int main(void) {
    srand(42);
    int fails = 0;
    fails += test_stage_table_preserves_reference_early_caps();
    fails += test_early_stages_use_configured_episode_cap();
    fails += expect_low_altitude_variant(0.0f, CURRICULUM_TAIL_CHASE,
        "stage0_low_alt_variant");
    fails += expect_low_altitude_variant(1.0f, CURRICULUM_HEAD_ON,
        "stage1_low_alt_variant");
    fails += expect_configured_or_low_alt_cap(0.0f, CURRICULUM_TAIL_CHASE, 3000,
        "stage0_configured_or_low_alt_max_steps");
    fails += expect_configured_or_low_alt_cap(1.0f, CURRICULUM_HEAD_ON, 3000,
        "stage1_configured_or_low_alt_max_steps");
    fails += expect_stage_cap(2.0f, CURRICULUM_VERTICAL, 300, "stage2_max_steps");
    fails += expect_stage_cap((float)CURRICULUM_SIDE_NEAR, CURRICULUM_SIDE_NEAR, 300,
        "stage6_max_steps");
    fails += expect_stage_cap((float)CURRICULUM_SIDE_MID, CURRICULUM_SIDE_MID, 300,
        "stage7_max_steps");
    return fails;
}
