#include "test_common.h"

#include <stdarg.h>

static int failf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fputc('\n', stderr);
    return 1;
}

static int nearly_equal(float a, float b, float tolerance) {
    return fabsf(a - b) <= tolerance;
}

static float horizontal_heading_deg(Vec3 velocity) {
    return atan2f(velocity.y, velocity.x) * RAD;
}

static int reset_at_stage(TestEnv* test, int stage, unsigned int seed) {
    test->env.rng = seed;
    test->env.curriculum_enabled = 1;
    test->env.curriculum_randomize = 0;
    test->env.configured_max_steps = 777;
    test->env.max_steps = 777;
    set_curriculum_stage(&test->env, stage);
    c_reset(&test->env);

    if (test->env.stage != stage) {
        return failf(
            "fixed stage changed during reset: requested=%d actual=%d",
            stage,
            test->env.stage
        );
    }
    return 0;
}

static int test_stage_table_contract(void) {
    if (CURRICULUM_COUNT != 21) {
        return failf("expected 21 curriculum stages, got %d", CURRICULUM_COUNT);
    }

    for (int stage = 0; stage < CURRICULUM_COUNT; stage++) {
        const StageConfig* config = &STAGES[stage];
        if (config->n != stage) {
            return failf("stage table index mismatch: index=%d n=%d", stage, config->n);
        }
        if (config->spawn == NULL || config->description == NULL) {
            return failf("stage %d has an incomplete table entry", stage);
        }
        if (config->weight < 0.0f || config->weight > 1.0f) {
            return failf("stage %d weight is outside [0, 1]: %.6f", stage, config->weight);
        }
        if (config->max_steps <= 0) {
            return failf("stage %d max_steps must be positive", stage);
        }
        if (stage > 0 && config->weight < STAGES[stage - 1].weight) {
            return failf("stage weights decrease at stage %d", stage);
        }
    }

    const StageConfig* hard = &STAGES[CURRICULUM_HARD_MANEUVERING];
    const StageConfig* crossing = &STAGES[CURRICULUM_CROSSING];
    const StageConfig* evasive = &STAGES[CURRICULUM_EVASIVE];
    const StageConfig* autoace = &STAGES[CURRICULUM_AUTOACE];

    if (!nearly_equal(hard->weight, 0.90f, 1e-6f)
            || hard->max_steps != 4000
            || hard->angle_min_deg != 0.0f
            || hard->angle_max_deg != 360.0f
            || hard->bank != 60) {
        return failf("stage 17 metadata drifted");
    }
    if (!nearly_equal(crossing->weight, 0.95f, 1e-6f)
            || crossing->max_steps != 4000
            || crossing->angle_min_deg != 45.0f
            || crossing->angle_max_deg != 45.0f
            || crossing->bank != 0) {
        return failf("stage 18 metadata drifted");
    }
    if (!nearly_equal(evasive->weight, 1.0f, 1e-6f)
            || evasive->max_steps != 4000
            || evasive->angle_min_deg != 0.0f
            || evasive->angle_max_deg != 360.0f
            || evasive->bank != 60) {
        return failf("stage 19 metadata drifted");
    }
    if (!nearly_equal(autoace->weight, 1.0f, 1e-6f)
            || autoace->max_steps != 6000
            || autoace->angle_min_deg != 0.0f
            || autoace->angle_max_deg != 360.0f
            || autoace->bank != 0) {
        return failf("stage 20 metadata drifted");
    }

    return 0;
}

static int test_episode_length_boundary(void) {
    TestEnv test;
    setup_env(&test, 0);

    if (reset_at_stage(&test, CURRICULUM_SIDE_MID, 7001) != 0) {
        return 1;
    }
    if (test.env.max_steps != 777) {
        return failf(
            "stage 7 must preserve configured max_steps: expected=777 actual=%d",
            test.env.max_steps
        );
    }

    if (reset_at_stage(&test, CURRICULUM_SIDE_FAR, 7002) != 0) {
        return 1;
    }
    if (test.env.max_steps != STAGES[CURRICULUM_SIDE_FAR].max_steps) {
        return failf(
            "stage 8 must use stage max_steps: expected=%d actual=%d",
            STAGES[CURRICULUM_SIDE_FAR].max_steps,
            test.env.max_steps
        );
    }

    return 0;
}

static int test_stage_17_hard_maneuvering(void) {
    TestEnv test;
    setup_env(&test, 0);

    int saw_left = 0;
    int saw_right = 0;
    int saw_weave = 0;
    for (int sample = 0; sample < 256; sample++) {
        if (reset_at_stage(&test, CURRICULUM_HARD_MANEUVERING, 17000 + sample) != 0) {
            return 1;
        }

        Vec3 relative = sub3(test.env.opponent.pos, test.env.player.pos);
        if (relative.x < 200.0f || relative.x > 400.0f
                || fabsf(relative.y) > 100.0f
                || fabsf(relative.z) > 50.0f) {
            return failf(
                "stage 17 spawn escaped geometry: relative=(%.3f, %.3f, %.3f)",
                relative.x,
                relative.y,
                relative.z
            );
        }
        if (test.env.max_steps != 4000) {
            return failf("stage 17 max_steps drifted: %d", test.env.max_steps);
        }

        switch (test.env.opponent_ap.mode) {
            case AP_HARD_TURN_LEFT: saw_left = 1; break;
            case AP_HARD_TURN_RIGHT: saw_right = 1; break;
            case AP_WEAVE: saw_weave = 1; break;
            default:
                return failf(
                    "stage 17 selected invalid autopilot mode %d",
                    test.env.opponent_ap.mode
                );
        }
    }

    if (!saw_left || !saw_right || !saw_weave) {
        return failf(
            "stage 17 sampling missed a hard mode: left=%d right=%d weave=%d",
            saw_left,
            saw_right,
            saw_weave
        );
    }
    return 0;
}

static int test_stage_18_crossing(void) {
    TestEnv test;
    setup_env(&test, 0);

    int saw_left = 0;
    int saw_right = 0;
    for (int sample = 0; sample < 128; sample++) {
        if (reset_at_stage(&test, CURRICULUM_CROSSING, 18000 + sample) != 0) {
            return 1;
        }

        Vec3 relative = sub3(test.env.opponent.pos, test.env.player.pos);
        float heading = horizontal_heading_deg(test.env.opponent.vel);
        if (relative.x < 100.0f || relative.x > 200.0f
                || fabsf(relative.y) < 300.0f
                || fabsf(relative.y) > 500.0f
                || fabsf(relative.z) > 50.0f) {
            return failf(
                "stage 18 spawn escaped geometry: relative=(%.3f, %.3f, %.3f)",
                relative.x,
                relative.y,
                relative.z
            );
        }
        if (!nearly_equal(fabsf(heading), 45.0f, 0.02f)) {
            return failf("stage 18 heading is not 45 degrees: %.6f", heading);
        }
        if (relative.y * test.env.opponent.vel.y >= 0.0f) {
            return failf("stage 18 opponent is flying away from the crossing");
        }
        if (test.env.opponent_ap.mode != AP_STRAIGHT || test.env.max_steps != 4000) {
            return failf(
                "stage 18 behavior drifted: mode=%d max_steps=%d",
                test.env.opponent_ap.mode,
                test.env.max_steps
            );
        }

        saw_left |= relative.y < 0.0f;
        saw_right |= relative.y > 0.0f;
    }

    if (!saw_left || !saw_right) {
        return failf("stage 18 did not sample both crossing sides");
    }
    return 0;
}

static int test_stage_19_evasive(void) {
    TestEnv test;
    setup_env(&test, 0);

    int saw_evasive = 0;
    int saw_other_hard_mode = 0;
    for (int sample = 0; sample < 256; sample++) {
        if (reset_at_stage(&test, CURRICULUM_EVASIVE, 19000 + sample) != 0) {
            return 1;
        }

        Vec3 relative = sub3(test.env.opponent.pos, test.env.player.pos);
        float distance = norm3(relative);
        if (test.env.player.pos.z < 3500.0f || test.env.player.pos.z > 4500.0f
                || test.env.opponent.pos.z < 2500.0f
                || test.env.opponent.pos.z > 4800.0f
                || distance < 300.0f
                || distance > 500.0f) {
            return failf(
                "stage 19 spawn escaped geometry: player_z=%.3f opponent_z=%.3f distance=%.3f",
                test.env.player.pos.z,
                test.env.opponent.pos.z,
                distance
            );
        }
        if (test.env.max_steps != 4000) {
            return failf("stage 19 max_steps drifted: %d", test.env.max_steps);
        }

        switch (test.env.opponent_ap.mode) {
            case AP_EVASIVE:
                saw_evasive = 1;
                break;
            case AP_HARD_TURN_LEFT:
            case AP_HARD_TURN_RIGHT:
            case AP_WEAVE:
            case AP_TURN_LEFT:
            case AP_TURN_RIGHT:
                saw_other_hard_mode = 1;
                break;
            default:
                return failf(
                    "stage 19 selected invalid autopilot mode %d",
                    test.env.opponent_ap.mode
                );
        }
    }

    if (!saw_evasive || !saw_other_hard_mode) {
        return failf(
            "stage 19 sampling missed expected mode classes: evasive=%d other=%d",
            saw_evasive,
            saw_other_hard_mode
        );
    }
    return 0;
}

static int test_stage_20_autoace(void) {
    TestEnv test;
    setup_env(&test, 0);

    for (int sample = 0; sample < 128; sample++) {
        if (reset_at_stage(&test, CURRICULUM_AUTOACE, 20000 + sample) != 0) {
            return 1;
        }

        Vec3 relative = sub3(test.env.opponent.pos, test.env.player.pos);
        float distance = norm3(relative);
        if (test.env.player.pos.z < 2500.0f || test.env.player.pos.z > 4000.0f
                || test.env.opponent.pos.z < 2000.0f
                || test.env.opponent.pos.z > 4500.0f
                || distance < 400.0f
                || distance > 700.0f) {
            return failf(
                "stage 20 spawn escaped geometry: player_z=%.3f opponent_z=%.3f distance=%.3f",
                test.env.player.pos.z,
                test.env.opponent.pos.z,
                distance
            );
        }
        if (test.env.opponent_ap.mode != AP_PURSUIT_LAG || test.env.max_steps != 6000) {
            return failf(
                "stage 20 behavior drifted: mode=%d max_steps=%d",
                test.env.opponent_ap.mode,
                test.env.max_steps
            );
        }
    }

    if (reset_at_stage(&test, CURRICULUM_AUTOACE, 20999) != 0) {
        return 1;
    }
    memset(test.actions, 0, sizeof(test.actions));
    for (int step = 0; step < 250; step++) {
        t_step(&test);
        if (!isfinite(test.env.player.pos.x)
                || !isfinite(test.env.player.pos.y)
                || !isfinite(test.env.player.pos.z)
                || !isfinite(test.env.player.vel.x)
                || !isfinite(test.env.player.vel.y)
                || !isfinite(test.env.player.vel.z)
                || !isfinite(test.env.opponent.pos.x)
                || !isfinite(test.env.opponent.pos.y)
                || !isfinite(test.env.opponent.pos.z)
                || !isfinite(test.env.opponent.vel.x)
                || !isfinite(test.env.opponent.vel.y)
                || !isfinite(test.env.opponent.vel.z)) {
            return failf("stage 20 produced non-finite flight state at step %d", step);
        }
        for (int observation = 0; observation < TEST_OBS_SIZE; observation++) {
            if (!isfinite(test.observations[observation])) {
                return failf(
                    "stage 20 produced non-finite observation %d at step %d",
                    observation,
                    step
                );
            }
        }
    }

    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_stage_table_contract();
    failures += test_episode_length_boundary();
    failures += test_stage_17_hard_maneuvering();
    failures += test_stage_18_crossing();
    failures += test_stage_19_evasive();
    failures += test_stage_20_autoace();

    if (failures == 0) {
        printf("curriculum stage geometry: PASS\n");
    }
    return failures;
}
