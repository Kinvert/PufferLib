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

static void sample_spawn(
        TestEnv* test, long global_step, unsigned int seed, int num_agents) {
    Vec3 position = vec3(0.0f, 0.0f, 2500.0f);
    Vec3 velocity = vec3(80.0f, 0.0f, 0.0f);
    test->env.rng = seed;
    dogfight_bind_rng(&test->env.rng);
    test->env.num_agents = num_agents;
    test->env.curriculum_enabled = 1;
    test->env.curriculum_randomize = 0;
    test->env.curriculum_target = 0.0f;
    test->env.native_spawn_curriculum = 1;
    test->env.native_acquisition_steps = 100000;
    test->env.native_spawn_total_steps = 1000000;
    test->env.global_step = global_step;
    test->env.configured_max_steps = 300;
    test->env.max_steps = test->env.configured_max_steps;
    reset_plane(&test->env.player, position, velocity);
    spawn_by_curriculum(&test->env, position, velocity);
}

static float horizontal_bearing_deg(const Dogfight* env) {
    Vec3 relative = sub3(env->opponent.pos, env->player.pos);
    return atan2f(fabsf(relative.y), relative.x) * RAD;
}

static float elevation_deg(const Dogfight* env) {
    Vec3 relative = sub3(env->opponent.pos, env->player.pos);
    float horizontal = sqrtf(
        relative.x * relative.x + relative.y * relative.y);
    return atan2f(fabsf(relative.z), horizontal) * RAD;
}

static int test_easy_frontier(void) {
    TestEnv test;
    setup_env(&test, 0);

    int left = 0;
    int right = 0;
    int low_energy = 0;
    for (int sample = 0; sample < 512; sample++) {
        sample_spawn(&test, 0, 10000U + (unsigned int)sample, 2);
        Vec3 relative = sub3(test.env.opponent.pos, test.env.player.pos);
        float bearing = horizontal_bearing_deg(&test.env);
        float speed_delta = fabsf(
            norm3(test.env.player.vel) - norm3(test.env.opponent.vel));

        if (test.env.stage != 0) {
            return failf("easy frontier selected stage %d", test.env.stage);
        }
        if (relative.x < 199.99f || relative.x > 400.01f
                || fabsf(relative.y) < 25.99f
                || fabsf(relative.y) > 50.01f) {
            return failf(
                "stage-0 tail geometry escaped range: x=%.6f y=%.6f",
                relative.x,
                relative.y);
        }
        if (bearing < 3.70f || bearing > 14.04f) {
            return failf("stage-0 bearing escaped range: %.6f", bearing);
        }
        if (speed_delta > 0.01f) {
            return failf("stage-0 spawn changed relative speed: %.6f", speed_delta);
        }
        int is_low_energy = test.env.player.pos.z < 500.0f;
        if (test.env.max_steps != (is_low_energy ? 2000 : 300)) {
            return failf(
                "stage-0 max_steps mismatch: low=%d steps=%d",
                is_low_energy,
                test.env.max_steps);
        }
        left += relative.y < 0.0f;
        right += relative.y > 0.0f;
        low_energy += is_low_energy;
    }

    if (left < 200 || right < 200 || low_energy < 60 || low_energy > 150) {
        return failf(
            "stage-0 mixture balance failed: left=%d right=%d low=%d",
            left,
            right,
            low_energy);
    }
    return 0;
}

static int test_easy_frontier_preserves_spawn_rng(void) {
    TestEnv progressive;
    TestEnv direct;
    setup_env(&progressive, 0);
    setup_env(&direct, 0);

    const unsigned int seed = 424242U;
    Vec3 position = vec3(0.0f, 0.0f, 2500.0f);
    Vec3 velocity = vec3(80.0f, 0.0f, 0.0f);
    sample_spawn(&progressive, 0, seed, 2);

    direct.env.rng = seed;
    dogfight_bind_rng(&direct.env.rng);
    direct.env.num_agents = 2;
    direct.env.configured_max_steps = 300;
    direct.env.max_steps = direct.env.configured_max_steps;
    reset_plane(&direct.env.player, position, velocity);
    STAGES[0].spawn(&direct.env, position, velocity);

    if (progressive.env.rng != direct.env.rng) {
        return failf(
            "stage-0 selector consumed spawn RNG: progressive=%u direct=%u",
            progressive.env.rng,
            direct.env.rng);
    }
    return 0;
}

static int test_final_frontier(void) {
    TestEnv test;
    setup_env(&test, 0);

    int final_level = 0;
    int rear_geometry = 0;
    int vertical_geometry = 0;
    for (int sample = 0; sample < 512; sample++) {
        sample_spawn(&test, 1050000, 20000U + (unsigned int)sample, 2);

        if (test.env.stage < 0 || test.env.stage > 10) {
            return failf("final schedule selected invalid stage %d", test.env.stage);
        }
        if (!isfinite(test.env.player.pos.x)
                || !isfinite(test.env.opponent.pos.x)
                || !isfinite(test.env.player.vel.x)
                || !isfinite(test.env.opponent.vel.x)) {
            return failf("final schedule produced non-finite state");
        }

        if (test.env.stage == 10) {
            float bearing = horizontal_bearing_deg(&test.env);
            float elevation = elevation_deg(&test.env);
            float speed_delta = fabsf(
                norm3(test.env.player.vel) - norm3(test.env.opponent.vel));
            final_level++;
            rear_geometry += bearing > 90.0f;
            vertical_geometry += elevation > 5.0f;

            if (bearing < 119.99f || bearing > 175.01f) {
                return failf("final bearing escaped range: %.6f", bearing);
            }
            if (elevation < 41.98f || elevation > 61.40f) {
                return failf("final elevation escaped range: %.6f", elevation);
            }
            if (speed_delta > 0.01f) {
                return failf("final spawn retained speed advantage: %.6f", speed_delta);
            }
            if (test.env.max_steps != 2500) {
                return failf(
                    "final max_steps must be 2500, got %d", test.env.max_steps);
            }
        }
    }

    if (final_level < 300
            || rear_geometry != final_level
            || vertical_geometry != final_level) {
        return failf(
            "final mixture lacks diversity: final=%d rear=%d vertical=%d",
            final_level,
            rear_geometry,
            vertical_geometry);
    }
    return 0;
}

static int test_uniform_consolidation(void) {
    TestEnv test;
    setup_env(&test, 0);

    int counts[11] = {0};
    for (int sample = 0; sample < 2048; sample++) {
        sample_spawn(&test, 1100000, 40000U + (unsigned int)sample, 2);
        if (test.env.stage < 0 || test.env.stage > 10) {
            return failf(
                "consolidation selected invalid stage %d", test.env.stage);
        }
        counts[test.env.stage]++;
    }

    for (int stage = 0; stage <= 10; stage++) {
        if (counts[stage] < 100 || counts[stage] > 280) {
            return failf(
                "consolidation stage %d count escaped uniform bounds: %d",
                stage,
                counts[stage]);
        }
    }
    return 0;
}

static int test_configurable_frontier_fraction(void) {
    TestEnv test;
    setup_env(&test, 0);
    test.env.native_frontier_fraction = 0.5f;

    int frontier_count = 0;
    for (int sample = 0; sample < 512; sample++) {
        unsigned int seed = 2654435761U * ((unsigned int)sample + 1U);
        sample_spawn(&test, 1050000, seed, 2);
        frontier_count += test.env.stage == 10;
    }

    if (frontier_count < 220 || frontier_count > 300) {
        return failf(
            "configured 50%% frontier fraction drifted: %d", frontier_count);
    }
    return 0;
}

static int test_repeatability(void) {
    TestEnv test;
    setup_env(&test, 0);

    sample_spawn(&test, 600000, 424242U, 2);
    CurriculumStage stage = test.env.stage;
    Vec3 player_position = test.env.player.pos;
    Vec3 player_velocity = test.env.player.vel;
    Vec3 opponent_position = test.env.opponent.pos;
    Vec3 opponent_velocity = test.env.opponent.vel;

    sample_spawn(&test, 600000, 424242U, 2);
    if (test.env.stage != stage
            || memcmp(&test.env.player.pos, &player_position, sizeof(Vec3)) != 0
            || memcmp(&test.env.player.vel, &player_velocity, sizeof(Vec3)) != 0
            || memcmp(&test.env.opponent.pos, &opponent_position, sizeof(Vec3)) != 0
            || memcmp(&test.env.opponent.vel, &opponent_velocity, sizeof(Vec3)) != 0) {
        return failf("same seed and step did not reproduce the spawn exactly");
    }
    return 0;
}

static int test_fixed_eval_is_unchanged(void) {
    TestEnv test;
    setup_env(&test, 0);

    for (int sample = 0; sample < 128; sample++) {
        sample_spawn(&test, 1000000, 30000U + (unsigned int)sample, 1);
        float bearing = horizontal_bearing_deg(&test.env);
        if (test.env.stage != 0) {
            return failf("fixed eval changed stage: %d", test.env.stage);
        }
        if (bearing > 15.0f) {
            return failf(
                "fixed eval used native training geometry: bearing=%.6f",
                bearing);
        }
    }
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_easy_frontier();
    failures += test_easy_frontier_preserves_spawn_rng();
    failures += test_final_frontier();
    failures += test_uniform_consolidation();
    failures += test_configurable_frontier_fraction();
    failures += test_repeatability();
    failures += test_fixed_eval_is_unchanged();

    if (failures == 0) {
        printf("native self-play spawn: PASS\n");
    }
    return failures;
}
