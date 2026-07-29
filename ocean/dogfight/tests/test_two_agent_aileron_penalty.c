#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../dogfight.h"

static Plane level_plane(float x) {
    Plane plane;
    memset(&plane, 0, sizeof(plane));
    plane.pos = vec3(x, 0.0f, 2000.0f);
    plane.vel = vec3(80.0f, 0.0f, 0.0f);
    plane.ori.w = 1.0f;
    plane.g_force = 1.0f;
    return plane;
}

static int test_magnitude_penalty(void) {
    Dogfight neutral_env;
    Dogfight deflected_env;
    memset(&neutral_env, 0, sizeof(neutral_env));
    memset(&deflected_env, 0, sizeof(deflected_env));
    neutral_env.rcfg.aileron_magnitude_penalty = 0.01f;
    deflected_env.rcfg.aileron_magnitude_penalty = 0.01f;

    Plane self = level_plane(0.0f);
    Plane other = level_plane(100.0f);
    float neutral_actions[NUM_ATNS] = {0};
    float deflected_actions[NUM_ATNS] = {0};
    deflected_actions[2] = 0.75f;

    float neutral = dogfight_two_agent_dense_reward(
        &neutral_env, &self, &other, neutral_actions, 0, 0.0f, 0.0f);
    float deflected = dogfight_two_agent_dense_reward(
        &deflected_env, &self, &other, deflected_actions, 0, 0.0f, 0.0f);
    float expected_delta = 0.75f * 0.01f;
    float actual_delta = neutral - deflected;

    if (fabsf(actual_delta - expected_delta) > 1.0e-6f) {
        fprintf(stderr,
            "aileron penalty delta %.8f, expected %.8f\n",
            actual_delta, expected_delta);
        return 1;
    }
    return 0;
}

static int test_mirrored_steering_alignment(void) {
    const float scale = 0.02f;
    float left_correct = dogfight_two_agent_steering_alignment_reward(
        0.25f, -1.0f, scale);
    float left_wrong = dogfight_two_agent_steering_alignment_reward(
        0.25f, 1.0f, scale);
    float right_correct = dogfight_two_agent_steering_alignment_reward(
        -0.25f, 1.0f, scale);
    float right_wrong = dogfight_two_agent_steering_alignment_reward(
        -0.25f, -1.0f, scale);
    float neutral = dogfight_two_agent_steering_alignment_reward(
        0.25f, 0.0f, scale);

    if (fabsf(left_correct - right_correct) > 1.0e-6f
            || fabsf(left_wrong - right_wrong) > 1.0e-6f
            || left_correct <= neutral
            || neutral <= left_wrong
            || fabsf(neutral) > 1.0e-6f) {
        fprintf(stderr,
            "mirrored steering alignment failed: "
            "left=(%.6f, %.6f) right=(%.6f, %.6f) neutral=%.6f\n",
            left_correct,
            left_wrong,
            right_correct,
            right_wrong,
            neutral);
        return 1;
    }
    return 0;
}

static int test_steering_alignment_decay(void) {
    const float configured = 0.20f;
    float initial = dogfight_two_agent_steering_scale_for_stage(configured, 0);
    float middle = dogfight_two_agent_steering_scale_for_stage(configured, 5);
    float final = dogfight_two_agent_steering_scale_for_stage(configured, 10);
    if (fabsf(initial - 0.20f) > 1.0e-6f
            || fabsf(middle - 0.10f) > 1.0e-6f
            || fabsf(final) > 1.0e-6f) {
        fprintf(stderr,
            "steering alignment decay failed: %.6f %.6f %.6f\n",
            initial,
            middle,
            final);
        return 1;
    }
    return 0;
}

static int test_independent_steering_publication(void) {
    float left_reward = dogfight_two_agent_publish_steering_reward(
        0.0f, 0.25f, -1.0f, 0.20f);
    float right_reward = dogfight_two_agent_publish_steering_reward(
        0.0f, -0.25f, 1.0f, 0.20f);
    if (left_reward <= 0.0f || right_reward <= 0.0f) {
        fprintf(stderr,
            "independent steering publication failed: %.6f %.6f\n",
            left_reward,
            right_reward);
        return 1;
    }
    return 0;
}

static int test_native_acquisition_contract(void) {
    Dogfight env;
    memset(&env, 0, sizeof(env));
    env.native_spawn_curriculum = 1;
    env.native_acquisition_steps = 64;
    env.native_acquisition_rehearsal_cycle_steps = 32;
    env.native_acquisition_rehearsal_steps = 4;
    env.global_step = 63;
    if (!dogfight_two_agent_in_native_acquisition(&env)) {
        fprintf(stderr, "native acquisition ended before configured step\n");
        return 1;
    }
    env.global_step = 64;
    if (dogfight_two_agent_in_native_acquisition(&env)) {
        fprintf(stderr, "native acquisition continued into competitive phase\n");
        return 1;
    }
    env.global_step = 91;
    if (dogfight_two_agent_in_native_acquisition(&env)) {
        fprintf(stderr, "native rehearsal began before configured window\n");
        return 1;
    }
    env.global_step = 92;
    if (!dogfight_two_agent_in_native_acquisition(&env)) {
        fprintf(stderr, "native rehearsal did not begin at configured window\n");
        return 1;
    }
    env.global_step = 96;
    if (dogfight_two_agent_in_native_acquisition(&env)) {
        fprintf(stderr, "native rehearsal crossed cycle boundary\n");
        return 1;
    }

    float left_actions[NUM_ATNS] = {0};
    float right_actions[NUM_ATNS] = {0};
    float left_wrong_actions[NUM_ATNS] = {0};
    float right_wrong_actions[NUM_ATNS] = {0};
    left_actions[2] = -1.0f;
    right_actions[2] = 1.0f;
    left_wrong_actions[2] = 1.0f;
    right_wrong_actions[2] = -1.0f;
    float left_correct = dogfight_two_agent_native_acquisition_reward(
        0.25f, left_actions, 1.0f, 0.1f);
    float left_wrong = dogfight_two_agent_native_acquisition_reward(
        0.25f, left_wrong_actions, 1.0f, 0.1f);
    float right_correct = dogfight_two_agent_native_acquisition_reward(
        -0.25f, right_actions, 1.0f, 0.1f);
    float right_wrong = dogfight_two_agent_native_acquisition_reward(
        -0.25f, right_wrong_actions, 1.0f, 0.1f);
    if (fabsf(left_correct - right_correct) > 1.0e-6f
            || fabsf(left_wrong - right_wrong) > 1.0e-6f
            || left_correct < 0.999f
            || left_wrong > -0.999f) {
        fprintf(stderr,
            "native acquisition reward is not mirrored: %.6f %.6f %.6f %.6f\n",
            left_correct,
            left_wrong,
            right_correct,
            right_wrong);
        return 1;
    }

    left_actions[0] = 1.0f;
    left_actions[1] = -1.0f;
    left_actions[3] = 1.0f;
    float unstable = dogfight_two_agent_native_acquisition_reward(
        0.25f, left_actions, 1.0f, 0.1f);
    if (fabsf(unstable - 0.7f) > 1.0e-6f) {
        fprintf(stderr,
            "native acquisition did not penalize unstable controls: %.6f\n",
            unstable);
        return 1;
    }
    float rehearsal = dogfight_two_agent_native_acquisition_reward(
        0.25f, left_actions, 1.0f, 0.0f);
    if (rehearsal < 0.999f) {
        fprintf(stderr,
            "native rehearsal constrained non-aileron controls: %.6f\n",
            rehearsal);
        return 1;
    }
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_magnitude_penalty();
    failures += test_mirrored_steering_alignment();
    failures += test_steering_alignment_decay();
    failures += test_independent_steering_publication();
    failures += test_native_acquisition_contract();
    if (failures == 0) {
        printf("two-agent aileron reward contract passed\n");
    }
    return failures;
}
